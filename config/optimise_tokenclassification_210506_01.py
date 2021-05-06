import os
import numpy as np

from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

from pipeline.pipeline_tokenclassifier import TokenClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_tokenclassification_210506_01_hasDataset_lite"
model_checkpoint = "distilbert-base-uncased"
saved_model_path = os.path.join('saved_model/',name,)
label_list = [0,1]
batch_size = 16

# __________________________________________________________________ ||
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    return {
        "accuracy": np.sum((predictions==1)*(labels==1)) / np.sum(labels==1),
    }

# __________________________________________________________________ ||
begin_config = ObjDict(
    input_dataset_dir = "data/huggingface/",
    #input_dataset_name = 'optimise_tokenclassification_210506_01/'
    #input_dataset_name = 'optimise_tokenclassification_210506_01_hasDataset/'
    input_dataset_name = 'optimise_tokenclassification_210506_01_hasDataset_lite/',
    train_test_split = 0.1,
    )

# __________________________________________________________________ ||
train_config = ObjDict(
        train_batch_size = 16,
        per_gpu_train_batch_size = 16,
        num_train_epochs = 10,
        learning_rate = 2e-5,
        adam_epsilon = 1e-9,
        warmup_steps = 1,
        gradient_accumulation_steps = 1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = 9999.,
        save_steps = 1,
        output_dir = './',
        max_steps = 999999999.,
        
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
        )

config.model_checkpoint = model_checkpoint
config.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

config.saved_model_path = saved_model_path

# __________________________________________________________________ ||
config.slurm_cfg_name = 'submit.cfg'
config.slurm_job_dir = os.path.join('job/',config.name+'/')
config.slurm_commands = """echo \"{job_name}\"
cd {base_path}
source setup_hpg.sh
python3 {pyscript} {cfg_path}
""".format(
            job_name=config.name,
            pyscript="run_pipeline.py",
            cfg_path="config/"+config.name+".py",
            base_path=os.environ['BASE_PATH'],
            output_path=config.slurm_job_dir,
            )

