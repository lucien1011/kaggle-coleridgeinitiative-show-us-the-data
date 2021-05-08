import os
import numpy as np

from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

from pipeline.pipeline_tokenclassifier import TokenClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_tokenclassification_210507_01_hasDataset"
model_checkpoint = "distilbert-base-uncased"
label_list = [0,1]

# __________________________________________________________________ ||
def accuracy(preds,labels):
    preds = torch.argmax(preds.logits,axis=2)
    match = (preds==1)*(batch['labels']==1)
    return torch.sum(match) / torch.sum(labels==1)

pipeline = TokenClassifierPipeline()

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
    model_checkpoint = model_checkpoint,
    input_csv_path = "data/train_hasDataset.csv",
    #input_dataset_dir = "data/huggingface/",
    #input_dataset_name = 'optimise_tokenclassification_210506_01/'
    #input_dataset_name = 'optimise_tokenclassification_210506_01_hasDataset/'
    #input_dataset_name = 'optimise_tokenclassification_210506_01_hasDataset_lite/',
    train_size = 0.8,
    val_size = 0.1,
    )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 8,
        per_gpu_train_batch_size = 8,
        val_batch_size = 8,
        num_train_epochs = 3,
        learning_rate = 2e-5,
        adam_epsilon = 1e-9,
        warmup_steps = 1,
        gradient_accumulation_steps = 1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = 9999.,
        save_steps = 100,
        output_dir = os.path.join('log/',name),
        max_steps = 999999999.,
        weight_decay = 0.01,
        n_gpu = 0,
        logging_steps = 100,
        compute_metric = accuracy,
        )

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# __________________________________________________________________ ||
slurm_job_dir = os.path.join('job/',name+'/')
slurm_cfg = ObjDict(
    name = name,
    slurm_cfg_name = 'submit.cfg',
    slurm_job_dir = slurm_job_dir,
    slurm_commands = """echo \"{job_name}\"
cd {base_path}
source setup_hpg.sh
python3 {pyscript} {cfg_path}
""".format(
            job_name=name,
            pyscript="run_pipeline.py",
            cfg_path="config/"+name+".py",
            base_path=os.environ['BASE_PATH'],
            output_path=slurm_job_dir,
            ),
    )
