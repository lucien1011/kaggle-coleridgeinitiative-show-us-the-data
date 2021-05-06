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
config = ObjDict(

    name = name,

    #input_seq_df = 'data/train.csv',
    #input_seq_df = 'data/train_hasDataset.csv',
    input_seq_df = 'data/train_hasDataset_lite.csv',
    columns_to_remove = [],
    
    pp = TokenClassifierPipeline(),
    train_test_split = 0.1,

)

# __________________________________________________________________ ||
config.input_dataset_dir = "data/huggingface/"
#config.input_dataset_name = 'optimise_tokenclassification_210506_01/'
#config.input_dataset_name = 'optimise_tokenclassification_210506_01_hasDataset/'
config.input_dataset_name = 'optimise_tokenclassification_210506_01_hasDataset_lite/'

# __________________________________________________________________ ||
config.model_checkpoint = model_checkpoint
config.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
config.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
config.train_args = TrainingArguments(
    output_dir=saved_model_path,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True,
    do_eval=True,
    num_train_epochs=10,
    weight_decay=0.01,
)
config.compute_metrics = compute_metrics
config.data_collator = DataCollatorForTokenClassification(config.tokenizer)

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

