import os
import numpy as np
import torch

from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

from pipeline.pipeline_tokenclassifier import TokenClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_tokenclassification_210509_01_hasDataset"
base_pretrained = "distilbert-base-uncased"
pretrain_model = "log/optimise_tokenclassification_210507_01_hasDataset/checkpoint-16400/"
pretrain_tokenizer = "log/optimise_tokenclassification_210507_01_hasDataset/tokenizer/" 
save_preprocess_path = "data/optimise_tokenclassification_210508_01_hasDataset/" 
label_list = [0,1]

# __________________________________________________________________ ||
pipeline = TokenClassifierPipeline()

model = AutoModelForTokenClassification.from_pretrained(pretrain_model,config=AutoConfig.from_pretrained(base_pretrained))
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer,config=AutoConfig.from_pretrained(base_pretrained))

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
    input_csv_path = "data/train_hasDataset.csv",
    train_size = 0.8,
    val_size = 0.1,
    tokenizer = tokenizer,
    save_preprocess_path = save_preprocess_path,
    load_preprocess = True,
    )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 16,
        per_gpu_train_batch_size = 8,
        val_batch_size = 128,
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
        )

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
