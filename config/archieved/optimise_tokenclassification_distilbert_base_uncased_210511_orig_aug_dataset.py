import os
import numpy as np
import torch

from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

from pipeline.pipeline_tokenclassifier import TokenClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_tokenclassification_distilbert_base_uncased_210511_orig_aug_dataset"
base_pretrained = "distilbert-base-uncased"
preprocess_train_dir = "data/optimise_tokenclassification_distilbert_base_uncased_210511_orig_aug_dataset/train/" 
preprocess_test_dir = "data/optimise_tokenclassification_distilbert_base_uncased_210511_orig_aug_dataset/test/" 
label_list = [0,1]

# __________________________________________________________________ ||
pipeline = TokenClassifierPipeline()

model = AutoModelForTokenClassification.from_pretrained(base_pretrained)
tokenizer = AutoTokenizer.from_pretrained(base_pretrained)

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
    train_csv_path = "data/train_sequence_orig_aug_dataset.csv",
    train_size = 0.8,
    val_size = 0.1,
    tokenizer = tokenizer,
    preprocess_train_dir = preprocess_train_dir,
    load_preprocess = True,
    test_csv_path = 'data/test_sequence.csv',
    preprocess_test_dir = preprocess_test_dir,
    )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 16,
        per_gpu_train_batch_size = 1,
        val_batch_size = 128,
        num_train_epochs = 3,
        learning_rate = 2e-5,
        adam_epsilon = 1e-9,
        warmup_steps = 0.1,
        gradient_accumulation_steps = 1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = 9999.,
        save_steps = 1000,
        output_dir = os.path.join('log/',name),
        max_steps = 999999999.,
        weight_decay = 0.01,
        n_gpu = 0,
        logging_steps = 100,
        )

# __________________________________________________________________ ||
evaluate_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-40000"),
        device = 'cuda',
        batch_size = 256,
        test = True,
        output_text_path = os.path.join('tmp',name+'_test.txt'),
        extract_text_path = os.path.join('tmp',name+'_extract.txt'),
        )

# __________________________________________________________________ ||
extract_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-40000"),
        device = 'cpu',
        test = True,
        write_predicted_only = True,
        write_per_step = 1,
        extract_text_path = os.path.join('log',name,'extract_test_40000.txt'),
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
