import os
import numpy as np
import torch

from transformers import BertForTokenClassification,BertTokenizerFast,BertConfig,BertModel

from pipeline.pipeline_tokenmulticlassifier import TokenMultiClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "TokenMultiClass_bert_base_uncased_210519_01"
base_pretrained = "bert-base-uncased"

t2_dir = "/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/coleridgeinitiative-show-us-the-data/data/"
preprocess_train_dir = os.path.join(t2_dir,name,"train/")
preprocess_test_dir = os.path.join(t2_dir,name,"test/")

label_list = range(4)

# __________________________________________________________________ ||
pipeline = TokenMultiClassifierPipeline()

model = BertForTokenClassification.from_pretrained('model/'+base_pretrained,num_labels=len(label_list))

tokenizer = BertTokenizerFast.from_pretrained('tokenizer/'+base_pretrained)

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
        train_csv_path = "data/train_sequence_has_dataset.csv",
        train_size = 0.8,
        val_size = 0.1,
        tokenizer = tokenizer,
        preprocess_train_dir = preprocess_train_dir,
        load_preprocess = True,
        test_csv_path = 'data/test_sequence.csv',
        preprocess_test_dir = preprocess_test_dir,
        input_ids_name = "input_ids.pt",
        attention_mask_name = "attention_mask.pt",
        labels_name = "multilabels.pt",
        overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
        )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 8,
        per_gpu_train_batch_size = 1,
        val_batch_size = 128,
        num_train_epochs = 5,
        learning_rate = 2e-5,
        betas=(0.9,0.999),
        adam_epsilon = 1e-9,
        weight_decay = 0.00,
        warmup_steps = 0.1,
        gradient_accumulation_steps = 1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = 9999.,
        save_steps = 0,
        output_dir = os.path.join('log/',name),
        max_steps = 999999999.,
        n_gpu = 0,
        logging_steps = 100,
        )

# __________________________________________________________________ ||
evaluate_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-8000"),
        device = 'cuda',
        batch_size = 128,
        test = True,
        print_per_step = 1,
        dataset_tokens_path = "data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/dataset_tokens.pt",
        )

# __________________________________________________________________ ||
extract_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-epoch-4"),
        device = 'cuda',
        test = True,
        write_predicted_only = True,
        write_per_step = 1,
        extract_text_path = os.path.join('log',name,'checkpoint-epoch-4.txt'),
        )

# __________________________________________________________________ ||
slurm_job_dir = os.path.join('log/',name+'/')
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
    memory = '32gb',
    email = 'kin.ho.lo@cern.ch',
    time = '72:00:00',
    gpu = 'geforce:1',
    )