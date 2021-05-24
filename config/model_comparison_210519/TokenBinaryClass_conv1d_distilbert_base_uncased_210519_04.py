import os
import numpy as np
import torch

from transformers import DistilBertTokenizerFast

from model.DistilBertConv1d import DistilBertConv1dForTokenClassification,DistilBertConv1dConfig
from pipeline.pipeline_tokenmulticlassifier import TokenMultiClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "TokenBinaryClass_conv1d_distilbert_base_uncased_210519_04"
base_pretrained = "distilbert-base-uncased"

t2_dir = "/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/coleridgeinitiative-show-us-the-data/data/"
preprocess_train_dir = os.path.join(t2_dir,"TokenBinaryClass_conv1d_distilbert_base_uncased_210519_01","train/")
preprocess_test_dir = os.path.join(t2_dir,"TokenBinaryClass_conv1d_distilbert_base_uncased_210519_01","test/")

label_list = range(2)
nlabel = len(label_list)

# __________________________________________________________________ ||
pipeline = TokenMultiClassifierPipeline()

config = DistilBertConv1dConfig(
        num_labels = nlabel,
        conv_setting = [
            {"in_channels":768, "out_channels":16, "kernel_size":15,},
            {"in_channels":16, "out_channels":nlabel, "kernel_size":15,},
            ],
        )
model = DistilBertConv1dForTokenClassification.from_pretrained(
    'model/'+base_pretrained,
    config=config,
    )

tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/'+base_pretrained)

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
        labels_name = "labels.pt",
        overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
        )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 8,
        per_gpu_train_batch_size = 1,
        val_batch_size = 128,
        num_train_epochs = 10,
        learning_rate = 2e-5,
        betas=(0.9,0.999),
        adam_epsilon = 1e-9,
        weight_decay = 1e-4,
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
predict_cfg = ObjDict(
        model_dir = os.path.join('log',name,),
        model_key = 'checkpoint-epoch-',
        device = "cuda",
        batch_size = 256,
        output_dir = os.path.join(t2_dir,name,"pred/"),
        pred_name = "labels",
        pred_extension = ".pt",
        dataset_fraction = 0.2,
        dataset_save_name = "validation_dataset.pt",
        )

# __________________________________________________________________ ||
evaluate_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-8000"),
        device = 'cuda',
        batch_size = 256,
        test = True,
        print_per_step = 1,
        dataset_tokens_path = "data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/dataset_tokens.pt",
        )

# __________________________________________________________________ ||
extract_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-epoch-2"),
        device = 'cuda',
        test = True,
        write_predicted_only = True,
        write_per_step = 1,
        extract_text_path = os.path.join('log',name,'checkpoint-epoch-2.txt'),
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
