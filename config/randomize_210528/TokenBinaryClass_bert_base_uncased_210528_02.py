import os
import numpy as np
import torch

from transformers import BertForTokenClassification,BertTokenizerFast,BertConfig,BertModel

from pipeline.pipeline_tokenmulticlassifier import TokenMultiClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
base_dir = "randomize_210528"
name = "TokenBinaryClass_bert_base_uncased_210528_02"
base_pretrained = "bert-base-uncased"
plot_label = "bert-base-uncased-conv1d-4layer"

t2_dir = "/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/coleridgeinitiative-show-us-the-data/data/"
preprocess_train_dir = os.path.join(t2_dir,"TokenBinaryClass_bert_base_uncased_210522_01","train/")
preprocess_test_dir = os.path.join(t2_dir,"rcdataset_210526","TokenBinaryClass_bert_base_uncased_210522_01","test/")

label_list = range(2)
nlabel = len(label_list)

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
        test_csv_path = 'storage/input/rcdataset/df_rcdataset.csv',
        preprocess_test_dir = preprocess_test_dir,
        input_ids_name = "input_ids.pt",
        attention_mask_name = "attention_mask.pt",
        labels_name = "labels.pt",
        overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
        )

# __________________________________________________________________ ||
randomize_cfg = ObjDict(
        fraction = 0.1,
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
        output_dir = os.path.join('log/',base_dir,name),
        max_steps = 999999999.,
        n_gpu = 0,
        logging_steps = 100,
        )

# __________________________________________________________________ ||
predict_cfg = ObjDict(
        model_dir = os.path.join('log',base_dir,name,),
        model_key = 'checkpoint-epoch-',
        device = "cuda",
        batch_size = 128,
        output_dir = os.path.join(t2_dir,base_dir,name,"pred/"),
        pred_name = "labels",
        pred_extension = ".pt",
        val_dataset_name = "val_dataset",
        )

# __________________________________________________________________ ||
extract_cfg = ObjDict(
        model_dir = os.path.join('log',name,),
        model_key = 'checkpoint-epoch-',
        device = "cuda",
        batch_size = 256,
        output_dir = os.path.join(t2_dir,base_dir,name,"extract/"),
        extract_file_name = "pred_ids.pt",
        dataset_name = "test_dataset",
        )

# __________________________________________________________________ ||
evaluate_cfg = ObjDict(
        pretrain_model = os.path.join("log",base_dir,name,"checkpoint-8000"),
        device = 'cuda',
        batch_size = 128,
        test = True,
        print_per_step = 1,
        dataset_tokens_path = "data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/dataset_tokens.pt",
        )

# __________________________________________________________________ ||
slurm_job_dir = os.path.join('log/',base_dir,name+"/")
slurm_cfg = ObjDict(
    name = name,
    slurm_cfg_name = 'submit.cfg',
    slurm_job_dir = slurm_job_dir,
    memory = '32gb',
    email = 'kin.ho.lo@cern.ch',
    time = '72:00:00',
    gpu = 'geforce:1',
    )
