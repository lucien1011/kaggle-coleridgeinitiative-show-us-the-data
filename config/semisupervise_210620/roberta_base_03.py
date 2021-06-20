import os
import numpy as np
import torch

from model.CustomRobertaForTokenClassification import CustomRobertaForTokenClassification
from transformers import RobertaTokenizerFast,RobertaConfig,RobertaModel,RobertaConfig

from pipeline.Preprocessor import Preprocessor 
from pipeline.CustomTrainer import CustomTrainer 
from utils.objdict import ObjDict

# __________________________________________________________________ ||
base_dir = "semisupervise_210620"
name = "roberta_base_03"
base_pretrained = "roberta-base"

t2_dir = "/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/coleridgeinitiative-show-us-the-data/preprocess_data/"

preprocess_train_dir = os.path.join(t2_dir,"semisupervise_210618","roberta_base_01","train/")
#preprocess_train_dir = os.path.join(t2_dir,"semisupervise_210618","roberta_base_01","train_filter_by_train_labels/")
preprocess_val_dir = os.path.join(t2_dir,"semisupervise_210618","roberta_base_01","val/")
preprocess_test_dir = os.path.join(t2_dir,"semisupervise_210618","roberta_base_01","test/")

result_dir = "/blue/avery/kinho.lo/kaggle/kaggle-coleridgeinitiative-show-us-the-data/storage/results/"

label_list = range(2)
nlabel = len(label_list)
cls1w = 100

assert cls1w > 1.
class_weight = [1./(1.-1/cls1w),cls1w,]

# __________________________________________________________________ ||
preprocessor = Preprocessor()
trainer = CustomTrainer()

model_args = dict(num_labels=len(label_list),output_hidden_states=False,use_all_attention=False,class_weight=class_weight)
model = CustomRobertaForTokenClassification.from_pretrained('model/'+base_pretrained,**model_args).to('cuda')

tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer/'+base_pretrained)

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
        tokenizer = tokenizer,
        train_csv_path = 'storage/input/pack_data_210618/train_sequence.csv',
        preprocess_train_dir = preprocess_train_dir,
        val_csv_path = 'storage/input/pack_data_210618/test_sequence.csv',
        preprocess_val_dir = preprocess_val_dir,
        test_csv_path = 'storage/input/pack_data_210618/val_sequence.csv',
        preprocess_test_dir = preprocess_test_dir,

        input_ids_name = "input_ids.pt",
        attention_mask_name = "attention_mask.pt",
        labels_name = "labels.pt",
        pred_labels_name = "pred_labels.pt",
        dataset_masks_name = "dataset_masks.pt",
        offset_mapping_name = "offset_mapping.pt",
        overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
        sample_weight_name = "sample_weight.pt",
        
        create_by_offset_mapping = True,

        tokenize_args = dict(padding='max_length',max_length=512,truncation=True,return_overflowing_tokens=True,return_offsets_mapping=True,return_tensors='pt',)
        )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 4,
        val_batch_size = 128,
        num_train_epochs = 1,
        warmup_steps = 0.1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = -1,
        save_steps = 1000,
        max_steps = 00,
        output_dir = os.path.join(result_dir,base_dir,name),
        logging_steps = 500,
        optimizer_type = "AdamW",
        optimizer_args = ObjDict(
            lr = 5e-6,
            betas = (0.9,0.999),
            eps = 1e-9,
            weight_decay = 0.01,
            no_decay = ["bias","LayerNorm.weight"],
            ),
        scheduler_type = "get_linear_schedule_with_warmup",
        sampler_type = "RandomSampler",
        niter = 10,
        thresholds = [0.999,0.999,0.99,0.99,0.99,0.97,0.97,0.97,0.97,0.95],
        matched_datasets = False,
        tokenizer=tokenizer,
        )

# __________________________________________________________________ ||
calculate_score_cfg = ObjDict(
        model_dir = os.path.join(result_dir,base_dir,name,),
        model_key = 'checkpoint-epoch-2',
        device = "cuda",
        batch_size = 128,
        )

# __________________________________________________________________ ||
slurm_job_dir = os.path.join(result_dir,base_dir,name+"/")
slurm_cfg = ObjDict(
        name = name,
        slurm_cfg_name = 'submit.cfg',
        slurm_job_dir = slurm_job_dir,
        memory = '32gb',
        email = 'kin.ho.lo@cern.ch',
        time = '72:00:00',
        gpu = 'quadro',
        )
