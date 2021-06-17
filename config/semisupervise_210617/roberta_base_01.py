import os
import numpy as np
import torch

from model.CustomRobertaForTokenClassification import CustomRobertaForTokenClassification
from transformers import RobertaTokenizerFast,RobertaConfig,RobertaModel,RobertaConfig
from transformers import AdamW

from pipeline.Preprocessor import Preprocessor 
from pipeline.CustomTrainer import CustomTrainer 
from utils.objdict import ObjDict

# __________________________________________________________________ ||
base_dir = "semisupervise_210617"
name = "roberta_base_01"
base_pretrained = "roberta-base"

t2_dir = "/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/coleridgeinitiative-show-us-the-data/preprocess_data/"

#preprocess_train_dir = os.path.join(t2_dir,base_dir,name,"train/")
preprocess_train_dir = os.path.join(t2_dir,base_dir,name,"train_filter_by_train_labels/")
preprocess_test_dir = os.path.join(t2_dir,base_dir,name,"test/")

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
model = CustomRobertaForTokenClassification.from_pretrained('model/'+base_pretrained,**model_args)

tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer/'+base_pretrained)

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
        tokenizer = tokenizer,
        train_csv_path = 'storage/input/pack_data_210610/train_sequence.csv',
        preprocess_train_dir = preprocess_train_dir,
        test_csv_path = 'storage/input/pack_data_210610/val_sequence.csv',
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
        val_batch_size = 8,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        betas=(0.9,0.999),
        adam_epsilon = 1e-9,
        weight_decay = 0.1,
        warmup_steps = 0.1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = 2.,
        save_steps = 100,
        output_dir = os.path.join(result_dir,base_dir,name),
        logging_steps = 100,
        no_decay = ["bias","LayerNorm.weight"],
        scheduler_type = "get_linear_schedule_with_warmup",
        sampler_type = "RandomSampler",
        niter = 2,
        threshold = 0.95,
        )
optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in train_cfg.no_decay)],"weight_decay": train_cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in train_cfg.no_decay)], "weight_decay": 0.0},
        ]
train_cfg.optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=train_cfg.learning_rate, 
        betas=train_cfg.betas,
        eps=train_cfg.adam_epsilon,
        weight_decay=train_cfg.weight_decay,
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
