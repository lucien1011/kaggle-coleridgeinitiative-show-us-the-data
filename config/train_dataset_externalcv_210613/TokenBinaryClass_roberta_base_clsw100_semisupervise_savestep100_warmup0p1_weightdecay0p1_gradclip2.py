import os
import numpy as np
import torch

from model.CustomRobertaForTokenClassification import CustomRobertaForTokenClassification
from transformers import RobertaTokenizerFast,RobertaConfig,RobertaModel,RobertaConfig

from pipeline.pipeline_tokenmulticlassifier import TokenMultiClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
base_dir = "train_dataset_externalcv_210613"
name = "TokenBinaryClass_roberta_base_clsw100_semisupervise_savestep100_warmup0p1_weightdecay0p1_gradclip2_iter2"
base_pretrained = "roberta-base"

t2_dir = "/cmsuf/data/store/user/t2/users/klo/MiscStorage/ForLucien/Kaggle/coleridgeinitiative-show-us-the-data/preprocess_data/"

#preprocess_train_dir = os.path.join(t2_dir,"train_dataset_externalcv_210613_TokenBinaryClass_roberta_base_210612_checkpt5100","TokenBinaryClass_roberta_base","train/")
preprocess_train_dir = os.path.join(t2_dir,"train_dataset_externalcv_210613_TokenBinaryClass_roberta_base_210612_checkpt5100","TokenBinaryClass_roberta_base","train_filter_by_dataset/")
preprocess_test_dir = os.path.join(t2_dir,"train_dataset_externalcv_210612","TokenBinaryClass_roberta_base","test/")
preprocess_val_dir = os.path.join(t2_dir,"train_dataset_externalcv_210612","TokenBinaryClass_roberta_base","val/")

result_dir = "/blue/avery/kinho.lo/kaggle-coleridgeinitiative-show-us-the-data/storage/results/"

label_list = range(2)
nlabel = len(label_list)
cls1w = 100

assert cls1w > 1.
class_weight = [1./(1.-1/cls1w),cls1w,]

# __________________________________________________________________ ||
pipeline = TokenMultiClassifierPipeline()

model_args = dict(num_labels=len(label_list),output_hidden_states=False,use_all_attention=False,class_weight=class_weight)
model = CustomRobertaForTokenClassification.from_pretrained('model/'+base_pretrained,**model_args)

tokenizer = RobertaTokenizerFast.from_pretrained('tokenizer/'+base_pretrained)

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
        tokenizer = tokenizer,
        train_csv_path = "storage/results/train_dataset_externalcv_210612/TokenBinaryClass_roberta_base_clsw100_semisupervise_savestep100_warmup0p1_weightdecay0p1_gradclip2/checkpoint-step-5100/train_pred_summary.csv",
        preprocess_train_dir = preprocess_train_dir,
        test_csv_path = 'storage/input/pack_data_210610/test_sequence.csv',
        preprocess_test_dir = preprocess_test_dir,
        val_csv_path = "storage/input/pack_data_210610/val_sequence.csv",
        preprocess_val_dir = preprocess_val_dir,

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
randomize_cfg = ObjDict(
        fraction = 0.0,
        )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        train_batch_size = 4,
        per_gpu_train_batch_size = 1,
        val_batch_size = 8,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        betas=(0.9,0.999),
        adam_epsilon = 1e-9,
        weight_decay = 0.1,
        warmup_steps = 0.1,
        gradient_accumulation_steps = 1,
        seed = 1,
        device = 'cuda',
        max_grad_norm = 2.,
        save_steps = 100,
        output_dir = os.path.join(result_dir,base_dir,name),
        max_steps = 999999999.,
        n_gpu = 0,
        logging_steps = 100,
        )

# __________________________________________________________________ ||
predict_cfg = ObjDict(
        model_dir = os.path.join(result_dir,base_dir,name,),
        model_key = 'checkpoint-epoch-',
        device = "cuda",
        batch_size = 256,
        output_dir = os.path.join(result_dir,base_dir,name,"pred/"),
        pred_name = "labels",
        pred_extension = ".pt",
        val_dataset_name = "val_dataset",
        )

# __________________________________________________________________ ||
extract_cfg = ObjDict(
        model_dir = os.path.join(result_dir,base_dir,name,),
        model_key = 'checkpoint-epoch-',
        device = "cuda",
        batch_size = 256,
        output_dir = os.path.join(result_dir,base_dir,name,"extract/"),
        extract_file_name = "pred_ids.pt",
        val_dataset_name = "val_dataset",
        )

# __________________________________________________________________ ||
evaluate_cfg = ObjDict(
        model_dir = os.path.join(result_dir,base_dir,name,),
        model_key = 'checkpoint-epoch-',
        device = "cuda",
        batch_size = 64,
        n_sample = 100,
        output_dir = os.path.join(result_dir,base_dir,name,"extract/"),
        extract_file_name = "pred_ids.pt",
        val_dataset_name = "val_dataset",
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
