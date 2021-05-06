import os
import json
import sys
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer

from utils.preprocessing import json_to_text,json_to_list
from utils.progressbar import progressbar
from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
def make_label(input_ids,dataset_ids,length):
    start_index = 1
    dataset_length = len(dataset_ids)-2
    while not all([input_ids[start_index+i] == dataset_ids[i] for i in range(dataset_length)]):
        start_index += 1
        
    output = [0 for _ in range(length)]
    for i in range(dataset_length):
        output[start_index+i] = 1
    return output

def convert_type(input_dict):
    
    tokenized_inputs = tokenizer(input_dict['sentence'],truncation=True,padding=True)
    
    labels = []
    for i,dataset in enumerate(input_dict['dataset']):
        tokenized_dataset = tokenizer(dataset)
        
        labels.append(make_label(tokenized_inputs.input_ids[i],tokenized_dataset['input_ids'][1:-1],sum(tokenized_inputs.attention_mask[i])))
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# __________________________________________________________________ ||
cfg = ObjDict.read_from_file_python3(sys.argv[1])
verbose = True

# __________________________________________________________________ ||
datasets = load_dataset("csv",data_files=[cfg.input_seq_df,])
dataset = datasets['train'].remove_columns("Unnamed: 0.1")
dataset = dataset.remove_columns("token")
dataset = dataset.remove_columns("index")
dataset = dataset.remove_columns("label")
dataset = dataset.remove_columns("hasDataset")

# __________________________________________________________________ ||
tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint)

# __________________________________________________________________ ||
processed_dataset = dataset.map(convert_type,batched=True)
dataset_dict = processed_dataset.train_test_split(test_size=0.1)

# __________________________________________________________________ ||
mkdir_p(cfg.input_dataset_dir)
dataset_dict.save_to_disk(os.path.join(cfg.input_dataset_dir,cfg.input_dataset_name))
