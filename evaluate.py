import os
import sys
import torch

from transformers import AutoModelForTokenClassification
from datasets import load_from_disk

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

cfg = ObjDict.read_from_file_python3(sys.argv[1])

device = 'cuda'
print('Device',device)

processed_dataset = load_from_disk(os.path.join(cfg.input_dataset_dir,cfg.input_dataset_name))
dataset_dict = processed_dataset['train'].train_test_split(
        #test_size=cfg.train_test_split
        test_size=0.5,
        )

model = AutoModelForTokenClassification.from_pretrained(cfg.saved_model_path)

labels = torch.tensor([d+[-100 for _ in range(512-len(d))] for d in dataset_dict['test']['labels']])

inputs = torch.tensor(dataset_dict['test']['input_ids'])

params = model.parameters()

model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)

with torch.no_grad():
    preds = model(inputs)
preds = torch.argmax(preds.logits,axis=2)
match = (preds==1)*(labels==1)
print("Acc: ",torch.sum(match)/torch.sum(labels==1))
