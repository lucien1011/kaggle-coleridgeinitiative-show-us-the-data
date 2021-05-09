import os
import sys
import torch

from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pipeline.pipeline_tokenclassifier import compute_metrics
from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

pretrain_model = "log/optimise_tokenclassification_210508_01_hasDataset/checkpoint-8200/"
device = 'cuda'
batch_size = 512

# __________________________________________________________________ ||
inputs = cfg.pipeline.load_preprocess_data(cfg.preprocess_cfg)
model = AutoModelForTokenClassification.from_pretrained(pretrain_model,config=AutoConfig.from_pretrained(cfg.base_pretrained))

# __________________________________________________________________ ||
model = model.to(device)

test_sampler = RandomSampler(inputs.test_dataset)
test_dataloader = DataLoader(inputs.test_dataset, sampler=test_sampler, batch_size=256)

softmax = torch.nn.Softmax(dim=-1)
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        if step % 10 == 0:
            batch = tuple(t.to(device) for t in batch)
            batch_test = {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}
            with torch.no_grad():
                preds = model(**batch_test)
                d = compute_metrics(preds,batch_test['labels'])
            print("*"*50)
            print("step ",step)
            print("*"*50)
            print(" | ".join(["{name} {value:4.4f}".format(name=name,value=value) for name,value in d.items()]))
            #print((probs > 0.5).nonzero(as_tuple=True))
            #print(batch_test['labels'].nonzero(as_tuple=True))
