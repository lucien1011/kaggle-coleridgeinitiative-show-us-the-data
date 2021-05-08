import os
import sys
import torch

from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchmetrics.functional import accuracy,auroc,f1,precision,recall

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

pretrain_model = "log/optimise_tokenclassification_210507_01_hasDataset/checkpoint-16400/config.json"
device = 'cuda'
batch_size = 512

# __________________________________________________________________ ||
inputs = cfg.pipeline.load_preprocess_data(cfg.preprocess_cfg)
model_config = AutoConfig.from_pretrained(pretrain_model)
model = AutoModelForTokenClassification.from_config(model_config)

# __________________________________________________________________ ||
model = model.to(device)

test_sampler = RandomSampler(inputs.test_dataset)
test_dataloader = DataLoader(inputs.test_dataset, sampler=test_sampler, batch_size=128)

softmax = torch.nn.Softmax(dim=-1)
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        if step % 10 == 0:
            batch = tuple(t.to(device) for t in batch)
            batch_test = {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}
            with torch.no_grad():
                preds = model(**batch_test)
            ntoken = sum(batch_test['attention_mask'])
            probs = softmax(preds.logits)[:,:,1].flatten()
            labels_flatten = batch_test['labels'].flatten()
            d = {
                "accuracy": accuracy(probs,labels_flatten),
                "auroc": auroc(probs,labels_flatten),
                "f1": f1(probs,labels_flatten),
                "precision": precision(probs,labels_flatten),
                "recall": recall(probs,labels_flatten),
            }
            print("*"*50)
            print("step ",step)
            print("*"*50)
            print(" | ".join(["{name} {value:4.2f}".format(name=name,value=value) for name,value in d.items()]))
            #print((probs > 0.5).nonzero(as_tuple=True))
            #print(batch_test['labels'].nonzero(as_tuple=True))
