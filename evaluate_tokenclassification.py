import os
import sys
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

# __________________________________________________________________ ||
inputs = cfg.pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
model = cfg.model.from_pretrained(cfg.extract_cfg.pretrain_model)

# __________________________________________________________________ ||
model = model.to(cfg.evaluate_cfg.device)

test_sampler = RandomSampler(inputs.test_dataset)
test_dataloader = DataLoader(inputs.test_dataset, sampler=test_sampler, batch_size=cfg.evaluate_cfg.batch_size)

with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        if step % cfg.evaluate_cfg.print_per_step == 0:
            batch = tuple(t.to(cfg.evaluate_cfg.device) for t in batch)
            batch_test = {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}
            with torch.no_grad():
                preds = model(**batch_test)
                d = cfg.pipeline.compute_metrics(preds,batch_test['labels'])
            print("*"*50)
            print("step ",step)
            print("*"*50)
            print(" | ".join(["{name} {value:4.4f}".format(name=name,value=value) for name,value in d.items()]))
