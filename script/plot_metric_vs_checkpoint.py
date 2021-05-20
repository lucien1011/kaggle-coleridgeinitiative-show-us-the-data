import os
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
device = 'cuda'
plot_per_checkpt = 1
replace_str = "checkpoint-epoch-"

# __________________________________________________________________ ||
def compute_metrics(model,inputs,device,pipeline):
   
    val_sampler = RandomSampler(inputs.val_dataset)
    val_dataloader = DataLoader(inputs.val_dataset, sampler=val_sampler, batch_size=cfg.evaluate_cfg.batch_size)
    batch_val = tuple(t.to(device) for t in iter(val_dataloader).next())
    batch_val = {"input_ids": batch_val[0],"attention_mask": batch_val[1],"labels": batch_val[2]}
    with torch.no_grad():
        preds_val = model(**batch_val)
        metrics_val = pipeline.compute_metrics(preds_val,batch_val['labels'],num_classes=model.num_labels)
    return metrics_val

# __________________________________________________________________ ||
if __name__ == "__main__":

    cfg = ObjDict.read_all_from_file_python3(sys.argv[1])
    pipeline = cfg.pipeline

    print("Loading inputs")
    inputs = pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
    print("Finsih loading inputs")
 
    out_dict = {}
    checkpts = [c for c in os.listdir(cfg.train_cfg.output_dir) if replace_str in c]
    checkpts.sort()
    for i in tqdm(range(len(checkpts))):
        
        if i % plot_per_checkpt != 0: continue
        c = checkpts[i]
        
        tqdm.write("Processing checkpoint "+c)
        model = cfg.model.from_pretrained(os.path.join(cfg.train_cfg.output_dir,c)).to(device)
        metrics = compute_metrics(model,inputs,device,pipeline)
        tqdm.write("Finish processing checkpoint "+c)
        
        c_int = int(c.replace(replace_str,""))
        for name,value in metrics.items():
            if name not in out_dict:
                out_dict[name] = {}
            out_dict[name][c_int] = value

    ip = 0
    np = len(out_dict)
    fig, ax = plt.subplots(np,1,figsize=(15, 8*np))
    for name,mdict in out_dict.items():
        x = list(mdict.keys())
        x.sort()
        y = [mdict[i] for i in x]
        
        ax[ip].plot(x,y)
        ax[ip].set_ylabel(name)
        ax[ip].set_xlabel("training step")

        ip += 1
        
    mkdir_p(cfg.train_cfg.output_dir)
    fig.savefig(os.path.join(cfg.train_cfg.output_dir,"metric_vs_training_step.png"))
