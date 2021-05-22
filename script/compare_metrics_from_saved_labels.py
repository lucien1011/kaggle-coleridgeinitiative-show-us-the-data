import os
import sys

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

def compute_metric_dict(cfg,device='cuda'):

    pp = cfg.pipeline
    
    val_dataset_path = os.path.join(cfg.predict_cfg.output_dir,cfg.predict_cfg.dataset_save_name)
    val_dataset = torch.load(val_dataset_path)
    
    iterator = DataLoader(val_dataset,batch_size=len(val_dataset))
    batch = tuple(t.to(device) for t in iter(iterator).next())
    batch = {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}
    
    out_dict = {}
    checkpts = pp.get_model_checkpts(cfg.predict_cfg.model_dir,cfg.predict_cfg.model_key)
    for c in checkpts:
        cdir = os.path.join(cfg.predict_cfg.output_dir,c)
        if not os.path.exists(cdir):
            print(cdir+" not exist")
            continue
        fs = [f for f in os.listdir(cdir) if ".pt" in f]
        fs.sort(key=lambda x: int(x.replace(".pt","").split("_")[-1]))
        preds = torch.cat([torch.load(os.path.join(cdir,f)).logits for f in fs])
        metrics = pp.compute_metrics(preds,batch['labels'],cfg.nlabel,islogit=True)
        pp.print_message(c)
        
        c_int = int(c.replace(cfg.predict_cfg.model_key,""))
        for name,value in metrics.items():
            if name not in out_dict:
                out_dict[name] = {}
            out_dict[name][c_int] = value
    
    return out_dict

if __name__ == "__main__":

    paths = sys.argv[1].split(",")
    output_dir = sys.argv[2]
    cfgs = [ObjDict.read_all_from_file_python3(p) for p in paths]
    data = {c.name:compute_metric_dict(c) for c in cfgs}

    assert len(data) > 0
    assert all([list(data[c.name].keys()) == list(data[cfgs[0].name].keys()) for c in cfgs[1:]])

    ip = 0
    np = len(data[cfgs[0].name])
    mkdir_p(output_dir)
    names = data[cfgs[0].name].keys()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        print("Plotting "+name)
        fig, ax = plt.subplots(figsize=(10, 5))
        for c,d in data.items():
            x = list(d[name].keys())
            x.sort()
            y = [d[name][i] for i in x]
            
            ax.plot(x,y,label=c)
            ax.set_ylabel(name)
            ax.set_xlabel("training step")
            ip += 1
        plt.legend(loc='best')
        print("Saving "+name+" to "+output_dir)
        fig.savefig(os.path.join(output_dir,name+"_vs_training_step.png"))
