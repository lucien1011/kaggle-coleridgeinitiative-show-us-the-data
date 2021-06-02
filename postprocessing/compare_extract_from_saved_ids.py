import os
import sys
import glob
import argparse

import torch
from torch.utils.data import DataLoader
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

fkey = "pred_ids"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('output_dir',type=str)
    return parser.parse_args()

def compute_extract_attribute(pred_ids):
    npred_token = torch.sum(pred_ids!=-1,axis=1)
    nbatch = len(pred_ids)
    return {
        "number_pred_token": torch.sum(npred_token),
        "number_text_with_dataset": torch.sum(npred_token!=0) / nbatch,
    }

def compute_text_metric_dict(cfg,device='cuda'):

    pp = cfg.pipeline 
    out_dict = {}
    checkpts = pp.get_model_checkpts(cfg.extract_cfg.model_dir,cfg.extract_cfg.model_key)
    textids = pd.read_csv(cfg.preprocess_cfg.test_csv_path).id
    for c in checkpts:
        cdir = os.path.join(cfg.extract_cfg.output_dir,c)
        if not os.path.exists(cdir):
            print(cdir+" not exist")
            continue
        fs = [f for f in os.listdir(cdir) if ".pt" in f and fkey in f]
        if not fs:
            print(cdir+" is empty, skipping")
            continue
        #fs.sort(key=lambda x: int(x.replace(".pt","").split("_")[-1]))
        pred_ids = torch.cat([torch.load(os.path.join(cdir,f)) for f in fs])
        pp.print_message(c)
        extract_attributes = compute_extract_attribute(pred_ids)
        
        c_int = int(c.replace(cfg.extract_cfg.model_key,""))
        for name,value in extract_attributes.items():
            if name not in out_dict:
                out_dict[name] = {}
            out_dict[name][c_int] = value
    
    return out_dict

if __name__ == "__main__":

    args = parse_arguments()

    if "*" in args.input_path:
        paths = glob.glob(args.input_path)
    else:
        paths = args.input_path.split(",")
    output_dir = args.output_dir

    assert len(paths) > 0
    
    cfgs = [ObjDict.read_all_from_file_python3(p) for p in paths]
    assert all([hasattr(c,"plot_label",) for c in cfgs])
    
    data = {}
    for c in cfgs:
        out_dict = compute_text_metric_dict(c)
        if out_dict:
            data[c] = out_dict
    cfgs = list(data.keys())

    assert len(data) > 0
    assert all([list(data[c].keys()) == list(data[cfgs[0]].keys()) for c in cfgs[1:]])

    np = len(data[cfgs[0]])
    mkdir_p(output_dir)
    names = data[cfgs[0]].keys()
    fig, ax = plt.subplots(figsize=(10, 5))
    for ip,name in enumerate(names):
        print("Plotting "+name)
        fig, ax = plt.subplots(figsize=(10, 5))
        for c,d in data.items():
            x = list(d[name].keys())
            x.sort()
            y = [d[name][i] for i in x]
            
            ax.plot(x,y,label=c.plot_label)
            ax.set_ylabel(name)
            ax.set_xlabel("training step")
            ip += 1
        plt.legend(loc='best')
        print("Saving "+name+" to "+output_dir)
        fig.savefig(os.path.join(output_dir,name+"_vs_training_step.png"))
