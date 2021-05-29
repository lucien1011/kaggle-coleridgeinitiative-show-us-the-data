import os
import sys
import glob
import pickle
import argparse
from collections import defaultdict

import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

fkey = "pred_ids"
of_mapping_key = "overflow_mapping"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('output_dir',type=str)
    return parser.parse_args()

def compute_extract_dict(cfg,device='cuda'):

    pp = cfg.pipeline 
    out_dict = {}
    checkpts = pp.get_model_checkpts(cfg.extract_cfg.model_dir,cfg.extract_cfg.model_key)
    textids = pd.read_csv(cfg.preprocess_cfg.test_csv_path).id
    for c in checkpts:
        cdir = os.path.join(cfg.extract_cfg.output_dir,c)
        if not os.path.exists(cdir):
            print(cdir+" not exist")
            continue
        fs = [f for f in os.listdir(cdir) if ".pt" in f]
        if not fs:
            print(cdir+" is empty, skipping")
            continue
        print("Compute predicted text dictionary for "+c)
        c_int = int(c.replace(cfg.extract_cfg.model_key,""))
        pred_ids = torch.cat([torch.load(os.path.join(cdir,f)) for f in fs if fkey in f])
        op_indices = torch.cat([torch.load(os.path.join(cdir,f)) for f in fs if of_mapping_key in f])
        pred_ids[pred_ids==-1] = 0
        strs = cfg.tokenizer.batch_decode(pred_ids,skip_special_tokens=True)

        for i in range(len(strs)):
            textid = str(textids[int(op_indices[i])])
            if c_int not in out_dict:
                out_dict[c_int] = defaultdict(str)
            out_dict[c_int][textid] += strs[i]
            if strs[i]: out_dict[c_int][textid] +=","
    
    return out_dict

def compute_article_coverage_from_dict(pred_text):

    out_dict = {cname:dict() for cname in pred_text}
    for cname,checkptd in pred_text.items():
        for c_int,textd in checkptd.items():
            out_dict[cname][c_int] = sum([bool(ts) for ts in textd.values()]) / len(textd)

    return out_dict

def compute_dataset_overlap_from_dict(pred_text,dataset_tokens):

    out_dict = {cname:dict() for cname in pred_text}
    for cname,checkptd in pred_text.items():
        for c_int,textd in checkptd.items():
            out_dict[cname][c_int] = sum([len([t for t in ts if t in dataset_tokens])/len(ts) for ts in textd.values() if ts]) / len([1 for ts in textd.values() if ts])

    return out_dict

def get_dataset_tokens(cfg):
    df = pd.read_csv(cfg.preprocess_cfg.train_csv_path)
    datasets = []
    for d in df.dataset.unique():
        datasets.extend(cfg.tokenizer.tokenize(d))
    return set(datasets)

def draw_figure(output_dir,name,input_dict):
    mkdir_p(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    print("Plotting "+name)
    for c,d in input_dict.items():
        x = list(d.keys())
        x.sort()
        y = [d[i] for i in x]
        
        ax.plot(x,y,label=c.plot_label)
        ax.set_ylabel(name)
        ax.set_xlabel("training step")
    plt.legend(loc='best')
    print("Saving "" to "+output_dir)
    fig.savefig(os.path.join(output_dir,name+"_vs_training_step.png"))

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
        print("*"*100)
        print("Processing "+c.name)
        print("*"*100)
        out_dict = compute_extract_dict(c)
        if out_dict:
            data[c] = out_dict
    cfgs = list(data.keys())

    assert len(data) > 0

    article_coverage_dict = compute_article_coverage_from_dict(data)
    draw_figure(args.output_dir,"article_coverage",article_coverage_dict)
    
    dataset_tokens = get_dataset_tokens(cfgs[0])
    dataset_overlap_dict = compute_dataset_overlap_from_dict(data,dataset_tokens)
    draw_figure(args.output_dir,"dataset_overlap",dataset_overlap_dict)
