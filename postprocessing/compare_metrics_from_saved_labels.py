import os
import sys
import glob
import argparse

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

fkey = "labels"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('output_dir',type=str)
    parser.add_argument('-d','--dataset_name',type=str,default="train:val_dataset",)
    parser.add_argument('--ymin',type=float,default=0.,)
    parser.add_argument('--ymax',type=float,default=1.,)
    return parser.parse_args()

def compute_metric_dict(cfg,device='cuda',dataset_name="train:val_dataset"):

    pp = cfg.pipeline
    print("Processing: "+cfg.name)
    print("Use dataset: "+dataset_name)

    dataset_info = dataset_name.split(":")
    assert len(dataset_info) == 2
    mode,dataset_name = dataset_info
    assert mode == "train" or mode == "test"
    
    if mode == "train":
        inputs = pp.load_preprocess_train_data(cfg.preprocess_cfg)
    elif mode == "test":
        inputs = pp.load_preprocess_test_data(cfg.preprocess_cfg)
    val_dataset = getattr(inputs,dataset_name)
    
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
        fs = [f for f in os.listdir(cdir) if ".pt" in f and fkey in f]
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

    args = parse_arguments()

    if "*" in args.input_path:
        paths = glob.glob(args.input_path)
    else:
        paths = args.input_path.split(",")
    output_dir = args.output_dir
    dataset_name = args.dataset_name

    assert len(paths) > 0
    
    cfgs = [ObjDict.read_all_from_file_python3(p) for p in paths]
    assert all([hasattr(c,"plot_label") for c in cfgs])
    
    data = {}
    for c in cfgs:
        out_dict = compute_metric_dict(c,dataset_name=dataset_name)
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
            ax.set_ylim(args.ymin,args.ymax)
            ip += 1
        plt.legend(loc='best')
        print("Saving "+name+" to "+output_dir)
        fig.savefig(os.path.join(output_dir,name+"_vs_training_step.png"))
