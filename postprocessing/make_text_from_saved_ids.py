import os
import glob
import argparse

import torch

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

next_line = '\n'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--write_dataset_only',action='store_true')
    return parser.parse_args()

def convert_ids_to_string(cfg,device='cuda',out_str='pred_str.txt',write_dataset_only=False):
    pp = cfg.pipeline
    out_dict = {}
    checkpts = pp.get_model_checkpts(cfg.extract_cfg.model_dir,cfg.extract_cfg.model_key)
    
    outfile = open(os.path.join(cfg.extract_cfg.output_dir,out_str),"w")
    checkpts.sort()
    for c in checkpts:
        outfile.write("*"*100+next_line)
        outfile.write(c+next_line)
        outfile.write("*"*100+next_line)
        cdir = os.path.join(cfg.extract_cfg.output_dir,c)
        if not os.path.exists(cdir):
            print(cdir+" not exist")
            continue
        fs = [f for f in os.listdir(cdir) if ".pt" in f]
        if not fs:
            print("empty folder: {}, skipping".format(c))
            continue
        pred_ids = torch.cat([torch.load(os.path.join(cdir,f)) for f in fs])
        pred_ids[pred_ids==-1] = 0
        if write_dataset_only:
            pred_ids = pred_ids[torch.sum(pred_ids,axis=1)!=0]
        pp.print_message(c)

        strs = cfg.tokenizer.batch_decode(pred_ids,skip_special_tokens=True)
        outfile.write(next_line.join([str(i)+", "+s for i,s in enumerate(strs) if s]))
        outfile.write(next_line)
    outfile.close()

if __name__ == "__main__":

    args = parse_arguments()

    if "*" in args.input_path:
        paths = glob.glob(args.input_path)
    else:
        paths = args.input_path.split(",")

    assert len(paths) > 0
    
    cfgs = [ObjDict.read_all_from_file_python3(p) for p in paths]
    assert all([hasattr(c,"plot_label",) for c in cfgs])
    
    data = {}
    for c in cfgs:
        convert_ids_to_string(c,write_dataset_only=args.write_dataset_only)        
