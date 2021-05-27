import os
import glob
import argparse

import pandas as pd
import torch

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

next_line = '\n'
saved_label_key = "pred_ids"
of_mapping_key = "overflow_mapping"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--write_dataset_only',action='store_true')
    parser.add_argument('--add_id',action='store_true')
    return parser.parse_args()

def convert_ids_to_string(cfg,device='cuda',out_str='pred_str.txt',write_dataset_only=False,add_id=False):
    pp = cfg.pipeline
    out_dict = {}
    checkpts = pp.get_model_checkpts(cfg.extract_cfg.model_dir,cfg.extract_cfg.model_key)
    if add_id:
        textids = pd.read_csv(cfg.preprocess_cfg.test_csv_path).id
    
    outfile = open(os.path.join(cfg.extract_cfg.output_dir,out_str),"w")
    checkpts.sort()
    for c in checkpts:
        print("Processing checkpoint "+c)
        outfile.write("*"*100+next_line)
        outfile.write(c+next_line)
        outfile.write("*"*100+next_line)
        cdir = os.path.join(cfg.extract_cfg.output_dir,c)
        if not os.path.exists(cdir):
            print(cdir+" not exist")
            continue
        fs = [f for f in os.listdir(cdir) if ".pt" in f]
        fs.sort()
        if not fs:
            print("empty folder: {}, skipping".format(c))
            continue
        pred_ids = torch.cat([torch.load(os.path.join(cdir,f)) for f in fs if saved_label_key in f])
        if add_id:
            ids = torch.cat([torch.load(os.path.join(cdir,f)) for f in fs if of_mapping_key in f])
        pred_ids[pred_ids==-1] = 0
        if write_dataset_only:
            pred_idx = torch.sum(pred_ids,axis=1)!=0
            pred_ids = pred_ids[pred_idx]
            ids = ids[pred_idx]
        pp.print_message(c)

        strs = cfg.tokenizer.batch_decode(pred_ids,skip_special_tokens=True)
        if add_id:
            outfile.write(next_line.join([str(textids[int(ids[i])])+": "+str(i)+", "+s for i,s in enumerate(strs) if s]))
        else:
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
        convert_ids_to_string(c,write_dataset_only=args.write_dataset_only,add_id=args.add_id)        
