import os
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

from utils.preprocessing import json_to_list

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    parser.add_argument('dataset_dir',type=str)
    parser.add_argument('output_dir',type=str)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    fnames = os.listdir(args.input_dir)
    train_dataset_names = pickle.load(open(os.path.join(args.dataset_dir,"train_dataset_names.p"),"rb"))
    val_dataset_names = pickle.load(open(os.path.join(args.dataset_dir,"val_dataset_names.p"),"rb"))
    output_dir = args.output_dir

    assert len(fnames) > 0
    
    fnames.sort()
    outdict = defaultdict(list)
    for fname in fnames:
        print("Processing ",fname)
        input_path = os.path.join(args.input_dir,fname)
        f = open(input_path,"r")
        text = " ".join(f.readlines())
        outdict['text'].append(text)
        outdict['id'].append(fname)
        outdict["test_dataset"].append("|".join([d for d in train_dataset_names if d in text]))
        outdict["external_dataset"].append("|".join([d for d in val_dataset_names if d in text]))
    df = pd.DataFrame(outdict)
    df.to_csv(os.path.join(args.output_dir,"test_sequence.csv"),index=False) 
