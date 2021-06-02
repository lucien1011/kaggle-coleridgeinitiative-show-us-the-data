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
    parser.add_argument('output_dir',type=str)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    train_df = pd.read_csv(os.path.join(args.input_dir,"train.csv"))
    fnames = os.listdir(os.path.join(args.input_dir,"train/"))

    train_dataset_names = pickle.load(open(os.path.join(args.input_dir,"train_dataset_names.p"),"rb"))
    val_dataset_names = pickle.load(open(os.path.join(args.input_dir,"val_dataset_names.p"),"rb"))

    out_dict = defaultdict(list)
    for fname in tqdm(fnames):
        fid = fname.replace(".json","")
        contents = json_to_list(fid)
        text = " ".join(contents)
        out_dict["text"].append(text)
        out_dict["id"].append(fid)
        out_dict["train_dataset"].append("|".join([d for d in train_dataset_names if d in text]))
        out_dict["external_dataset"].append("|".join([d for d in val_dataset_names if d in text]))

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(args.output_dir,"train_sequence.csv"),index=False)
