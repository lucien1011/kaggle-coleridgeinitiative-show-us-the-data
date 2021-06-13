import os
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

from utils.preprocessing import json_to_list,clean_text
from utils.mkdir_p import mkdir_p

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    parser.add_argument('output_dir',type=str)
    parser.add_argument('--df_name',type=str,default='train_sequence.csv')
    parser.add_argument('--data_unique_map_path',type=str,default='storage/input/rcdataset/data_sets_unique_map.p')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    train_df = pd.read_csv(os.path.join(args.input_dir,"train.csv"))
    fnames = os.listdir(os.path.join(args.input_dir,"train/"))

    dataset_unique_map_path = args.data_unique_map_path
    dataset_unique_map = pickle.load(open(dataset_unique_map_path,"rb"))
    
    train_dataset_names = train_df.dataset_label.unique().tolist()

    out_dict = defaultdict(list)
    for fname in tqdm(fnames):
        fid = fname.replace(".json","")
        contents = json_to_list(fid)
        text = " ".join(contents)
        out_dict["id"].append(fid)
        clean_val_datasets = []
        val_datasets = []
        for d,info in dataset_unique_map.items():
            if d in text:
                if any([td in d for td in train_dataset_names]): continue
                val_datasets.append(d)
                #clean_val_datasets.append(info['unique_dataset_name'])
                #text = text.replace(d,info['unique_dataset_name'])
                clean_val_datasets.append(clean_text(d))
        out_dict["external_dataset"].append("|".join(list(set(clean_val_datasets))))
        out_dict["orig_external_dataset"].append("|".join(list(set(val_datasets))))
        out_dict["train_dataset"].append("|".join([clean_text(d) for d in train_dataset_names if d in text]))
        out_dict["text"].append(clean_text(text))

    mkdir_p(args.output_dir)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(os.path.join(args.output_dir,args.df_name),index=False)
