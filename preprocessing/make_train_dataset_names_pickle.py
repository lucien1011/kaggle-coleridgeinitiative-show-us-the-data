import os
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

from utils.preprocessing import clean_text

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('output_path',type=str)
    parser.add_argument('--label_name',type=str,default='dataset')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    df = pd.read_csv(args.input_path)
    dataset_names = []
    for ds in df[args.label_name].unique().tolist():
        if type(ds) != str: continue
        for d in ds.split("|"):
            if d not in dataset_names:
                dataset_names.append(clean_text(d))
    pickle.dump(dataset_names,open(args.output_path,"wb"))
