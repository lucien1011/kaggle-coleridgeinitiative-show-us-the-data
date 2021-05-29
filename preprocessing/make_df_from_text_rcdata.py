import os
import pandas as pd
import pickle
from collections import defaultdict

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    parser.add_argument('output_dir',type=str)
    parser.add_argument('--df_name',type=str,default='df_rcdataset.csv')
    parser.add_argument('--dataset_map',type=str,default='')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    fnames = os.listdir(args.input_dir)
    output_dir = args.output_dir

    assert len(fnames) > 0
    
    fnames.sort()
    outdict = defaultdict(list)
    if args.dataset_map:
        pub_to_dataset_map = pickle.load(open(args.dataset_map,"rb"))
    for fname in fnames:
        print("Processing ",fname)
        input_path = os.path.join(args.input_dir,fname)
        f = open(input_path,"r")
        text = " ".join(f.readlines())
        outdict['text'].append(text)
        outdict['id'].append(fname)
        if args.dataset_map:
            textId = int(fname.replace(".txt",""))
            dataset_str = "|".join(pub_to_dataset_map[textId]) if textId in pub_to_dataset_map else ''
            outdict['dataset'].append(dataset_str)
    df = pd.DataFrame(outdict)
    df.to_csv(os.path.join(args.output_dir,args.df_name),index=False)
