import os
import pandas as pd
from collections import defaultdict

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    parser.add_argument('output_dir',type=str)
    parser.add_argument('--df_name',type=str,default='df_rcdataset.csv')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    fnames = os.listdir(args.input_dir)
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
    df = pd.DataFrame(outdict)
    df.to_csv(os.path.join(args.output_dir,args.df_name),index=False)
