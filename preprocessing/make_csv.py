import os
import pandas as pd
from collections import defaultdict

from utils.preprocessing import json_to_list

if __name__ == "__main__":
    train_df = pd.read_csv("input/train.csv")
    
    train_df = train_df.iloc[:1000]
    datasets = train_df.dataset_label.unique()

    out_dict = defaultdict(list)
    fnames = os.listdir("input/train/")
    for fname in fnames:
        fid = fname.replace(".json","")
        contents = json_to_list(fid)
        text = " ".join(contents)
        dts_in_text = [d for d in datasets if d in text]
        dataset = ",".join(dts_in_text)
        ndataset = len(dts_in_text)

        out_dict["text"].append(text)
        out_dict["dataset"].append(dataset)
        out_dict["ndataset"].append(ndataset)
        out_dict["id"].append(fid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv("storage/data/train_sequence.csv",index=False)
