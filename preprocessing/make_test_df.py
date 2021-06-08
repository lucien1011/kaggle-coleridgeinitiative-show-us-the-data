import os
import pandas as pd
import pickle
from collections import defaultdict

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p
from utils.preprocessing import clean_text

def remove_reference(text):
    return text.split("REFERENCE")[0]

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    parser.add_argument('output_dir',type=str)
    parser.add_argument('--df_name',type=str,default='test_sequence.csv')
    parser.add_argument('--train_df_path',type=str,default='storage/input/kaggle_dataset/train.csv')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    output_dir = args.output_dir

    pub_to_dataset_map_path = os.path.join(args.input_dir,"pub_to_dataset_map.p")
    dataset_unique_map_path = os.path.join(args.input_dir,"data_sets_unique_map.p")
    text_path = os.path.join(args.input_dir,"text/")

    assert os.path.exists(pub_to_dataset_map_path) and os.path.exists(dataset_unique_map_path) and os.path.exists(text_path) and os.path.exists(args.train_df_path)
    
    fnames = os.listdir(text_path)
    
    assert len(fnames) > 0
    
    fnames.sort()
    outdict = defaultdict(list)
    pub_to_dataset_map = pickle.load(open(pub_to_dataset_map_path,"rb"))
    dataset_unique_map = pickle.load(open(dataset_unique_map_path,"rb"))
    train_df = pd.read_csv(args.train_df_path)
    train_dataset_names = train_df.dataset_label.unique().tolist()
    for fname in fnames:
        print("Processing ",fname)
        input_path = os.path.join(text_path,fname)
        f = open(input_path,"r")
        text = " ".join(f.readlines())
        text = remove_reference(text)

        textId = int(fname.replace(".txt",""))
        outdict["train_dataset"].append("|".join([clean_text(d) for d in train_dataset_names if d in text]))
        clean_datasets = []
        if textId in pub_to_dataset_map:
            orig_dataset_names = pub_to_dataset_map[textId]
            for orig_dataset_name in orig_dataset_names:
                if orig_dataset_name not in dataset_unique_map: continue
                text = text.replace(orig_dataset_name,dataset_unique_map[orig_dataset_name]['title'])
                clean_datasets.append(clean_text(dataset_unique_map[orig_dataset_name]['title']))

        outdict['text'].append(clean_text(text))
        outdict['id'].append(fname)

        dataset_str = "|".join(list(set((clean_datasets))))
        outdict['external_dataset'].append(dataset_str)

    mkdir_p(args.output_dir)
    df = pd.DataFrame(outdict)
    df.to_csv(os.path.join(args.output_dir,args.df_name),index=False)
