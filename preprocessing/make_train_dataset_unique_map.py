import json
import pickle
import pandas as pd
import pprint

input_path = "storage/input/combine_dataset/train.csv"
output_path = "storage/input/rcdataset/train_data_sets_unique_map.p"
dataset_unique_map_path = "storage/input/rcdataset/data_sets_unique_map.p"
dataset_unique_map = pickle.load(open(dataset_unique_map_path,"rb"))

out_dict = {}

df = pd.read_csv(input_path)
pprint.pprint([(d,d in dataset_unique_map) for d in df.dataset_label.unique()])


pickle.dump(out_dict,open(output_path,"wb"))
