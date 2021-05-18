import os
import sys
import pickle
import pandas as pd
import joblib

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

model = joblib.load(os.path.join(cfg.train_cfg.output_dir,cfg.train_cfg.saved_model_path))
df = pd.read_csv(cfg.preprocess_cfg.train_csv_path)

ds = df['dataset'].unique().tolist()

label_dict = {}
trained_labels_dict = {}
for i,l in enumerate(model.labels_):
    if l not in label_dict:
        label_dict[l] = []
    label_dict[l].append(ds[i])
    trained_labels_dict[ds[i]] = l

for key,value in label_dict.items():
    print(key,value)

pickle.dump(trained_labels_dict,open(os.path.join(cfg.train_cfg.output_dir,"trained_labels.pkl"),"wb"))
