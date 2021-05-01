import os
import pickle
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
cfg = ObjDict.read_from_file_python3(sys.argv[1])

# __________________________________________________________________ ||
df = pd.read_csv(cfg.input_df,index_col=0)

# __________________________________________________________________ ||
df['hasDataset'] = df['hasDataset'].apply(lambda x: int(x))
x = df['sentence'].to_numpy()
y = np.expand_dims(df['hasDataset'].to_numpy(),axis=1)
frac = np.sum(y==1) / y.shape[0]
sample_weight = (y==0)*(1./(1-frac)/2.) + (y==1)*(1./(frac)/2.)

# __________________________________________________________________ ||
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

sample_weight_train = (y_train==0)*(1./(1-frac)/2.) + (y_train==1)*(1./(frac)/2.)
sample_weight_val = (y_val==0)*(1./(1-frac)/2.) + (y_val==1)*(1./(frac)/2.)
sample_weight_test = (y_test==0)*(1./(1-frac)/2.) + (y_test==1)*(1./(frac)/2.)

# __________________________________________________________________ ||
mkdir_p(cfg.input_np_dir)

# __________________________________________________________________ ||
np.save(os.path.join(cfg.input_np_dir,"x_train.npy"),x_train)
np.save(os.path.join(cfg.input_np_dir,"y_train.npy"),y_train)
np.save(os.path.join(cfg.input_np_dir,"sample_weight_train.npy"),sample_weight_train)

np.save(os.path.join(cfg.input_np_dir,"x_val.npy"),x_val)
np.save(os.path.join(cfg.input_np_dir,"y_val.npy"),y_val)
np.save(os.path.join(cfg.input_np_dir,"sample_weight_val.npy"),sample_weight_val)

np.save(os.path.join(cfg.input_np_dir,"x_test.npy"),x_test)
np.save(os.path.join(cfg.input_np_dir,"y_test.npy"),y_test)
np.save(os.path.join(cfg.input_np_dir,"sample_weight_test.npy"),sample_weight_test)
