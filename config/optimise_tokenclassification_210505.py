import os

from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_tokenclassification_210505"

# __________________________________________________________________ ||
config = ObjDict(

    name = name,
    input_train_df = "input/train.csv",
    input_train_json = "input/train/",

)

# __________________________________________________________________ ||
config.input_np_dir = "data/optimise_tokenclassification_210505_01/" 
