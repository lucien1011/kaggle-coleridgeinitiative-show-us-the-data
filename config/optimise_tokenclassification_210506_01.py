import os

from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "optimise_tokenclassification_210505"

# __________________________________________________________________ ||
config = ObjDict(

    name = name,

    input_train_df = "input/train.csv",
    input_train_json = "input/train/",

    input_seq_df = 'data/optimise_tokenclassification_210505_01/test.csv',
    
    model_checkpoint = "distilbert-base-uncased", 

)

# __________________________________________________________________ ||
config.input_dataset_dir = "data/huggingface/"
config.input_dataset_name = 'optimise_tokenclassification_210506_01/'
