import os
import json
import sys
import pandas as pd

from pipeline.pipeline_tokenclassifier import TokenClassifierPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

# __________________________________________________________________ ||
pp = TokenClassifierPipeline()

_ = pp.create_preprocess_train_data(cfg.preprocess_cfg)
_ = pp.create_preprocess_test_data(cfg.preprocess_cfg)
