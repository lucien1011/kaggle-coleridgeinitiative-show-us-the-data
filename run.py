import sys

from utils.objdict import ObjDict

cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

pipeline = cfg.pipeline
jobs = sys.argv[2].split(",")

if "create_data" in jobs:
    if not cfg.preprocess_cfg.load_preprocess:
        _ = pipeline.create_preprocess_train_data(cfg.preprocess_cfg)

if "load_data" in jobs:
    inputs = pipeline.load_preprocess_train_data(cfg.preprocess_cfg)

if "train" in jobs:
    pipeline.train(inputs,cfg.model,cfg.train_cfg)

if "predict" in jobs:
    pipeline.predict(inputs,cfg.model,cfg.predict_cfg)
