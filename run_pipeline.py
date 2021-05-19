import sys

from utils.objdict import ObjDict

cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

pipeline = cfg.pipeline
if not cfg.preprocess_cfg.load_preprocess:
    _ = pipeline.create_preprocess_train_data(cfg.preprocess_cfg)
inputs = pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
pipeline.train(inputs,cfg.model,cfg.train_cfg)
