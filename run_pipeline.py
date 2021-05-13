import sys

from utils.objdict import ObjDict

cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

pipeline = cfg.pipeline
inputs = pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
pipeline.train(inputs,cfg.model,cfg.train_cfg)
