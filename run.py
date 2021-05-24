import argparse
import sys

from utils.objdict import ObjDict

job_keywords = [
        "create_data",
        "load_data",
        "train_data",
        "predict",
        "evaluate",
        ]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('jobs')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.cfg)
    
    pipeline = cfg.pipeline
    jobs = args.jobs.split(",")

    assert all([j in job_keywords for j in jobs])
    
    if "create_data" in jobs:
        inputs = pipeline.create_preprocess_train_data(cfg.preprocess_cfg)
    
    if "load_data" in jobs:
        inputs = pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
    
    if "train" in jobs:
        pipeline.train(inputs,cfg.model,cfg.train_cfg)
    
    if "predict" in jobs:
        pipeline.predict(inputs,cfg.model,cfg.predict_cfg)
    
    if "evaluate" in jobs:
        pipeline.evaluate(inputs,cfg.model,cfg.evaluate_cfg)
