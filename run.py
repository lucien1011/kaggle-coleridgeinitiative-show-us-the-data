import argparse
import sys

from utils.objdict import ObjDict

job_keywords = [
        "create_train_data",
        "load_train_data",
        "create_test_data",
        "load_test_data",
        "load_train_test_data",
        "randomize_train_data",
        "include_pred_dataset_as_label_train",
        "include_external_dataset_as_label_train",
        "include_external_dataset_as_label_test",
        "mask_test_dataset_name_train",
        "train",
        "semisupervise_train",
        ]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('jobs')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.cfg)
    
    jobs = args.jobs.split(",")

    jobs_not_supported = [j for j in jobs if j not in job_keywords]
    assert len(jobs_not_supported) == 0,"Not matching keywords: "+",".join(jobs_not_supported)
    
    if "create_train_data" in jobs:
        inputs = cfg.preprocessor.create_preprocess_train_data(cfg.preprocess_cfg)
    
    if "load_train_data" in jobs:
        inputs = cfg.preprocessor.load_preprocess_train_data(cfg.preprocess_cfg)
    
    if "create_test_data" in jobs:
        inputs = cfg.preprocessor.create_preprocess_test_data(cfg.preprocess_cfg)
    
    if "load_test_data" in jobs:
        inputs = cfg.preprocessor.load_preprocess_test_data(cfg.preprocess_cfg)

    if "load_train_test_data" in jobs:
        inputs = cfg.preprocessor.load_preprocess_train_test_data(cfg.preprocess_cfg)

    if "include_pred_dataset_as_label_train" in jobs:
        inputs.train_dataset = cfg.preprocessor.include_pred_dataset_as_label(inputs.train_dataset,cfg)

    if "include_external_dataset_as_label_train" in jobs:
        inputs.train_dataset = cfg.preprocessor.include_external_dataset_as_label(inputs.train_dataset,cfg)

    if "include_external_dataset_as_label_test" in jobs:
        inputs.val_dataset = cfg.preprocessor.include_external_dataset_as_label(inputs.val_dataset,cfg)

    if "mask_test_dataset_name_train" in jobs:
        inputs.train_dataset = cfg.preprocessor.mask_test_dataset_name_train(inputs.train_dataset,cfg)

    if "train" in jobs:
        cfg.trainer.train(inputs,cfg.model,cfg.train_cfg)

    if "semisupervise_train" in jobs:
        cfg.trainer.semisupervise_train(inputs,cfg.model,cfg.train_cfg)
