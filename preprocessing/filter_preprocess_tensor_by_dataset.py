import os
import torch
from collections import defaultdict

from utils.mkdir_p import mkdir_p

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',type=str)
    parser.add_argument('output_dir',type=str)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    input_ids = torch.load(os.path.join(args.input_dir,"input_ids.pt"))
    attention_mask = torch.load(os.path.join(args.input_dir,"attention_mask.pt"))
    overflow_to_sample_mapping = torch.load(os.path.join(args.input_dir,"overflow_to_sample_mapping.pt"))
    dataset_masks = torch.load(os.path.join(args.input_dir,"dataset_masks.pt"))
    labels = torch.load(os.path.join(args.input_dir,"labels.pt"))

    idx = labels.sum(axis=1).nonzero().squeeze()

    input_ids = input_ids[idx]
    attention_mask = attention_mask[idx]
    overflow_to_sample_mapping = overflow_to_sample_mapping[idx]
    dataset_masks = dataset_masks[idx]
    labels = labels[idx]

    mkdir_p(args.output_dir)

    torch.save(input_ids,os.path.join(args.output_dir,"input_ids.pt"))
    torch.save(attention_mask,os.path.join(args.output_dir,"attention_mask.pt"))
    torch.save(overflow_to_sample_mapping,os.path.join(args.output_dir,"overflow_to_sample_mapping.pt"))
    torch.save(dataset_masks,os.path.join(args.output_dir,"dataset_masks.pt"))
    torch.save(labels,os.path.join(args.output_dir,"labels.pt"))
