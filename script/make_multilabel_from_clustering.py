import sys
import pickle
import os
import torch
import pandas as pd

from pipeline.pipeline_tokenmulticlassifier import TokenMultiClassifierPipeline

labels_path = "data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/labels.pt"
overflow_path = "data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/overflow_to_sample_mapping.pt"

kMeans_path = "log/DatasetCluster_kMeans_210514_01/trained_labels.pkl"
df_path = "data/train_sequence_has_dataset.csv"

position_labels = torch.load(labels_path)
cluster_labels = pickle.load(open(kMeans_path,"rb"))
df = pd.read_csv(df_path)
overflow_mappings = torch.load(overflow_path)

for i in range(len(overflow_mappings)):
    position_labels[i] *= cluster_labels[df['dataset'][int(overflow_mappings[i])]]
torch.save(position_labels,"data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/clusterlabels.pt")
