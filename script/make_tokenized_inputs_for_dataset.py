import pandas as pd
from transformers import DistilBertTokenizerFast
import torch

df = pd.read_csv("data/train_sequence_has_dataset.csv")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(
        df['dataset'].unique().tolist(),
        padding='max_length',
        max_length=512,
        truncation=True,
        return_overflowing_tokens=True,
        return_tensors="pt"
        )
torch.save(inputs,"data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/dataset_tokens.pt")
