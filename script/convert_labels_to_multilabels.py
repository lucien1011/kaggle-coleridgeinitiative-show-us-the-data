import torch

input_pt_path = 'data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/labels.pt'
output_pt_path = 'data/optimise_TokenMultiClass_distilbert_base_uncased_210517/train/multilabels.pt'

labels = torch.load(input_pt_path)

for i in range(len(labels)):
    idx = (labels[i]==1).nonzero()
    if len(idx) > 2:
        labels[i][idx[1:-1]] = 2
    if len(idx) > 1:
        labels[i][idx[-1]] = 3

torch.save(labels,output_pt_path)
