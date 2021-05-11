import os
import json
import pandas as pd

from utils.progressbar import progressbar
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
input_dir = 'input/train/'
output_dir = 'data/'
verbose = True

df = pd.read_csv('input/train.csv')

# __________________________________________________________________ ||
out_dict = {
        "id": list(), 
        "text": list(), 
        "section": list(),
        "dataset": list(),
        "has_dataset": list(),
        }
for i,textId in enumerate(df['Id']):
 
    if verbose and i % 100 == 0: progressbar(i,df.shape[0])

    json_path = os.path.join(input_dir, (textId+'.json'))
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        out_dict['id'].extend([textId for d in json_decode])
        out_dict['has_dataset'].extend([df['dataset_label'][i] in d['text'] for d in json_decode])
        out_dict['text'].extend([d['text'] for d in json_decode])
        out_dict['section'].extend([d['section_title'] for d in json_decode])
        out_dict['dataset'].extend([df['dataset_label'][i] for d in json_decode])

out_df = pd.DataFrame(out_dict)
out_df = out_df[out_df['text'] != '']
out_df.to_csv(os.path.join(output_dir,'train_sequence.csv'),index=False)
