import os
import json
import pandas as pd

from nltk.tokenize import sent_tokenize

from utils.preprocessing import json_to_text,json_to_list
from utils.progressbar import progressbar

input_dir = 'input/train/'
output_dir = 'data/'
verbose = True

df = pd.read_csv('input/train.csv')

out_dict = {
        "sentence": list(), 
        "hasDataset": list(),
        "section": list(),
        }
for i,textId in enumerate(df['Id']):
 
    if verbose and i % 100 == 0: progressbar(i,df.shape[0])

    json_path = os.path.join(input_dir, (textId+'.json'))
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for isec,sec in enumerate(json_decode):
            token_sentences = sent_tokenize(sec['text'])
            out_dict['sentence'].extend(token_sentences)
            out_dict['section'].extend([sec['section_title'] for sen in token_sentences])
            out_dict['hasDataset'].extend([int(df['dataset_label'][i] in sen) for sen in token_sentences])

out_df = pd.DataFrame(out_dict)
out_df.to_csv(os.path.join(output_dir,'train.csv'))
