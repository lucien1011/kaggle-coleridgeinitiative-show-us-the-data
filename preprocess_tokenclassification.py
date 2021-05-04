import os
import json
import pandas as pd

from nltk.tokenize import sent_tokenize

from utils.preprocessing import json_to_text,json_to_list
from utils.progressbar import progressbar

def make_token_label(label,sen):
    ls = label.split()
    ws = sen.split()
    print(ls)
    print(ws)
    out = [0 for w in ws]
    for l in ls:
        out[ws.index(l)] = 1
    return out

input_dir = 'input/train/'
output_dir = 'data/'
verbose = True

df = pd.read_csv('input/train.csv')

out_dict = {
        "token": list(), 
        "label": list(),
        "section": list(),
        "dataset_label": list(),
        }
for i,textId in enumerate(df['Id']):
 
    if verbose and i % 100 == 0: progressbar(i,df.shape[0])

    json_path = os.path.join(input_dir, (textId+'.json'))
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for isec,sec in enumerate(json_decode):
            token_sentences = sent_tokenize(sec['text'])
            out_dict['token'].extend([sen.split() for sen in  token_sentences])
            out_dict['section'].extend([sec['section_title'] for sen in token_sentences])
            out_dict['label'].extend([ [0 for s in sen.split()] if df['dataset_label'][i] not in sen else make_token_label(df['dataset_label'][i],sen) for sen in token_sentences])

out_df = pd.DataFrame(out_dict)
out_df.to_csv(os.path.join(output_dir,'train.csv'))
