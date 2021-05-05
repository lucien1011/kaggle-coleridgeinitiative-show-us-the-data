import os
import json
import sys
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from utils.preprocessing import json_to_text,json_to_list
from utils.progressbar import progressbar
from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

def make_token_label(ls,ws):
    nl = len(ls)
    nw = len(ws)
    il = 0
    iw = 0
    out = [0 for w in ws]
    while iw < nw:
        if ws[iw] == ls[il] and all([ws[iw+j+1] == ls[il+j+1] for j,l in enumerate(ls[1:])]):
            for j,l in enumerate(ls):
                out[iw+j] = 1
            return out
        iw += 1

    #out = [0 for w in ws]
    #for l in ls:
    #    try:
    #        out[ws.index(l)] = 1
    #    except ValueError:
    #        print(ls,ws)
    #        #raise ValueError
    #return out

# __________________________________________________________________ ||
cfg = ObjDict.read_from_file_python3(sys.argv[1])
verbose = True

# __________________________________________________________________ ||
df = pd.read_csv(cfg.input_train_df)

# __________________________________________________________________ ||
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#lemmatizer = nltk.stem.WordNetLemmatizer()

# __________________________________________________________________ ||
out_dict = {
        "token": list(), 
        "label": list(),
        "section": list(),
        "dataset": list(),
        "sentence": list(),
        "hasDataset": list(),
        }
for i,textId in enumerate(df['Id']):
 
    if verbose and i % 100 == 0: progressbar(i,df.shape[0])
    tokenized_label = tokenizer.convert_ids_to_tokens(tokenizer(df['dataset_label'][i])['input_ids'])

    json_path = os.path.join(cfg.input_train_json, (textId+'.json'))
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for isec,sec in enumerate(json_decode):
            token_sentences = sent_tokenize(sec['text'])
            tokens = [tokenizer.convert_ids_to_tokens(tokenizer(sen)['input_ids']) for sen in token_sentences]
            #tokens = [ [lemmatizer.lemmatize(t) for t in tokenizer.tokenize(sen)] for sen in token_sentences]
            tokenized_labels = [tokenized_label[1:-1] for sen in token_sentences] 
            out_dict['token'].extend(tokens)
            out_dict['section'].extend([sec['section_title'] for sen in token_sentences])
            out_dict['label'].extend([ [0 for _ in tokens[isen]] if df['dataset_label'][i] not in sen else make_token_label(tokenized_labels[isen],tokens[isen]) for isen,sen in enumerate(token_sentences)])
            out_dict['sentence'].extend(token_sentences)
            out_dict['dataset'].extend([df['dataset_label'][i] for sen in token_sentences])
            out_dict['hasDataset'].extend([int(df['dataset_label'][i] in sen) for sen in token_sentences])

out_df = pd.DataFrame(out_dict)
mkdir_p(cfg.input_np_dir)
out_df.to_csv(os.path.join(cfg.input_np_dir,'train.csv'))
