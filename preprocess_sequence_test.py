import os
import json
import pandas as pd

from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
input_dir = 'input/test/'
output_dir = 'data/'
verbose = True

# __________________________________________________________________ ||
out_dict = {
        "id": list(), 
        "text": list(), 
        "section": list(),
        }

for i,fname in enumerate(os.listdir(input_dir)):
 
    if verbose: print("Processing",fname) 

    textId = fname.replace(".json","")
    json_path = os.path.join(input_dir,fname)
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        out_dict['id'].extend([textId for d in json_decode])
        out_dict['text'].extend([d['text'] for d in json_decode])
        out_dict['section'].extend([d['section_title'] for d in json_decode])

out_df = pd.DataFrame(out_dict)
out_df = out_df[out_df['text'] != '']
out_df.to_csv(os.path.join(output_dir,'test_sequence.csv'),index=False)
