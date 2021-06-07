import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pprint

from utils.preprocessing import clean_text

input_path = "storage/input/rcdataset/data_sets.json"
output_path = "storage/input/rcdataset/data_sets_word_frequency.png"

f = json.load(open(input_path,"r"))
count_dict = {}
for d in f:
    for m in d['mention_list']:
        for w in m.split():
            w2 = clean_text(w.lower())
            if w2 not in count_dict:
                count_dict[w2] = 0
            else:
                count_dict[w2] += 1
results = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:50]

fig,ax = plt.subplots(figsize=(10,10))
ax.tick_params(labelrotation=90)
ax.bar([r[0] for r in results],[r[1] for r in results],)
fig.savefig(output_path)
