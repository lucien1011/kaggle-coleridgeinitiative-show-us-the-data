import os
import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm

# __________________________________________________________________ ||
#aug_syn = naw.SynonymAug(aug_src='wordnet')
#aug_w2v = naw.WordEmbsAug('glove')

TOPK = 20
ACT = 'substitute'
aug_bert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased',action=ACT, top_k=TOPK)

input_csv_path = "data/train_sequence_has_dataset.csv"

# __________________________________________________________________ ||
df = pd.read_csv(input_csv_path)

out_dict = {key: [] for key in df.columns}
nrow = len(df)
for i in tqdm(range(nrow)):
    orig_dataset = df['dataset'][i]
    aug_dataset = aug_bert.augment(orig_dataset)
    out_dict['dataset'].append(aug_dataset)
    out_dict['text'].append(df['text'][i].replace(orig_dataset,aug_dataset))
    out_dict['id'].append(df['id'][i])
    out_dict['section'].append(df['section'][i])
    out_dict['has_dataset'].append(df['has_dataset'][i])
out_df = pd.DataFrame(out_dict)
out_df.to_csv(os.path.join(os.path.dirname(input_csv_path),'train_sequence_aug_dataset.csv'),index=False)   
