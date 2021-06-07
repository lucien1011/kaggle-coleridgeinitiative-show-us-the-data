import os
import torch
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm

from model.CustomBertForTokenClassification import CustomBertForTokenClassification
from transformers import BertTokenizerFast

preprocess_dir      = "storage/preprocess_data/train_dataset_externalcv_210605/TokenBinaryClass_bert_base_uncased/test/"
base_model          = 'bert-base-uncased'
model_dir           = "storage/results/train_dataset_externalcv_210606/TokenBinaryClass_bert_base_uncased_12hiddenout_classweight_semisupervise/checkpoint-epoch-2/"
device              = 'cuda'
batch_size          = 64
output_path         = "storage/results/train_dataset_externalcv_210606/TokenBinaryClass_bert_base_uncased_12hiddenout_classweight_semisupervise/checkpoint-epoch-2/pred_str.txt"
header              = "*"*100


ids = torch.load(os.path.join(preprocess_dir,"input_ids.pt"))
am = torch.load(os.path.join(preprocess_dir,"attention_mask.pt"))
dm = torch.load(os.path.join(preprocess_dir,"dataset_masks.pt"))
m = CustomBertForTokenClassification.from_pretrained(model_dir).to(device)
t = BertTokenizerFast.from_pretrained(base_model)
d = TensorDataset(ids,am,dm)
dataloader = DataLoader(d,batch_size=batch_size)

ftext = open(output_path,"w")
for step,batch in enumerate(tqdm(dataloader)):
    batch_ids = batch[0].to(device)
    batch_am = batch[1].to(device)
    batch_dm = batch[2].to(device)
    with torch.no_grad():
        o = m(input_ids=batch_ids,attention_mask=batch_am)
    inds = batch_dm.sum(axis=1).nonzero()
    pred_locs = torch.argmax(o.logits,axis=2) 
    for ind_t in inds:
        ind = int(ind_t)
        pred_tokens = t.decode(batch_ids[ind]*pred_locs[ind],skip_special_tokens=True)
        true_tokens = t.decode(batch_ids[ind]*batch_dm[ind],skip_special_tokens=True)
        write_str = "\n".join([
            header,
            "Step "+str(step),
            header,
            "pred: "+pred_tokens,
            "true: "+true_tokens,
            ])
        ftext.write(write_str)
ftext.close()
