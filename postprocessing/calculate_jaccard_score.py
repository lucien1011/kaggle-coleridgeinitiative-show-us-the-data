import os
import sys
import torch
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm

from model.CustomBertForTokenClassification import CustomBertForTokenClassification
from transformers import BertTokenizerFast

from utils.objdict import ObjDict

cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

header              = "*"*100
base_model          = cfg.base_pretrained
device              = cfg.calculate_score_cfg.device
batch_size          = cfg.calculate_score_cfg.batch_size

preprocess_dir      = cfg.preprocess_test_dir
model_dir           = os.path.join(cfg.calculate_score_cfg.model_dir,cfg.calculate_score_cfg.model_key)
output_path         = os.path.join(model_dir,"pred_str.txt")

def jaccard_similarity(s1, s2):
    l1 = s1.split(" ")
    l2 = s2.split(" ")    
    intersection = len(list(set(l1).intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    return float(intersection) / union

def split_list(ids):
    result = []
    tmp = []
    for id in ids:
        if id != 0:
            tmp.append(int(id))
        else:
            if tmp: result.append(tmp)
            tmp = []
    if tmp: result.append(tmp)
    return result

if __name__ == "__main__":

    ids = torch.load(os.path.join(preprocess_dir,"input_ids.pt"))
    am = torch.load(os.path.join(preprocess_dir,"attention_mask.pt"))
    dm = torch.load(os.path.join(preprocess_dir,"dataset_masks.pt"))
    m = CustomBertForTokenClassification.from_pretrained(model_dir).to(device)
    t = BertTokenizerFast.from_pretrained(base_model)
    d = TensorDataset(ids,am,dm)
    dataloader = DataLoader(d,batch_size=batch_size)
    
    jaccard_scores = []
    ftext = open(output_path,"w")
    for step,batch in enumerate(tqdm(dataloader)):
        if step > cfg.calculate_score_cfg.n_sample: continue
        batch_ids = batch[0].to(device)
        batch_am = batch[1].to(device)
        batch_dm = batch[2].to(device)
        with torch.no_grad():
            o = m(input_ids=batch_ids,attention_mask=batch_am)
        inds = batch_dm.sum(axis=1).nonzero()
        pred_locs = torch.argmax(o.logits,axis=2) 
        for ind_t in inds:
            ind = int(ind_t)
            pred_tokens = t.batch_decode(split_list(batch_ids[ind]*pred_locs[ind]),skip_special_tokens=True)
            pred_str = " ".join(pred_tokens)
            true_tokens = t.batch_decode(split_list(batch_ids[ind]*batch_dm[ind]),skip_special_tokens=True)
            true_str = " ".join(true_tokens)
            jaccard_score = jaccard_similarity(pred_str,true_str)
            jaccard_scores.append(jaccard_score)
            write_str = "\n".join([
                header,
                "Step "+str(step),
                header,
                "pred: "+pred_str,
                "true: "+true_str,
                "jaccard score: "+str(jaccard_score),
                " ",
                ])
            ftext.write(write_str)
    ftext.close()
    print("Average score: ",str(sum(jaccard_scores)/len(jaccard_scores)))
