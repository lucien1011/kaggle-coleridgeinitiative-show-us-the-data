import os
import argparse
import torch
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.CustomBertForTokenClassification import CustomBertForTokenClassification
from transformers import BertTokenizerFast

from utils.objdict import ObjDict

header = "*"*100

# __________________________________________________________________ ||
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--nmax',type=int,default=1e7)
    parser.add_argument('--model_key',type=str,default='checkpoint-epoch-')
    parser.add_argument('-p','--plot_to_path',type=str,default="")
    return parser.parse_args()

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

# __________________________________________________________________ ||
if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)

    base_model          = cfg.base_pretrained
    device              = cfg.calculate_score_cfg.device
    batch_size          = cfg.calculate_score_cfg.batch_size 
    preprocess_dir      = cfg.preprocess_test_dir

    ids = torch.load(os.path.join(preprocess_dir,"input_ids.pt"))
    am = torch.load(os.path.join(preprocess_dir,"attention_mask.pt"))
    dm = torch.load(os.path.join(preprocess_dir,"dataset_masks.pt"))
    t = BertTokenizerFast.from_pretrained(base_model)
    d = TensorDataset(ids,am,dm)
   
    checkpts = cfg.pipeline.get_model_checkpts(cfg.calculate_score_cfg.model_dir,args.model_key)
    assert len(checkpts) > 0
    assert all([c.replace(args.model_key,"").isdigit() for c in checkpts])
    checkpts.sort(key=lambda x: int(x.replace(args.model_key,"")))

    if args.plot_to_path:
        x,y = [],[]

    for checkpt in checkpts:
        print(header)
        print("Processing "+checkpt)
        print(header)
        
        model_dir = os.path.join(cfg.calculate_score_cfg.model_dir,checkpt)
        output_path = os.path.join(model_dir,"pred_str.txt")
        
        sampler = RandomSampler(d)
        dataloader = DataLoader(d,batch_size=batch_size,sampler=sampler)
        m = CustomBertForTokenClassification.from_pretrained(model_dir).to(device)
        
        jaccard_scores = []
        ftext = open(output_path,"w")
        for step,batch in enumerate(tqdm(dataloader)):
            if step > args.nmax: continue
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
        avg_jacc_score = sum(jaccard_scores)/len(jaccard_scores)
        print("Average score: ",str(avg_jacc_score))
        
        if args.plot_to_path:
            x.append(int(checkpt.replace(args.model_key,"")))
            y.append(avg_jacc_score)

if args.plot_to_path:
    fig,ax = plt.subplots()
    ax.plot(x,y)
    ax.set_ylabel("Average Jaccard Score")
    ax.set_xlabel("Epoch")
    fig.savefig(args.plot_to_path)
