import os
import torch
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
from tqdm import tqdm

from metrics.kaggle import jaccard_similarity,calculate_tp_fp_fn,fbeta

header = "*"*100

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

def calculate_fbeta_tp_fp_fn_from_cfg_checkpoint(cfg,checkpt,cut=0.5,nmax=100):

    base_model          = cfg.base_pretrained
    device              = cfg.calculate_score_cfg.device
    batch_size          = cfg.calculate_score_cfg.batch_size 
    preprocess_dir      = cfg.preprocess_test_dir

    ids = torch.load(os.path.join(preprocess_dir,"input_ids.pt"))
    am = torch.load(os.path.join(preprocess_dir,"attention_mask.pt"))
    dm = torch.load(os.path.join(preprocess_dir,"dataset_masks.pt"))
    t = cfg.tokenizer.from_pretrained(base_model)
    d = TensorDataset(ids,am,dm)
    model_dir = os.path.join(cfg.calculate_score_cfg.model_dir,checkpt)
    sampler = RandomSampler(d)
    dataloader = DataLoader(d,batch_size=batch_size,sampler=sampler)
    model_args = getattr(cfg,"model_args",{})
    m = cfg.model.from_pretrained(model_dir,**model_args).to(device)
    
    tot_tp,tot_fp,tot_fn = 0,0,0
    for step,batch in enumerate(tqdm(dataloader)):
        if step > nmax: continue
        batch_ids = batch[0].to(device)
        batch_am = batch[1].to(device)
        batch_dm = batch[2].to(device)
        with torch.no_grad():
            o = m(input_ids=batch_ids,attention_mask=batch_am)
        inds = batch_dm.sum(axis=1).nonzero()
        pred_locs = (torch.nn.functional.softmax(o.logits,dim=2)[:,:,1] > cut).long()
        for ind_t in inds:
            ind = int(ind_t)
            pred_tokens = t.batch_decode(split_list(batch_ids[ind]*pred_locs[ind]),skip_special_tokens=True)
            pred_tokens = [ts for ts in pred_tokens if len(ts) >= 3]
            pred_str = "|".join(pred_tokens)
            true_tokens = t.batch_decode(split_list(batch_ids[ind]*batch_dm[ind]),skip_special_tokens=True)
            true_str = "|".join(true_tokens)
            tp,fp,fn = calculate_tp_fp_fn(pred_tokens,true_tokens)
            tot_tp += tp
            tot_fp += fp
            tot_fn += fn
            write_str = "\n".join([
                header,
                "Step "+str(step),
                header,
                "pred: "+pred_str,
                "true: "+true_str,
                " ",
                ])
    fbeta_score = fbeta(tot_tp,tot_fp,tot_fn)
    return fbeta_score,tot_tp,tot_fp,tot_fn
