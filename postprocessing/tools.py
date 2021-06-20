import os
import torch
from torch.utils.data import DataLoader,TensorDataset,RandomSampler,SequentialSampler
from tqdm import tqdm

from metrics.kaggle import jaccard_similarity,calculate_tp_fp_fn,fbeta,calculate_precision

header = "*"*100

def split_list(ids,return_indices=False):
    result = []
    indices = []
    tmp = []
    tmp2 =[]
    for i,id in enumerate(ids):
        if id != 0:
            tmp.append(int(id))
            tmp2.append(i)
        else:
            if tmp: 
                result.append(tmp)
                indices.append(tmp2)
            tmp = []
    if tmp: 
        result.append(tmp)
        indices.append(tmp2)
    
    if return_indices:
        return result,indices
    else:
        return result

def calculate_fbeta_tp_fp_fn_from_cfg_checkpoint(cfg,checkpt,cut=0.5,nmax=100,input_true_tokens=None,beta=0.5):

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
    m.eval()

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
            if not input_true_tokens:
                true_tokens = t.batch_decode(split_list(batch_ids[ind]*batch_dm[ind]),skip_special_tokens=True)
            else:
               true_tokens = input_true_tokens
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
    fbeta_score = fbeta(tot_tp,tot_fp,tot_fn,beta=beta)
    return fbeta_score,tot_tp,tot_fp,tot_fn

def read_dataset(cfg):
    device              = cfg.calculate_score_cfg.device
    batch_size          = cfg.calculate_score_cfg.batch_size 
    preprocess_dir      = cfg.preprocess_test_dir

    ids = torch.load(os.path.join(preprocess_dir,"input_ids.pt"))
    am = torch.load(os.path.join(preprocess_dir,"attention_mask.pt"))
    dm = torch.load(os.path.join(preprocess_dir,"dataset_masks.pt"))
    d = TensorDataset(ids,am,dm)
    return d

def extract_text_from_pred(batch_ids,batch_am,model,tokenizer,cut,batch_size=128):
    outputs = []
    dataset = TensorDataset(batch_ids,batch_am)
    dataloader = DataLoader(dataset,batch_size=batch_size)
    for batch in dataloader:
        with torch.no_grad():
            output = model(input_ids=batch[0],attention_mask=batch[1])
        outputs.append(output.logits)
    outputs = torch.cat(outputs,axis=0)
    pred_locs = (torch.nn.functional.softmax(outputs,dim=2)[:,:,1] > cut).long()
    inds = pred_locs.sum(axis=1).nonzero()
    pred_strs = []
    pred_indices = []
    for ind_t in inds:
        ind = int(ind_t)
        pred_tokens = tokenizer.batch_decode(split_list(batch_ids[ind]*pred_locs[ind]),skip_special_tokens=True)
        pred_str = "|".join([t for t in pred_tokens if t])
        if pred_str: 
            pred_strs.append(pred_str)
            pred_indices.append(ind)
    return pred_strs,pred_indices

def extract_text_from_labels(batch_ids,batch_am,batch_labels,tokenizer,cut):
    inds = batch_labels.sum(axis=1).nonzero()
    true_strs = []
    true_indices = []
    for ind_t in inds:
        ind = int(ind_t)
        true_tokens = tokenizer.batch_decode(split_list(batch_ids[ind]*batch_labels[ind]),skip_special_tokens=True)
        true_str = "|".join([t for t in true_tokens if t])
        if true_str: 
            true_strs.append(true_str)
            true_indices.append(ind)
    return true_strs,true_indices

def calculate_precision_tp_fp_from_cfg_checkpoint(cfg,checkpt,true_tokens,cut=0.5,nmax=100,beta=0.5):

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
    m.eval()

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
        tp,fp = calculate_precision_from_pred_batch(batch_ids,batch_am,pred_locs,t,true_tokens)
        tot_tp += tp
        tot_fp += fp
    precision = tot_tp / (tot_tp + tot_fp)
    return precision,tot_tp,tot_fp

def calculate_precision_from_pred_batch(batch_ids,batch_am,pred_locs,tokenizer,true_tokens):
    tot_tp,tot_fp = 0,0
    pred_tokens = tokenizer.batch_decode(batch_ids*pred_locs,skip_special_tokens=True)
    tp,fp = calculate_precision(pred_tokens,true_tokens)
    tot_tp += tp
    tot_fp += fp
    return torch.tensor(tot_tp).float(),torch.tensor(tot_fp).float()
