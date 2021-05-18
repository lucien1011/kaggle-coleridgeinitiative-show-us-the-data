import os
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
device = 'cuda'
plot_per_checkpt = 10

# __________________________________________________________________ ||
def calculate_dataset_emb(model,dataset_inputs,device):
    with torch.no_grad():
        dataset_emb = model(
                input_ids=dataset_inputs['input_ids'].to(device),
                attention_mask=dataset_inputs['attention_mask'].to(device),
                output_hidden_states=True,
                ).hidden_states[-1]
    return dataset_emb

# __________________________________________________________________ ||
def calculate_similarity(model,test_dataset,dataset_emb,device):

    from torch.utils.data import DataLoader, RandomSampler 

    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)
    with torch.no_grad():
        
        seq_length = int(dataset_emb.size(1))
    
        min_context_sims = []
        
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            batch_test = {"input_ids": batch[0],"attention_mask": batch[1], "output_hidden_states": True,}
            batch_model_out = model(**batch_test)
            
            _,pred_label_idx = torch.split((torch.argmax(batch_model_out.logits,axis=2)!=0).nonzero(),1,dim=1,)
            pred_label_idx = torch.squeeze(pred_label_idx,axis=1)

            if pred_label_idx.numel():
                pred_label = torch.squeeze(batch_test['input_ids'],axis=0)[pred_label_idx]
                min_context_sims.append(pipeline.calculate_min_context_similarity(model,dataset_emb,pred_label))
        return torch.mean(torch.stack(min_context_sims)),len(min_context_sims)

# __________________________________________________________________ ||
if __name__ == "__main__":

    cfg = ObjDict.read_all_from_file_python3(sys.argv[1])
    pipeline = cfg.pipeline

    inputs = pipeline.load_preprocess_test_data(cfg.preprocess_cfg)
    dataset_inputs = torch.load(cfg.evaluate_cfg.dataset_tokens_path)
 
    sim_dict = {}
    npred_dict = {}
    checkpts = [c for c in os.listdir(cfg.train_cfg.output_dir) if "checkpoint" in c]
    for i in tqdm(range(len(checkpts))):
        if i % plot_per_checkpt != 0: continue
        c = checkpts[i]
        model = cfg.model.from_pretrained(os.path.join(cfg.train_cfg.output_dir,c)).to(device)
        dataset_emb = calculate_dataset_emb(model,dataset_inputs,device)
        sim,npred = calculate_similarity(model,inputs.test_dataset,dataset_emb,device)
        
        c_int = int(c.replace("checkpoint-",""))
        sim_dict[c_int] = sim
        npred_dict[c_int] = npred

    x = list(sim_dict.keys())
    x.sort()
    sims = [sim_dict[i] for i in x]
    npreds = [npred_dict[i] for i in x]
    
    fig, ax = plt.subplots(2,1,figsize=(15, 8))
    ax[0].plot(x,sims)
    ax[0].set_ylabel("maximum similarity")
    ax[0].set_xlabel("training step")
    
    ax[1].plot(x,npreds)
    ax[1].set_ylabel("number of prediction")
    ax[1].set_xlabel("training step")
    
    mkdir_p(cfg.train_cfg.output_dir)
    fig.savefig(os.path.join(cfg.train_cfg.output_dir,"metric_vs_training_step.png"))
