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
device = 'cpu'
plot_per_checkpt = 5

# __________________________________________________________________ ||
def calculate_dataset_emb(model,dataset_inputs,device):
    dataset_emb = model(
            input_ids=dataset_inputs['input_ids'],
            attention_mask=dataset_inputs['attention_mask'],
            output_hidden_states=True,
            ).hidden_states[-1][:,0,:].to(device)
    return dataset_emb

# __________________________________________________________________ ||
def calculate_similarity(model,inputs,dataset_emb,device,seq_length):
  
    min_context_sims = []
    
    ndata = len(inputs.input_ids)
    for step in range(ndata):
        batch_test = {
                "input_ids": inputs.input_ids[step].unsqueeze(0).to(device),
                "attention_mask": inputs.attention_mask[step].unsqueeze(0).to(device),
                "output_hidden_states": True,
                }
        outputs = model(**batch_test)
        
        _,pred_label_idx = torch.split((torch.argmax(outputs.logits,axis=2)!=0).nonzero(),1,dim=1,)
        pred_label_idx = torch.squeeze(pred_label_idx,axis=1)

        if pred_label_idx.numel():
            pred_label = torch.squeeze(batch_test['input_ids'],axis=0)[pred_label_idx]
            min_context_sims.append(pipeline.calculate_min_context_similarity(model,dataset_emb,pred_label,seq_length))
    return torch.mean(torch.stack(min_context_sims)),len(min_context_sims)

# __________________________________________________________________ ||
if __name__ == "__main__":

    cfg = ObjDict.read_all_from_file_python3(sys.argv[1])
    pipeline = cfg.pipeline

    print("Loading inputs")
    inputs = pipeline.load_preprocess_test_data(cfg.preprocess_cfg)
    dataset_inputs = torch.load(cfg.evaluate_cfg.dataset_tokens_path).to(device)
    print("Finsih loading inputs")
 
    sim_dict = {}
    npred_dict = {}
    checkpts = [c for c in os.listdir(cfg.train_cfg.output_dir) if "checkpoint" in c]
    checkpts.sort()
    for i in tqdm(range(len(checkpts))):
        
        if i % plot_per_checkpt != 0: continue
        c = checkpts[i]
        tqdm.write("Processing checkpoint "+c)
        
        model = cfg.model.from_pretrained(os.path.join(cfg.train_cfg.output_dir,c)).to(device)
        with torch.no_grad():
            dataset_emb = calculate_dataset_emb(model,dataset_inputs,device)
            sim,npred = calculate_similarity(model,inputs,dataset_emb,device,dataset_inputs.input_ids.shape[1])
        
        tqdm.write("Finish processing checkpoint "+c)
        
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
