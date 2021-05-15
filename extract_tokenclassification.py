import os
import sys
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

# __________________________________________________________________ ||
batch_size = 1

# __________________________________________________________________ ||
if cfg.extract_cfg.test:
    inputs = cfg.pipeline.load_preprocess_test_data(cfg.preprocess_cfg)
else:
    inputs = cfg.pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
model = cfg.model.from_pretrained(cfg.extract_cfg.pretrain_model)

# __________________________________________________________________ ||
model = model.to(cfg.extract_cfg.device)
softmax = torch.nn.Softmax(dim=-1)
t = open(cfg.extract_cfg.extract_text_path,"w")
for step in tqdm(range(len(inputs.input_ids))):
    input_id,mask = inputs.input_ids[step],inputs.attention_mask[step]
    if step % cfg.extract_cfg.write_per_step != 0: continue
    input_id = input_id.to(cfg.extract_cfg.device)
    mask = mask.to(cfg.extract_cfg.device)
    with torch.no_grad():
        preds = model(input_ids=torch.unsqueeze(input_id,0),attention_mask=torch.unsqueeze(mask,0))
        probs = softmax(preds.logits)
        labels = torch.argmax(probs,axis=2)
    highlighted_tokens = [str(t) for i,t in enumerate(cfg.tokenizer.convert_ids_to_tokens(input_id)) if labels[0,i] != 0]
    tqdm.write(str(highlighted_tokens))
    t.write("\n".join([
        "*"*50,
        "step "+str(step),
        #str(inputs.ids[step]) if cfg.extract_cfg.test else "",
        "Number of predicted label: "+str(int(torch.sum(labels))),
        "*"*50,
        " ".join(cfg.tokenizer.convert_ids_to_tokens(input_id)),
        "*"*50,
        "Predicted tokens: "+" ".join(highlighted_tokens) if highlighted_tokens else "",
        "*"*50,
        "",
        ]))
            
