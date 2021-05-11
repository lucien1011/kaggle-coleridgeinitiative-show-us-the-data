import os
import sys
import torch

from transformers import AutoConfig,AutoTokenizer,AutoModelForTokenClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pipeline.pipeline_tokenclassifier import compute_metrics
from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

# __________________________________________________________________ ||
cfg = ObjDict.read_all_from_file_python3(sys.argv[1])

# __________________________________________________________________ ||
batch_size = 1

# __________________________________________________________________ ||
#inputs = cfg.pipeline.load_preprocess_sequence_train_data(cfg.preprocess_cfg)
inputs = cfg.pipeline.load_preprocess_sequence_test_data(cfg.preprocess_cfg)
model = AutoModelForTokenClassification.from_pretrained(cfg.evaluate_cfg.pretrain_model,config=AutoConfig.from_pretrained(cfg.base_pretrained))

# __________________________________________________________________ ||
model = model.to(cfg.evaluate_cfg.device)
softmax = torch.nn.Softmax(dim=-1)
t = open(cfg.evaluate_cfg.output_text_path,"w")
for step,(input_id,mask) in enumerate(zip(inputs.input_ids,inputs.attention_mask)):
    if step % 1 != 0: continue
    input_id = input_id.to(cfg.evaluate_cfg.device)
    mask = mask.to(cfg.evaluate_cfg.device)
    with torch.no_grad():
        preds = model(input_ids=torch.unsqueeze(input_id,0),attention_mask=torch.unsqueeze(mask,0))
        probs = softmax(preds.logits)
        labels = torch.argmax(probs,axis=2)
    highlighted_tokens = [str(t) if labels[0,i] != 1 else "[[[["+str(t)+"]]]]"  for i,t in enumerate(cfg.tokenizer.convert_ids_to_tokens(input_id))]
    t.write("\n".join([
        "*"*50,
        "step "+str(step),
        "Number of predicted label: "+str(torch.sum(labels)),
        "*"*50,
        " ".join(highlighted_tokens),
        " ",
        ]))
            
