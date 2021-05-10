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

pretrain_model = "log/optimise_tokenclassification_210508_01_hasDataset/checkpoint-8200/"
device = 'cuda'
batch_size = 1
#output_text_path = 'tmp/predict_test_tokenclassification.txt'
output_text_path = 'tmp/predict_train_tokenclassification.txt'

# __________________________________________________________________ ||
#inputs = cfg.pipeline.load_preprocess_test_data(cfg.preprocess_cfg)
inputs = cfg.pipeline.load_preprocess_train_data(cfg.preprocess_cfg)
model = AutoModelForTokenClassification.from_pretrained(pretrain_model,config=AutoConfig.from_pretrained(cfg.base_pretrained))

# __________________________________________________________________ ||
model = model.to(device)
softmax = torch.nn.Softmax(dim=-1)
t = open(output_text_path,"w")
for step,(input_id,mask) in enumerate(zip(inputs.input_ids,inputs.attention_mask)):
    if step % 50 != 0: continue
    input_id = input_id.to(device)
    mask = mask.to(device)
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
            
