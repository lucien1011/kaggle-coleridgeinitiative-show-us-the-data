import os
import argparse
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset,RandomSampler
from collections import defaultdict
from tqdm import tqdm

from utils.objdict import ObjDict
from postprocessing.tools import read_dataset,extract_text_from_pred,extract_text_from_labels

header = "*"*100

# __________________________________________________________________ ||
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--checkpoint',type=str,default='checkpoint-epoch-1')
    parser.add_argument('--cut',type=float,default=0.5)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--out_per_point',type=int,default=1)
    parser.add_argument('--ofname',type=str,default="pred_summary.csv")
    parser.add_argument('--csv_name',type=str,default="test_csv_path")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)

    model_dir = os.path.join(cfg.calculate_score_cfg.model_dir,args.checkpoint)
    model_args = getattr(cfg,"model_args",{})
    m = cfg.model.from_pretrained(model_dir,**model_args).to(args.device)
    m.eval()

    tokenizer = cfg.tokenizer.from_pretrained(cfg.base_pretrained)
    
    csv_path = getattr(cfg.preprocess_cfg,args.csv_name)
    df = pd.read_csv(csv_path)

    outdict = defaultdict(list)
    for idx in tqdm(range(len(df))):
        if idx % args.out_per_point != 0: continue
        text = df.text[idx]
        train_dataset = df.train_dataset[idx]
        external_dataset = df.external_dataset[idx]
        inputs = tokenizer(text,padding='max_length',max_length=512,truncation=True,return_overflowing_tokens=True,return_tensors='pt',)
        pred_strs,_ = extract_text_from_pred(inputs.input_ids.to(args.device),inputs.attention_mask.to(args.device),m,tokenizer,args.cut)
        outdict["pred_dataset"].append("|".join(list(set(pred_strs))))
        outdict["train_dataset"].append(train_dataset)
        outdict["external_dataset"].append(external_dataset)
        outdict["text"].append(text)
    outdf = pd.DataFrame(outdict)
    outdf.to_csv(os.path.join(model_dir,args.ofname),index=False)
