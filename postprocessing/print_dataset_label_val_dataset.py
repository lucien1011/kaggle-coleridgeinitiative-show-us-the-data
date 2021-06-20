import os
import argparse

from torch.utils.data import DataLoader,TensorDataset,RandomSampler

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
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--nmax',type=int,default=10)
    parser.add_argument('--print_text',action='store_true')
    return parser.parse_args()

# __________________________________________________________________ ||
if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)
    
    model_dir = os.path.join(cfg.calculate_score_cfg.model_dir,args.checkpoint)
    model_args = getattr(cfg,"model_args",{})
    m = cfg.model.from_pretrained(model_dir,**model_args).to(args.device)
    m.eval()

    tokenizer = cfg.tokenizer.from_pretrained(cfg.base_pretrained)
 
    d = read_dataset(cfg)
    sampler = RandomSampler(d)
    dataloader = DataLoader(d,batch_size=args.batch_size,sampler=sampler)
    for step,batch in enumerate(dataloader):
        if step > args.nmax: break
        batch_ids = batch[0].to(args.device)
        batch_am = batch[1].to(args.device)
        batch_labels = batch[2].to(args.device)
        pred_strs,pred_indices = extract_text_from_pred(batch_ids,batch_am,m,tokenizer,args.cut)
        true_strs,true_indices = extract_text_from_labels(batch_ids,batch_am,batch_labels,tokenizer,args.cut)
        if pred_strs or true_strs:
            print(header)
            print("Pred: ",[(i,s) for i,s in zip(pred_indices,pred_strs)])
            print("True: ",[(i,s) for i,s in zip(true_indices,true_strs)])
            if args.print_text:
                for i in pred_indices:
                    print("pred ",i,tokenizer.decode(batch_ids[i],skip_special_tokens=True))
                for i in true_indices:
                    print("true ",i,tokenizer.decode(batch_ids[i],skip_special_tokens=True))
            print(header)
