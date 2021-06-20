import os
import argparse
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from postprocessing.tools import calculate_precision_tp_fp_from_cfg_checkpoint

header = "*"*100
ofname = "precision_vs_epoch.png"

# __________________________________________________________________ ||
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--nmax',type=int,default=1e7)
    parser.add_argument('--model_key',type=str,default='checkpoint-epoch-')
    parser.add_argument('--model_dir',type=str,default='')
    parser.add_argument('--cut',type=float,default=0.5)
    parser.add_argument('--plot_per_checkpt',type=int,default=1)
    parser.add_argument('-p','--plot_to_path',type=str,default="")
    parser.add_argument('--plot_to_model_dir',action='store_true')
    parser.add_argument('--dataset_path',type=str,default="storage/input/rcdataset/data_sets_unique_map.p")
    return parser.parse_args()

def get_model_checkpts(input_dir,key):
    return [c for c in os.listdir(input_dir) if key in c and os.path.isdir(os.path.join(input_dir,c))]

# __________________________________________________________________ ||
if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)

    datasets = list(pickle.load(open(args.dataset_path,"rb")).keys())
    if args.model_dir:
        cfg.calculate_score_cfg.model_dir = args.model_dir
    checkpts = get_model_checkpts(cfg.calculate_score_cfg.model_dir,args.model_key)
    assert len(checkpts) > 0
    assert all([c.replace(args.model_key,"").isdigit() for c in checkpts])
    checkpts.sort(key=lambda x: int(x.replace(args.model_key,"")))

    xs,ys = [],[]
    for ic,checkpt in enumerate(checkpts):
        if ic % args.plot_per_checkpt != 0: continue
        precision,tot_tp,tot_fp = calculate_precision_tp_fp_from_cfg_checkpoint(cfg,checkpt,datasets,args.cut,args.nmax)
        x = int(checkpt.replace(args.model_key,""))
        y = precision 
        print(checkpt,x,y)
        xs.append(x)
        ys.append(y)

    fig,ax = plt.subplots()
    ax.plot(xs,ys)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Epoch")
    if args.plot_to_path:
        fig.savefig(args.plot_to_path)
    elif args.plot_to_model_dir:
        fig.savefig(os.path.join(cfg.calculate_score_cfg.model_dir,ofname))
