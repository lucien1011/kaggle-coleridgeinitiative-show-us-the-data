import os
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from postprocessing.tools import calculate_fbeta_tp_fp_fn_from_cfg_checkpoint

header = "*"*100
ofname = "fbeta_vs_epoch.png"

# __________________________________________________________________ ||
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--nmax',type=int,default=1e7)
    parser.add_argument('--model_key',type=str,default='checkpoint-epoch-')
    parser.add_argument('--cut',type=float,default=0.5)
    parser.add_argument('--plot_per_checkpt',type=int,default=1)
    parser.add_argument('-p','--plot_to_path',type=str,default="")
    parser.add_argument('--plot_to_model_dir',action='store_true')
    return parser.parse_args()


# __________________________________________________________________ ||
if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)

    checkpts = cfg.pipeline.get_model_checkpts(cfg.calculate_score_cfg.model_dir,args.model_key)
    assert len(checkpts) > 0
    assert all([c.replace(args.model_key,"").isdigit() for c in checkpts])
    checkpts.sort(key=lambda x: int(x.replace(args.model_key,"")))

    xs,ys = [],[]
    for ic,checkpt in enumerate(checkpts):
        if ic % args.plot_per_checkpt != 0: continue
        fbeta_score,tot_tp,tot_fp,tot_fn = calculate_fbeta_tp_fp_fn_from_cfg_checkpoint(cfg,checkpt,args.cut,args.nmax)
        x = int(checkpt.replace(args.model_key,""))
        y = fbeta_score
        print(checkpt,x,y)
        xs.append(x)
        ys.append(y)

    fig,ax = plt.subplots()
    ax.plot(xs,ys)
    ax.set_ylabel("Fbeta Score")
    ax.set_xlabel("Epoch")
    if args.plot_to_path:
        fig.savefig(args.plot_to_path)
    elif args.plot_to_model_dir:
        fig.savefig(os.path.join(cfg.calculate_score_cfg.model_dir,ofname))
