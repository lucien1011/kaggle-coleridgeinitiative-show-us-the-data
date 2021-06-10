import os
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from postprocessing.tools import calculate_fbeta_tp_fp_fn_from_cfg_checkpoint

header = "*"*100
ofname = "fbeta_vs_cut.png"

# __________________________________________________________________ ||
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--nmax',type=int,default=1e7)
    parser.add_argument('--checkpoint',type=str,default='checkpoint-epoch-1')
    parser.add_argument('--cuts',type=str,default="0.5")
    parser.add_argument('-p','--plot_to_path',type=str,default="")
    parser.add_argument('--plot_to_model_dir',action='store_true')
    return parser.parse_args()

# __________________________________________________________________ ||
if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)

    cuts = [float(n) for n in args.cuts.split(",")]
    assert len(cuts) > 0
    assert all([type(c) == float for c in cuts])
    assert args.plot_to_path or args.plot_to_model_dir
    cuts.sort()

    x,y = [],[]
    for cut in cuts:
        fbeta_score,tot_tp,tot_fp,tot_fn = calculate_fbeta_tp_fp_fn_from_cfg_checkpoint(cfg,args.checkpoint,cut,args.nmax)
        x.append(cut)
        y.append(fbeta_score)

    fig,ax = plt.subplots()
    ax.plot(x,y)
    ax.set_ylabel("Fbeta Score")
    ax.set_xlabel("cut")
    if args.plot_to_path:
        fig.savefig(args.plot_to_path)
    elif args.plot_to_model_dir:
        fig.savefig(os.path.join(cfg.calculate_score_cfg.model_dir,ofname))
