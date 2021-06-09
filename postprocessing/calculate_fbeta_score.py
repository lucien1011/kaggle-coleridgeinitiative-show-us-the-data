import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from postprocessing.tools import calculate_fbeta_tp_fp_fn_from_cfg_checkpoint

# __________________________________________________________________ ||
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str)
    parser.add_argument('--nmax',type=int,default=1e7)
    parser.add_argument('--checkpoint',type=str,default='checkpoint-epoch-1')
    parser.add_argument('--cut',type=float,default=0.5)
    return parser.parse_args()

# __________________________________________________________________ ||
if __name__ == "__main__":

    args = parse_arguments()

    cfg = ObjDict.read_all_from_file_python3(args.input_path)
    fbeta_score,tot_tp,tot_fp,tot_fn = calculate_fbeta_tp_fp_fn_from_cfg_checkpoint(cfg,args.checkpoint,args.cut,args.nmax)

    print("fbeta: ",str(fbeta_score))
    print("total TP: ",str(tot_tp))
    print("total FP: ",str(tot_fp))
    print("total FN: ",str(tot_fn))

