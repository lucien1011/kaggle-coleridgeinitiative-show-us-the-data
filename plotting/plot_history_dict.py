import os
import pickle
import sys
import matplotlib.pyplot as plt

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

cfg = ObjDict.read_from_file_python3(sys.argv[1])

histd = pickle.load(open(cfg.saved_history_path,'rb'))

for ks in sys.argv[2].split(","):
    plt.clf()
    for k in ks.split(":"):
        epochs = histd[k]
        nepoch = len(histd[k])
        plt.plot(range(nepoch),histd[k],label=k)
    plt.legend(loc='best')
    plt.savefig(os.path.join(cfg.saved_model_path,ks+'.png'))
