import sys

from utils.objdict import ObjDict

cfg = ObjDict.read_from_file_python3(sys.argv[1])

pp = cfg.pp
pp.set_cfg(cfg)
pp.read_np_dir()
pp.train()
pp.save()
