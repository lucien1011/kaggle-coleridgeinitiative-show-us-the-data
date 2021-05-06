import sys

from pipeline.pipeline_tokenclassifier import TokenClassifierPipeline

from utils.objdict import ObjDict
from utils.mkdir_p import mkdir_p

cfg = ObjDict.read_from_file_python3(sys.argv[1])

pp = ClassifierPipeline()

pp.set_cfg(cfg)
pp.read_np_dir()
pp.train()
pp.save()
