import os
import numpy as np
import torch

from transformers import DistilBertModel,DistilBertTokenizer,DistilBertConfig
from sklearn.cluster import KMeans

from pipeline.pipeline_datasetcluster import DatasetClusterPipeline
from utils.objdict import ObjDict

# __________________________________________________________________ ||
name = "DatasetCluster_kMeans_210514_01"
base_pretrained = "distilbert-base-uncased"
preprocess_train_dir = "data/DatasetCluster_kMeans_210514_01/train/" 
preprocess_test_dir = "data/DatasetCluster_kMeans_210514_01/test/" 

# __________________________________________________________________ ||
pipeline = DatasetClusterPipeline()

embedding_model = DistilBertModel.from_pretrained('model/'+base_pretrained)
tokenizer = DistilBertTokenizer.from_pretrained('tokenizer/'+base_pretrained)

model = KMeans(n_clusters=20, random_state=0)

# __________________________________________________________________ ||
preprocess_cfg = ObjDict(
    train_csv_path = "data/train_sequence_has_dataset.csv",
    tokenizer = tokenizer,
    embedding_model = embedding_model,
    preprocess_train_dir = preprocess_train_dir,
    load_preprocess = True,
    )

# __________________________________________________________________ ||
train_cfg = ObjDict(
        nKs = 60,
        random_state=0,
        max_iter=1000,
        output_dir = os.path.join('log/',name),
        )

# __________________________________________________________________ ||
evaluate_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-8000"),
        device = 'cuda',
        batch_size = 256,
        test = True,
        )

# __________________________________________________________________ ||
extract_cfg = ObjDict(
        pretrain_model = os.path.join("log",name,"checkpoint-8000"),
        device = 'cpu',
        test = True,
        write_predicted_only = True,
        write_per_step = 1,
        extract_text_path = os.path.join('log',name,'extract_test_8000.txt'),
        )

# __________________________________________________________________ ||
slurm_job_dir = os.path.join('job/',name+'/')
slurm_cfg = ObjDict(
    name = name,
    slurm_cfg_name = 'submit.cfg',
    slurm_job_dir = slurm_job_dir,
    slurm_commands = """echo \"{job_name}\"
cd {base_path}
source setup_hpg.sh
python3 {pyscript} {cfg_path}
""".format(
            job_name=name,
            pyscript="run_pipeline.py",
            cfg_path="config/"+name+".py",
            base_path=os.environ['BASE_PATH'],
            output_path=slurm_job_dir,
            ),
    )
