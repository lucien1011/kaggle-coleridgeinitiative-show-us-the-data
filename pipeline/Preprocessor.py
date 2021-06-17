import os
import torch
import logging
import numpy as np
import pandas as pd
import bisect
import copy
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from utils.mkdir_p import mkdir_p
from utils.objdict import ObjDict
 
def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    l = l.tolist()
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def make_labels(input_ids,dataset_ids):
    seq_length = len(input_ids)
    output = [0 for _ in range(seq_length)] 
    for did in dataset_ids:
        indices = find_sub_list(did,input_ids)
        for start,end in indices:
            for ind in range(start,end+1):
                output[ind] = 1
    return output

def make_labels_by_offset_mapping(offset_mapping,dataset_indices,seq_length):
    output = [0 for _ in range(seq_length)] 
    start_indices,end_indices = map(list,zip(*offset_mapping))
    for dstart,dend in dataset_indices:
        start_index = bisect.bisect_left(start_indices,dstart)
        end_index = bisect.bisect_right(end_indices,dend)
        output[start_index:end_index] = [1 for _ in range(start_index,end_index)]
    return output

class Preprocessor(object):

    header = "*"*100

    def __init__(self,name='preprocess',log_level=logging.INFO):
        self.start_times = {}
        self.logger = self.make_logger(name,log_level,console=True)

    def make_logger(cls,name,level,filepath=None,console=True):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if filepath:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            logger.addHandler(fh)
        if console:
            console = logging.StreamHandler()
            console.setStream(tqdm)
            console.setLevel(level)
            logger.addHandler(console)
        return logger

    def print_header_message(self,message):
        self.logger.info(self.header)
        self.logger.info(message)
        self.logger.info(self.header)
    
    def start_count_time(self,key):
        if key not in self.start_times:
            self.start_times[key] = time.time()

    def print_elapsed_time(self,key):
        if key in self.start_times:
            elapsed_time = time.time() - self.start_times[key]
            self.print_header_message("Time used: {time:4.2f} seconds".format(time=elapsed_time))

    def tokenize_df(self,tokenizer,texts,tokenize_args):
        self.start_count_time("Tokenize")
        self.print_header_message("Tokenize text ")
        self.print_header_message("Dataframe shape: {shape:d}".format(shape=len(texts)))
        tokenized_inputs = tokenizer(texts,**tokenize_args)
        self.print_elapsed_time("Tokenize")
        return tokenized_inputs
    
    def create_preprocess_train_data(self,args):
        train_args = copy.deepcopy(args)
        train_args.input_csv_path = args.train_csv_path
        train_args.out_dir = args.preprocess_train_dir
        train_args.create_by_offset_mapping = getattr(args,"create_by_offset_mapping",False)
        
        train_dataset = self.create_preprocess_data(train_args)
        inputs = ObjDict(
                train_dataset = train_dataset,
                )
        return inputs

    def create_preprocess_test_data(self,args):
        test_args = copy.deepcopy(args)
        test_args.input_csv_path = args.test_csv_path
        test_args.out_dir = args.preprocess_test_dir
        test_args.create_by_offset_mapping = getattr(args,"create_by_offset_mapping",False)
        
        test_dataset = self.create_preprocess_data(test_args)
        inputs = ObjDict(
                test_dataset = test_dataset,
                )
        return inputs

    def load_preprocess_train_data(self,args):
        train_args = copy.deepcopy(args)
        train_args.input_csv_path = args.train_csv_path
        train_args.out_dir = args.preprocess_train_dir
        train_args.create_by_offset_mapping = getattr(args,"create_by_offset_mapping",False)
        train_args.mode = "train"

        train_dataset = self.load_preprocess_data(train_args)
        inputs = ObjDict(
                train_dataset = train_dataset,
                )
        return inputs

    def load_preprocess_test_data(self,args):
        test_args = copy.deepcopy(args)
        test_args.input_csv_path = args.test_csv_path
        test_args.out_dir = args.preprocess_test_dir
        test_args.create_by_offset_mapping = getattr(args,"create_by_offset_mapping",False)
        test_args.mode = "test"

        test_dataset = self.load_preprocess_data(test_args)
        inputs = ObjDict(
                test_dataset = test_dataset,
                )
        return inputs

    def load_preprocess_train_test_data(self,args):
        train_args = copy.deepcopy(args)
        train_args.input_csv_path = args.train_csv_path
        train_args.dataset_dir = args.preprocess_train_dir
        train_args.dataset_name = "train_dataset.pt"
        train_args.mode = "train"

        test_args = copy.deepcopy(args)
        test_args.input_csv_path = args.test_csv_path
        test_args.dataset_dir = args.preprocess_test_dir
        test_args.dataset_name = "test_dataset.pt"
        test_args.mode = "test"

        train_dataset = self.load_preprocess_data(train_args)
        test_dataset = self.load_preprocess_data(test_args)
        inputs = ObjDict(
                train_dataset = train_dataset,
                val_dataset = test_dataset,
                )
        return inputs

    def load_preprocess_data(self,args):
        dataset_path = os.path.join(args.dataset_dir,args.dataset_name)

        dataset_exists = os.path.exists(dataset_path)

        if dataset_exists:
            self.logger.info("[load_preprocess_"+args.mode+"_data] all dataset exists. Loading presaved dataset.")
            dataset = torch.load(dataset_path)
        else:
            input_ids = torch.load(os.path.join(args.dataset_dir,args.input_ids_name))
            attention_mask = torch.load(os.path.join(args.dataset_dir,args.attention_mask_name))
            dataset_mask = torch.load(os.path.join(args.dataset_dir,args.dataset_masks_name))
            sample_weight = torch.load(os.path.join(args.dataset_dir,args.sample_weight_name))
            labels = torch.load(os.path.join(args.dataset_dir,args.labels_name))
            dataset = TensorDataset(input_ids,attention_mask,dataset_mask,sample_weight,labels)

            pred_labels_path = os.path.join(args.dataset_dir,args.pred_labels_name)
            if os.path.exists(pred_labels_path):
                pred_labels = torch.load(pred_labels_path)
                dataset = TensorDataset(input_ids,attention_mask,dataset_mask,sample_weight,pred_labels,labels)
            else:
                dataset = TensorDataset(input_ids,attention_mask,dataset_mask,sample_weight,labels)

            if args.dataset_dir:
                mkdir_p(args.dataset_dir)
                torch.save(dataset,dataset_path)
        
        return dataset

    def create_preprocess_data(self,args):
        create_by_offset_mapping = getattr(args,"create_by_offset_mapping",False)
        if create_by_offset_mapping:
            dataset = self.create_preprocess_data_by_offset_mapping(args)
        else:
            dataset = self.create_preprocess_data_default(args)
        return dataset

    def create_preprocess_data_default(self,args):
        self.logger.info("[create_preprocess_data_default]")
        tokenizer = args.tokenizer
        df = pd.read_csv(args.input_csv_path)

        save_obj_key = ["input_ids","attention_mask","overflow_to_sample_mapping","labels","dataset_masks","offset_mapping","sample_weight"]
        out_path_dict = {k:os.path.join(args.out_dir,getattr(args,k+"_name"))  for k in save_obj_key}
        assert all([not os.path.exists(o) for o in out_path_dict.values()]), "preprocess data exists, please delete before create_preprocess_data"

        tokenized_inputs = self.tokenize_df(tokenizer,df.text.tolist(),args.tokenize_args)
        
        self.print_header_message("Make labels")
        labels = []
        dataset_masks = []
        ntext = len(tokenized_inputs['overflow_to_sample_mapping'])
        for i in tqdm(range(ntext)):

            df_index = int(tokenized_inputs['overflow_to_sample_mapping'][i])
            
            train_datasets = df['train_dataset'][df_index]
            external_datasets = df['external_dataset'][df_index]
            
            if not pd.isnull(train_datasets):
                train_datasets = train_datasets.split("|")
                tokenized_train_dataset = tokenizer(train_datasets)
                train_dataset_ids = [ts[1:-1] for ts in tokenized_train_dataset['input_ids']]
            else:
                train_dataset_ids = []

            if not pd.isnull(external_datasets):
                external_datasets = external_datasets.split("|")
                tokenized_external_dataset = tokenizer(external_datasets)
                external_dataset_ids = [ts[1:-1] for ts in tokenized_external_dataset['input_ids']]
            else:
                external_dataset_ids = []
            
            labels.append(make_labels(tokenized_inputs.input_ids[i],train_dataset_ids))
            dataset_masks.append(make_labels(tokenized_inputs.input_ids[i],external_dataset_ids))
        
        tokenized_inputs['labels'] = torch.tensor(labels)
        tokenized_inputs['dataset_masks'] = torch.tensor(dataset_masks)

        self.save_tokenized_input(tokenized_inputs,out_path_dict)
        
        dataset = TensorDataset(tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['dataset_masks'],tokenized_inputs['labels'],)
        return dataset

    def create_preprocess_data_by_offset_mapping(self,args):
        self.logger.info("[create_preprocess_data_by_offset_mapping]")
        tokenizer = args.tokenizer
        df = pd.read_csv(args.input_csv_path)
        
        make_pred_labels = "pred_dataset" in df.columns

        save_obj_key = ["input_ids","attention_mask","overflow_to_sample_mapping","labels","dataset_masks","offset_mapping","sample_weight"]
        if make_pred_labels:
            save_obj_key += ["pred_labels"]
        out_path_dict = {k:os.path.join(args.out_dir,getattr(args,k+"_name"))  for k in save_obj_key}
        assert all([not os.path.exists(o) for o in out_path_dict.values()]), "preprocess data exists, please delete before create_preprocess_data"

        tokenized_inputs = self.tokenize_df(tokenizer,df.text.tolist(),args.tokenize_args)

        self.print_header_message("Make labels")
        labels = []
        pred_labels = []
        dataset_masks = []
        ntext = len(tokenized_inputs['overflow_to_sample_mapping'])
        for i in tqdm(range(ntext)):
            df_index = int(tokenized_inputs['overflow_to_sample_mapping'][i])

            train_dataset_indices = self.make_indices_by_offset_mapping(df.text[df_index],df['train_dataset'][df_index],)
            external_dataset_indices = self.make_indices_by_offset_mapping(df.text[df_index],df['external_dataset'][df_index],)
            if make_pred_labels:
                pred_dataset_indices = self.make_indices_by_offset_mapping(df.text[df_index],df['pred_dataset'][df_index],)

            seq_length = len(tokenized_inputs.input_ids[i])
            offset_mapping = tokenized_inputs.offset_mapping[i].tolist()
            
            labels.append(make_labels_by_offset_mapping(offset_mapping,train_dataset_indices,seq_length))
            dataset_masks.append(make_labels_by_offset_mapping(offset_mapping,external_dataset_indices,seq_length))
            if make_pred_labels:
                pred_labels.append(make_labels_by_offset_mapping(offset_mapping,pred_dataset_indices,seq_length))
        
        tokenized_inputs['labels'] = torch.tensor(labels)
        if make_pred_labels: tokenized_inputs['pred_labels'] = torch.tensor(pred_labels)
        tokenized_inputs['dataset_masks'] = torch.tensor(dataset_masks)
        tokenized_inputs['sample_weight'] = torch.tensor(self.make_sample_weight(tokenized_inputs['overflow_to_sample_mapping'],df))

        self.save_tokenized_inputs(tokenized_inputs,out_path_dict,args.out_dir)
        
        if make_pred_labels:
            dataset = TensorDataset(tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['dataset_masks'],tokenized_inputs['pred_labels'],tokenized_inputs['labels'],)
        else:
            dataset = TensorDataset(tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['dataset_masks'],tokenized_inputs['labels'],)
        return dataset

    @classmethod
    def make_indices_by_offset_mapping(cls,text,datasets):
        if not pd.isnull(datasets):
            dataset_indices = []
            datasets = datasets.split("|")
            for d in datasets:
                start_index = text.index(d)
                end_index = start_index + len(d)
                dataset_indices.append((start_index,end_index))
        else:
            dataset_indices = []
        return dataset_indices

    def save_tokenized_inputs(self,tokenized_inputs,save_path_dict,out_dir):
        self.print_header_message("Saving")
        if out_dir:
            mkdir_p(out_dir)
            for k,path in save_path_dict.items():
                torch.save(tokenized_inputs[k],os.path.join(out_dir,path))
    
    @classmethod
    def make_sample_weight(cls,overflow_mapping,df):
        from collections import Counter
        tds = []
        for ds in df.train_dataset:
            if type(ds) == str:
                tds.extend(ds.split("|"))
        counts = Counter(tds)
        tot = len(tds)
        sample_weight = []
        for idx_t in overflow_mapping:
            idx = int(idx_t)
            if type(df.train_dataset[idx]) == str:
                sample_weight.append(np.prod([1./counts[d] for d in df.train_dataset[idx].split("|")]))
            else:
                sample_weight.append(0.)
        return sample_weight

    def include_pred_dataset_as_label(self,dataset,cfg):
        self.logger.info("[include_pred_dataset_as_label]")
        return TensorDataset(*[t for t in dataset.tensors[:-1]]+[torch.maximum(dataset.tensors[-1],dataset.tensors[4])])

    def include_external_dataset_as_label(self,dataset,cfg):
        self.logger.info("[include_external_dataset_as_label]")
        return TensorDataset(*[t for t in dataset.tensors[:-1]]+[torch.maximum(dataset.tensors[-1],dataset.tensors[2])])

    def mask_test_dataset_name_train(self,dataset,cfg):
        self.logger.info("[mask_test_dataset_name_train]")
        total_mask = torch.minimum(dataset.tensors[1],(dataset.tensors[2]==0).long())
        inputs = [dataset.tensors[0]]+[total_mask]+[t for t in dataset.tensors[2:]]
        return TensorDataset(*inputs)
