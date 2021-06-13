import os
import glob
import pickle
import torch
import logging
import random
import numpy as np
import pandas as pd
import bisect
import copy

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchmetrics.functional import accuracy,auroc,f1,precision,recall
from tqdm import tqdm, trange

from pipeline import Pipeline
from metrics.specificity import specificity
from utils.mkdir_p import mkdir_p
from utils.objdict import ObjDict

logger = logging.getLogger(__name__)

softmax = torch.nn.Softmax(dim=-1)
        
from torch.nn import CosineSimilarity
cos = CosineSimilarity(dim=1,eps=1e-6)

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

class TokenMultiClassifierPipeline(Pipeline):

    @classmethod
    def compute_metrics(cls,preds,labels,num_classes,islogit=False,average=None,smooth=0.5):
        if islogit:
            probs = softmax(preds)
        else:
            probs = softmax(preds.logits)

        pred_labels = torch.argmax(probs,axis=2)

        pred_labels_flatten = pred_labels.flatten(0,1)
        labels_flatten = labels.flatten(0,1)
        
        return {
            "accuracy": accuracy(pred_labels_flatten,labels_flatten,num_classes=num_classes,average='micro',),
            "f1": f1(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1],
            "precision": precision(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1],
            "recall": recall(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1],
            "ntf": labels_flatten.sum(),
        }
    
    @classmethod
    def patch_train_batch(cls,batch):
        return {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "sample_weight": batch[3],
                "labels": batch[-1],
                }

    @classmethod
    def patch_test_batch(cls,batch):
        return {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
                }
    
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
            self.print_message("[load_preprocess_"+args.mode+"_data] all dataset exists. Loading presaved dataset.")
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
        self.print_message("[create_preprocess_data_default]")
        tokenizer = args.tokenizer
        df = pd.read_csv(args.input_csv_path)

        save_obj_key = ["input_ids","attention_mask","overflow_to_sample_mapping","labels","dataset_masks","offset_mapping","sample_weight"]
        out_path_dict = {k:os.path.join(args.out_dir,getattr(args,k+"_name"))  for k in save_obj_key}
        assert all([not os.path.exists(o) for o in out_path_dict.values()]), "preprocess data exists, please delete before create_preprocess_data"

        tokenized_inputs = self.tokenize_df(tokenizer,df.text.tolist(),args.tokenize_args)
        
        print("Make labels")
        self.print_header()
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
        self.print_message("[create_preprocess_data_by_offset_mapping]")
        tokenizer = args.tokenizer
        df = pd.read_csv(args.input_csv_path)
        
        make_pred_labels = "pred_dataset" in df.columns

        save_obj_key = ["input_ids","attention_mask","overflow_to_sample_mapping","labels","dataset_masks","offset_mapping","sample_weight"]
        if make_pred_labels:
            save_obj_key += ["pred_labels"]
        out_path_dict = {k:os.path.join(args.out_dir,getattr(args,k+"_name"))  for k in save_obj_key}
        assert all([not os.path.exists(o) for o in out_path_dict.values()]), "preprocess data exists, please delete before create_preprocess_data"

        tokenized_inputs = self.tokenize_df(tokenizer,df.text.tolist(),args.tokenize_args)

        self.print_header()
        print("Make labels")
        self.print_header()
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

    @classmethod
    def save_tokenized_inputs(cls,tokenized_inputs,save_path_dict,out_dir):
        cls.print_header()
        print("Saving")
        cls.print_header()
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

    def randomize_train_data(self,inputs,cfg):
        self.print_message("[randomize_train_data]")
        assert cfg.randomize_cfg.fraction > 0. and cfg.randomize_cfg.fraction < 1.
        train_dataset = inputs.train_dataset
        mask = torch.rand(train_dataset.tensors[-1].shape) < cfg.randomize_cfg.fraction
        pos_label = train_dataset.tensors[-1].bool()
        train_dataset.tensors[0][pos_label*mask] = torch.randint(3,cfg.tokenizer.vocab_size,(torch.sum(mask*pos_label),))
        inputs.train_dataset = TensorDataset(*train_dataset.tensors)

    def include_pred_dataset_as_label(self,dataset,cfg):
        self.print_message("[include_pred_dataset_as_label]")
        return TensorDataset(*[t for t in dataset.tensors[:-1]]+[torch.maximum(dataset.tensors[-1],dataset.tensors[4])])

    def include_external_dataset_as_label(self,dataset,cfg):
        self.print_message("[include_external_dataset_as_label]")
        return TensorDataset(*[t for t in dataset.tensors[:-1]]+[torch.maximum(dataset.tensors[-1],dataset.tensors[2])])

    def mask_test_dataset_name_train(self,dataset,cfg):
        self.print_message("[mask_test_dataset_name_train]")
        total_mask = torch.minimum(dataset.tensors[1],(dataset.tensors[2]==0).long())
        inputs = [dataset.tensors[0]]+[total_mask]+[t for t in dataset.tensors[2:]]
        return TensorDataset(*inputs)

    def predict(self,inputs,model,args):
        checkpts = self.get_model_checkpts(args.model_dir,args.model_key)
        mkdir_p(args.output_dir) 
        dataset = getattr(inputs,args.val_dataset_name)
        for c in checkpts:
            self.print_message("Processing checkpoint "+c) 
            cdir = os.path.join(args.output_dir,c)
            mkdir_p(cdir)
            if len(glob.glob(os.path.join(cdir,"*.pt"))) > 0: 
                self.print_message("Skipping "+cdir)
                continue
            model = model.from_pretrained(os.path.join(args.model_dir,c)).to(args.device)
            model.eval()
            data_sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=args.batch_size,sampler=data_sampler,)
            iterator = tqdm(dataloader, desc="Iteration",)
            for step,batch in enumerate(iterator):
                batch = tuple(t.to(args.device) for t in batch)
                overflow_mapping = batch[2]
                batch = {"input_ids": batch[0],"attention_mask": batch[1],}
                with torch.no_grad():
                    preds = model(**batch)
                torch.save(preds,os.path.join(args.output_dir,c,args.pred_name+"_"+str(step)+args.pred_extension)) 
                torch.save(overflow_mapping,os.path.join(args.output_dir,c,"overflow_mapping_"+str(step)+args.pred_extension)) 

    def extract(self,inputs,model,args):
        checkpts = self.get_model_checkpts(args.model_dir,args.model_key)
        mkdir_p(args.output_dir) 
        dataset = getattr(inputs,args.dataset_name)
        for c in checkpts:
            self.print_message("Processing checkpoint "+c) 
            cdir = os.path.join(args.output_dir,c)
            mkdir_p(cdir)
            if len(glob.glob(os.path.join(cdir,"*.pt"))) > 0: 
                self.print_message("Skipping "+cdir)
                continue
            model = model.from_pretrained(os.path.join(args.model_dir,c)).to(args.device)
            model.eval()
            dataloader = DataLoader(dataset, batch_size=args.batch_size)
            iterator = tqdm(dataloader, desc="Iteration",)
            for step,batch in enumerate(iterator):
                batch = tuple(t.to(args.device) for t in batch)
                overflow_mapping = batch[2]
                batch = {"input_ids": batch[0],"attention_mask": batch[1],}
                with torch.no_grad():
                    output = model(**batch)
                    idx = torch.argmax(output.logits,axis=2)
                    tokens = torch.where(idx*batch['attention_mask'] != 0,batch['input_ids'],-1)
                extract_file_name,extract_file_extension = os.path.splitext(args.extract_file_name)
                torch.save(tokens,os.path.join(args.output_dir,c,extract_file_name+"_"+str(step)+extract_file_extension)) 
                torch.save(overflow_mapping,os.path.join(args.output_dir,c,"overflow_mapping_"+str(step)+extract_file_extension)) 
 
    def evaluate(self,inputs,model,args):
        checkpts = self.get_model_checkpts(args.model_dir,args.model_key)
        mkdir_p(args.output_dir) 
        dataset = getattr(inputs,args.val_dataset_name)
        for c in checkpts:
            self.print_message("Processing checkpoint "+c) 
            cdir = os.path.join(args.output_dir,c)
            mkdir_p(cdir)
            if len(glob.glob(os.path.join(cdir,"*.pt"))) > 0: 
                self.print_message("Skipping "+cdir)
                continue
            model = model.from_pretrained(os.path.join(args.model_dir,c)).to(args.device)
            model.eval()
            data_sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=args.batch_size,sampler=data_sampler,)
            iterator = tqdm(dataloader, desc="Iteration",)
            scores = {}
            for step,batch in enumerate(iterator):
                if step > args.n_sample: continue
                batch = tuple(t.to(args.device) for t in batch)
                batch = self.patch_test_batch(batch)
                if int(batch['labels'].sum()) == 0: continue
                with torch.no_grad():
                    preds = model(**batch)
                metrics = self.compute_metrics(preds,batch['labels'],num_classes=model.num_labels)
                for name,value in metrics.items():
                    if name not in scores:
                        scores[name] = {}
                    else:
                        scores[name][step] = value

            for name,value in scores.items():
                print(name+": ",str(np.mean([float(m) for s,m in value.items()])))

    @classmethod
    def calculate_min_context_similarity(cls,model,dataset_emb,pred_label,seq_length):
        """
        dataset_emb: [ number of dataset names , sequence_length , embedding_dimension ]
        pred_label: [ number of predicted tokens ]
        model: embedding model
        """
        pred_length = len(pred_label)
        pred_length = pred_label.numel()
        pred_label = torch.nn.functional.pad(pred_label,pad=(1,0),value=101)
        pred_label = torch.nn.functional.pad(pred_label,pad=(0,1),value=102)
        pred_label = torch.nn.functional.pad(pred_label,pad=(0,seq_length-pred_length-2),value=0)
        attention_mask = (pred_label != 0).long()
        with torch.no_grad():
            pred_emb = model(
                    input_ids=torch.unsqueeze(pred_label,axis=0),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    ).hidden_states[-1][:,0,:]
        sim_matrix = cos(dataset_emb,pred_emb)
        #context_sim = torch.min(torch.sqrt(torch.sum(sim_matrix**2,axis=1)))
        context_sim = torch.max(sim_matrix)
        return context_sim

