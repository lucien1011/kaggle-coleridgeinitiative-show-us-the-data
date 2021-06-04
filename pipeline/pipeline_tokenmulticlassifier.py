import os
import glob
import pickle
import torch
import logging
import random
import numpy as np
import pandas as pd

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

        intersection = pred_labels * labels
        union = pred_labels + labels - intersection
        IoU = (intersection + smooth)/(union + smooth)
        
        return {
            "IoU": IoU.mean(),
            "accuracy": accuracy(pred_labels_flatten,labels_flatten,num_classes=num_classes,average='micro',),
            "f1": f1(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1],
            "precision": precision(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1],
            "recall": recall(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1],
            "ntf": labels_flatten.sum(),
            ##"auroc": auroc(probs,labels_flatten),
            #"f1": f1(probs,labels_flatten,),#num_classes=num_classes,average=average,),
            #"precision": precision(probs,labels_flatten,),#num_classes=num_classes,average=average,),
            #"recall": recall(probs,labels_flatten,),#num_classes=num_classes,average=average,),
            ##"specificity": specificity(probs,labels,num_classes=num_classes),
        }
    
    @classmethod
    def patch_batch(cls,batch):
        return {"input_ids": batch[0],"attention_mask": batch[3],"labels": batch[-1]}

    def create_preprocess_train_data(self,args):
        train_args = ObjDict(
                tokenizer = args.tokenizer,
                input_csv_path = args.train_csv_path,
                out_dir = args.preprocess_train_dir,
                input_ids_name = "input_ids.pt",
                attention_mask_name = "attention_mask.pt",
                labels_name = "labels.pt",
                dataset_masks_name = "dataset_masks.pt",
                overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
                )
        train_dataset = self.create_preprocess_data(train_args)
        inputs = ObjDict(
                train_dataset = train_dataset,
                )
        return inputs

    def create_preprocess_test_data(self,args):
        test_args = ObjDict(
                tokenizer = args.tokenizer,
                input_csv_path = args.test_csv_path,
                out_dir = args.preprocess_test_dir,
                input_ids_name = "input_ids.pt",
                attention_mask_name = "attention_mask.pt",
                labels_name = "labels.pt",
                dataset_masks_name = "dataset_masks.pt",
                overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
                )
        test_dataset = self.create_preprocess_data(test_args)
        inputs = ObjDict(
                test_dataset = test_dataset,
                )
        return inputs

    def load_preprocess_train_data(self,args):
        train_args = ObjDict(
                dataset_dir = args.preprocess_train_dir,
                dataset_name = "train_dataset.pt",
                mode = "train",
                input_ids_name = "input_ids.pt",
                attention_mask_name = "attention_mask.pt",
                labels_name = "labels.pt",
                dataset_masks_name = "dataset_masks.pt",
                overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
                )
        train_dataset = self.load_preprocess_data(train_args)
        inputs = ObjDict(
                train_dataset = train_dataset,
                )
        return inputs

    def load_preprocess_test_data(self,args):
        test_args = ObjDict(
                dataset_dir = args.preprocess_test_dir,
                dataset_name = "test_dataset.pt",
                mode = "test",
                input_ids_name = "input_ids.pt",
                attention_mask_name = "attention_mask.pt",
                labels_name = "labels.pt",
                dataset_masks_name = "dataset_masks.pt",
                overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
                )
        test_dataset = self.load_preprocess_data(test_args)
        inputs = ObjDict(
                test_dataset = test_dataset,
                )
        return inputs

    def load_preprocess_train_test_data(self,args):
        train_args = ObjDict(
                tokenzier = args.tokenizer,
                dataset_dir = args.preprocess_train_dir,
                dataset_name = "train_dataset.pt",
                mode = "train",
                input_ids_name = "input_ids.pt",
                attention_mask_name = "attention_mask.pt",
                labels_name = "labels.pt",
                dataset_masks_name = "dataset_masks.pt",
                overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
                )
        test_args = ObjDict(
                tokenzier = args.tokenizer,
                dataset_dir = args.preprocess_test_dir,
                dataset_name = "test_dataset.pt",
                mode = "test",
                input_ids_name = "input_ids.pt",
                attention_mask_name = "attention_mask.pt",
                labels_name = "labels.pt",
                dataset_masks_name = "dataset_masks.pt",
                overflow_to_sample_mapping_name = "overflow_to_sample_mapping.pt",
                )
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
            labels = torch.load(os.path.join(args.dataset_dir,args.labels_name))
            total_mask = torch.minimum(attention_mask,(dataset_mask==0).long())
            dataset = TensorDataset(input_ids,attention_mask,dataset_mask,total_mask,labels)

            if args.dataset_dir:
                mkdir_p(args.dataset_dir)
                torch.save(dataset,dataset_path)
        
        return dataset

    def create_preprocess_data(self,args):
        tokenizer = args.tokenizer
        df = pd.read_csv(args.input_csv_path)

        out_input_ids_path = os.path.join(args.out_dir,args.input_ids_name)
        out_attention_mask_path = os.path.join(args.out_dir,args.attention_mask_name)
        out_overflow_to_sample_mapping_path = os.path.join(args.out_dir,args.overflow_to_sample_mapping_name)
        out_labels_path = os.path.join(args.out_dir,args.labels_name)
        out_dataset_masks_path = os.path.join(args.out_dir,args.dataset_masks_name)
        
        out_file_paths = [out_input_ids_path,out_attention_mask_path,out_overflow_to_sample_mapping_path,out_labels_path]
        assert all([not os.path.exists(o) for o in out_file_paths])

        self.print_header()
        self.start_count_time("Tokenize")
        print("Tokenize text ")
        print("Dataframe shape: ",df.shape)
        self.print_header()
        tokenized_inputs = tokenizer(
                df['text'].tolist(),
                padding='max_length',
                max_length=512,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors='pt',
                )
        self.print_elapsed_time("Tokenize")
        self.print_header()
        
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

        self.print_header()
        print("Saving")
        self.print_header()
        if args.out_dir:
            mkdir_p(args.out_dir)
            torch.save(tokenized_inputs['input_ids'],out_input_ids_path)
            torch.save(tokenized_inputs['attention_mask'],out_attention_mask_path)
            torch.save(tokenized_inputs['overflow_to_sample_mapping'],out_overflow_to_sample_mapping_path)
            torch.save(tokenized_inputs['labels'],out_labels_path)
            torch.save(tokenized_inputs['dataset_masks'],out_dataset_masks_path)
        
        dataset = TensorDataset(tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['dataset_masks'],tokenized_inputs['labels'],)
        return dataset

    def randomize_train_data(self,inputs,cfg):
        self.print_message("[randomize_train_data]")
        assert cfg.randomize_cfg.fraction > 0. and cfg.randomize_cfg.fraction < 1.
        train_dataset = inputs.train_dataset
        mask = torch.rand(train_dataset.tensors[-1].shape) < cfg.randomize_cfg.fraction
        pos_label = train_dataset.tensors[-1].bool()
        train_dataset.tensors[0][pos_label*mask] = torch.randint(3,cfg.tokenizer.vocab_size,(torch.sum(mask*pos_label),))
        inputs.train_dataset = TensorDataset(*train_dataset.tensors)

    def include_external_dataset_as_label(self,dataset,cfg):
        self.print_message("[include_external_dataset_as_label]")
        return TensorDataset(*[t for t in dataset.tensors[:-1]]+[torch.maximum(dataset.tensors[-1],dataset.tensors[-3])])

    def mask_dataset_name(self,dataset,cfg):
        self.print_message("[mask_dataset_name]")
        ids = dataset.tensors[0]
        ids[dataset.tensors[-1].bool()] = -100
        inputs = [ids]+[t for t in dataset.tensors[1:]]
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
            dataloader = DataLoader(dataset, batch_size=args.batch_size)
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
        model = model.from_pretrained(args.pretrain_model).to(args.device)
        dataset_inputs = torch.load(args.dataset_tokens_path)
        test_sampler = RandomSampler(inputs.test_dataset)
        test_dataloader = DataLoader(inputs.test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        with torch.no_grad():
            dataset_emb = model(
                    input_ids=dataset_inputs['input_ids'].to(args.device),
                    attention_mask=dataset_inputs['attention_mask'].to(args.device),
                    output_hidden_states=True,
                    ).hidden_states[-1]
            seq_length = int(dataset_emb.size(1))

            min_context_sims = []
            
            for step, batch in enumerate(test_dataloader):
                if step % args.print_per_step == 0:
                    batch = tuple(t.to(args.device) for t in batch)
                    batch_test = {"input_ids": batch[0],"attention_mask": batch[1], "output_hidden_states": True,}
                    batch_model_out = model(**batch_test)
                    
                    _,pred_label_idx = torch.split((torch.argmax(batch_model_out.logits,axis=2)!=0).nonzero(),1,dim=1,)
                    pred_label_idx = torch.squeeze(pred_label_idx)

                    if pred_label_idx.numel():
                        pred_label = torch.squeeze(batch_test['input_ids'])[pred_label_idx]
                        min_context_sims.append(self.calculate_min_context_similarity(model,dataset_emb,pred_label,seq_length))
            self.print_header()
            print("minimum context similarity / number of prediction: {cos} / {npred:d}".format(
                cos=str(torch.mean(torch.stack(min_context_sims))),
                npred=len(min_context_sims),
                )
                )
            self.print_header()


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

