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

def make_label(input_ids,dataset_ids,length,binary_label=True):
    start_index = 1
    dataset_length = len(dataset_ids)
    seq_length = len(input_ids)
    found_indices = []

    while start_index + dataset_length < seq_length:
        if not all([input_ids[start_index+i] == dataset_ids[i] for i in range(dataset_length)]):
            start_index += 1 
        else:
            found_indices.append(start_index)
            start_index += dataset_length

    output = [0 for _ in range(length)]
    if found_indices:
        if binary_label:
            for start_index in found_indices:
                for i in range(dataset_length):
                    output[start_index+i] = 1
        else:
            for start_index in found_indices:
                for i in range(dataset_length):
                    if i == 0:
                        output[start_index+i] = 1
                    elif i == dataset_length - 1:
                        output[start_index+i] = 3
                    else:
                        output[start_index+i] = 2

    return output

def make_labels(input_ids,dataset_ids,binary_label=True):
    start_index = 1
    max_dataset_length = max([len(d) for d in dataset_ids])
    dataset_length = {i:len(d) for i,d in enumerate(dataset_ids)}
    ndataset = len(dataset_ids)
    seq_length = len(input_ids)
    found_indices = []

    while start_index + max_dataset_length < seq_length:
        found = True
        max_length = -1
        for i in range(ndataset):
            if all([input_ids[start_index+j] == dataset_ids[i][j] for j in range(dataset_length[i])]):
                if dataset_length[i] > max_length:
                    max_length = dataset_length[i]
            else:
                found = False
                break
        if not found:
            start_index += 1 
        else:
            found_indices.append(start_index)
            start_index += max_length

    output = [0 for _ in range(length)]
    if found_indices:
        if binary_label:
            for start_index in found_indices:
                for i in range(dataset_length):
                    output[start_index+i] = 1
        else:
            for start_index in found_indices:
                for i in range(dataset_length):
                    if i == 0:
                        output[start_index+i] = 1
                    elif i == dataset_length - 1:
                        output[start_index+i] = 3
                    else:
                        output[start_index+i] = 2

    return output


class TokenMultiClassifierPipeline(Pipeline):

    @classmethod
    def compute_metrics(cls,preds,labels,num_classes,islogit=False,average='macro',):
        if islogit:
            probs = softmax(preds).flatten(0,1)
        else:
            probs = softmax(preds.logits).flatten(0,1)
        labels_flatten = labels.flatten(0,1)
        return {
            "accuracy": accuracy(probs,labels_flatten,num_classes=num_classes,average=average,),
            #"auroc": auroc(probs,labels_flatten),
            "f1": f1(probs,labels_flatten,num_classes=num_classes,average=average,),
            "precision": precision(probs,labels_flatten,num_classes=num_classes,average=average,),
            "recall": recall(probs,labels_flatten,num_classes=num_classes,average=average,),
            #"specificity": specificity(probs,labels,num_classes=num_classes),
        }

    def preprocess(self,args):
        if args.load_preprocess:
            return self.load_preprocess_train_data(args)
        else:
            return self.create_preprocess_train_data(args)

    def load_preprocess_train_data(self,args):
        train_dataset_path = os.path.join(args.preprocess_train_dir,"train_dataset.pt")
        val_dataset_path = os.path.join(args.preprocess_train_dir,"val_dataset.pt")
        test_dataset_path = os.path.join(args.preprocess_train_dir,"test_dataset.pt")

        train_dataset_exists = os.path.exists(train_dataset_path)
        val_dataset_exists = os.path.exists(val_dataset_path)
        test_dataset_exists = os.path.exists(test_dataset_path)

        if train_dataset_exists and val_dataset_exists and test_dataset_exists:
            self.print_message("[load_preprocess_train_data] all dataset exists. Loading presaved dataset.")
            train_dataset = torch.load(train_dataset_path)
            val_dataset = torch.load(val_dataset_path)
            test_dataset = torch.load(test_dataset_path)
            return ObjDict(
                    train_dataset = train_dataset,
                    val_dataset = val_dataset,
                    test_dataset = test_dataset,
                    )
        else:
            input_ids = torch.load(os.path.join(args.preprocess_train_dir,args.input_ids_name))
            attention_mask = torch.load(os.path.join(args.preprocess_train_dir,args.attention_mask_name))
            labels = torch.load(os.path.join(args.preprocess_train_dir,args.labels_name))

            dataset = TensorDataset(input_ids,attention_mask,labels)
            train_size = int(args.train_size * len(dataset))
            val_size = int(args.val_size * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

            if args.preprocess_train_dir:
                mkdir_p(args.preprocess_train_dir)
                torch.save(train_dataset,train_dataset_path)
                torch.save(val_dataset,val_dataset_path)
                torch.save(test_dataset,test_dataset_path)

            inputs = ObjDict(
                    dataset = dataset,
                    train_dataset = train_dataset,
                    val_dataset = val_dataset,
                    test_dataset = test_dataset,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels,
                    )
            return inputs

    def create_preprocess_train_data(self,args):
        tokenizer = args.tokenizer
        df = pd.read_csv(args.train_csv_path)

        out_input_ids_path = os.path.join(args.preprocess_train_dir,args.input_ids_name)
        out_attentation_mask_path = os.path.join(args.preprocess_train_dir,args.attentation_mask_name)
        out_overflow_to_sample_mapping_path = os.path.join(args.preprocess_train_dir,args.overflow_to_sample_mapping_name)
        out_labels_path = os.path.join(args.preprocess_train_dir,args.labels_name)
        
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
        ntext = len(tokenized_inputs['overflow_to_sample_mapping'])
        for i in tqdm(range(ntext)):
            datasets = df['dataset'][int(tokenized_inputs['overflow_to_sample_mapping'][i])]
            multidataset = "|" in datasets
            if multidataset:
                datasets = datasets.split("|")
            tokenized_dataset = tokenizer(datasets)
            if multidataset:
                labels.append(make_labels(tokenized_inputs.input_ids[i],[ts[1:-1] for ts in tokenized_dataset['input_ids']],len(tokenized_inputs.attention_mask[i]),))
            else:
                labels.append(make_label(tokenized_inputs.input_ids[i],tokenized_dataset['input_ids'][1:-1],len(tokenized_inputs.attention_mask[i]),))
        tokenized_inputs['labels'] = torch.tensor(labels)

        self.print_header()
        print("Saving")
        self.print_header()
        if args.preprocess_train_dir:
            mkdir_p(args.preprocess_train_dir)
            torch.save(tokenized_inputs['input_ids'],out_input_ids_path)
            torch.save(tokenized_inputs['attention_mask'],out_attention_mask_path)
            torch.save(tokenized_inputs['overflow_to_sample_mapping'],out_overflow_to_sample_mapping_path)
            torch.save(tokenized_inputs['labels'],out_labels_path)
        
        dataset = TensorDataset(tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['labels'])
        train_size = int(args.train_size * len(dataset))
        val_size = int(args.val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        inputs = ObjDict(
                dataset = dataset,
                train_dataset = train_dataset,
                val_dataset = val_dataset,
                test_dataset = test_dataset,
                )
        return inputs

    def create_preprocess_test_data(self,args):
        tokenizer = args.tokenizer
        df = pd.read_csv(args.test_csv_path)
        self.print_header()
        self.start_count_time("Tokenize")
        print("Tokenize text ")
        print("Dataframe shape: ",df.shape)
        tokenized_inputs = tokenizer(
                df['text'].tolist(),
                padding='max_length',
                max_length=512,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt"
                )
        tokenized_inputs['id'] = df['id'].tolist()
        self.print_elapsed_time("Tokenize")
        self.print_header()

        if 'dataset' in df:
            print("Make labels")
            self.print_header()
            labels = []
            ntext = len(tokenized_inputs['overflow_to_sample_mapping'])
            for i in tqdm(range(ntext)):
                tokenized_dataset = tokenizer(df['dataset'][int(tokenized_inputs['overflow_to_sample_mapping'][i])])
                labels.append(make_label(tokenized_inputs.input_ids[i],tokenized_dataset['input_ids'][1:-1],len(tokenized_inputs.attention_mask[i]),))
            tokenized_inputs['labels'] = torch.tensor(labels)

            self.print_header()
            print("Saving")
            self.print_header()

        if args.preprocess_test_dir:
            mkdir_p(args.preprocess_test_dir)
            torch.save(tokenized_inputs['id'],os.path.join(args.preprocess_test_dir,"id.pt"))
            torch.save(tokenized_inputs['input_ids'],os.path.join(args.preprocess_test_dir,"input_ids.pt"))
            torch.save(tokenized_inputs['attention_mask'],os.path.join(args.preprocess_test_dir,"attention_mask.pt"))
            torch.save(tokenized_inputs['overflow_to_sample_mapping'],os.path.join(args.preprocess_test_dir,"overflow_to_sample_mapping.pt"))
            if 'dataset' in df: 
                torch.save(tokenized_inputs['labels'],os.path.join(args.preprocess_test_dir,"labels.pt"))

        return tokenized_inputs
    
    def load_preprocess_test_data(self,args):
        out_input_ids_path = os.path.join(args.preprocess_test_dir,args.input_ids_name)
        out_attention_mask_path = os.path.join(args.preprocess_test_dir,args.attention_mask_name)
        #out_overflow_to_sample_mapping_path = os.path.join(args.preprocess_test_dir,args.overflow_to_sample_mapping_name)
        
        out_file_paths = [out_input_ids_path,out_attention_mask_path,]
        assert all([os.path.exists(o) for o in out_file_paths])

        self.print_message("[load_preprocess_test_data] all dataset exists. Loading presaved dataset.")
        #ids = torch.load(os.path.join(args.preprocess_test_dir,"id.pt"))
        input_ids = torch.load(os.path.join(args.preprocess_test_dir,"input_ids.pt"))
        attention_mask = torch.load(os.path.join(args.preprocess_test_dir,"attention_mask.pt"))
        overflow_to_sample_mapping = torch.load(os.path.join(args.preprocess_test_dir,"overflow_to_sample_mapping.pt"))
        if os.path.exists(os.path.join(args.preprocess_test_dir,"labels.pt")):
            labels = torch.load(os.path.join(args.preprocess_test_dir,"labels.pt"))
        else:
            labels = None

        dataset = TensorDataset(input_ids,attention_mask,overflow_to_sample_mapping)
        inputs = ObjDict(
                test_dataset = dataset,
                input_ids = input_ids,
                attention_mask = attention_mask,
                overflow_to_sample_mapping = overflow_to_sample_mapping,
                labels = labels,
                )
        return inputs

    def randomize_train_data(self,inputs,cfg):
        self.print_message("[randomize_train_data]")
        assert cfg.randomize_cfg.fraction > 0. and cfg.randomize_cfg.fraction < 1.
        train_dataset = inputs.train_dataset.dataset[inputs.train_dataset.indices]
        mask = torch.rand(train_dataset[0].shape) < cfg.randomize_cfg.fraction
        pos_label = train_dataset[2].bool()
        train_dataset[0][pos_label*mask] = torch.randint(3,cfg.tokenizer.vocab_size,(torch.sum(mask*pos_label),))
        inputs.train_dataset = TensorDataset(*train_dataset)

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

