import os
import pickle
import torch
import logging
import random
import numpy as np
import pandas as pd

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchmetrics.functional import accuracy,auroc,f1,precision,recall
from tqdm import tqdm, trange

from pipeline import Pipeline
from metrics.specificity import specificity
from utils.mkdir_p import mkdir_p
from utils.objdict import ObjDict

logger = logging.getLogger(__name__)
    
softmax = torch.nn.Softmax(dim=-1)

def make_label(input_ids,dataset_ids,length):
    start_index = 1
    dataset_length = len(dataset_ids)
    seq_length = len(input_ids)
    found = False
    while start_index + dataset_length < seq_length:
        if not all([input_ids[start_index+i] == dataset_ids[i] for i in range(dataset_length)]):
            start_index += 1
        else:
            found = True
            break
        
    output = [0 for _ in range(length)]
    if found:
        for i in range(dataset_length):
            output[start_index+i] = 1
    return output

class TokenClassifierPipeline(Pipeline):
 
    @classmethod
    def compute_metrics(cls,preds,labels):
        probs = softmax(preds.logits)[:,:,1].flatten()
        labels_flatten = labels.flatten()
        return {
            "accuracy": accuracy(probs,labels_flatten),
            #"auroc": auroc(probs,labels_flatten),
            "f1": f1(probs,labels_flatten),
            "precision": precision(probs,labels_flatten),
            "recall": recall(probs,labels_flatten),
            "specificity": specificity(probs,labels_flatten),
        }

    def preprocess_train_data(self,args):
        if args.load_preprocess:
            return self.load_preprocess_train_data(args)
        else:
            return self.create_preprocess_train_data(args)

    def load_preprocess_train_data(self,args):
        input_ids = torch.load(os.path.join(args.preprocess_train_dir,"input_ids.pt"))
        attention_mask = torch.load(os.path.join(args.preprocess_train_dir,"attention_mask.pt"))
        labels = torch.load(os.path.join(args.preprocess_train_dir,"labels.pt"))

        dataset = TensorDataset(input_ids,attention_mask,labels)
        train_size = int(args.train_size * len(dataset))
        val_size = int(args.val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
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

        print("Tokenize text")
        tokenized_inputs = tokenizer(df['text'].tolist(),pad_to_max_length=True,max_length=512,truncation=True,return_overflowing_tokens=True,return_tensors='pt',)

        print("Make labels")
        labels = []
        ntext = len(tokenized_inputs['overflow_to_sample_mapping'])
        for i in tqdm(range(ntext)):
            tokenized_dataset = tokenizer(df['dataset'][int(tokenized_inputs['overflow_to_sample_mapping'][i])])
            labels.append(make_label(tokenized_inputs.input_ids[i],tokenized_dataset['input_ids'][1:-1],len(tokenized_inputs.attention_mask[i]),))
        tokenized_inputs['labels'] = torch.tensor(labels)

        print("Saving")
        if args.preprocess_train_dir:
            mkdir_p(args.preprocess_train_dir)
            torch.save(tokenized_inputs['input_ids'],os.path.join(args.preprocess_train_dir,"input_ids.pt"))
            torch.save(tokenized_inputs['attention_mask'],os.path.join(args.preprocess_train_dir,"attention_mask.pt"))
            torch.save(tokenized_inputs['overflow_to_sample_mapping'],os.path.join(args.preprocess_train_dir,"overflow_to_sample_mapping.pt"))
            torch.save(tokenized_inputs['labels'],os.path.join(args.preprocess_train_dir,"labels.pt"))
        
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
        tokenized_inputs = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
        tokenized_inputs['id'] = df['id'].tolist()

        if args.preprocess_test_dir:
            mkdir_p(args.preprocess_test_dir)
            torch.save(tokenized_inputs['id'],os.path.join(args.preprocess_test_dir,"id.pt"))
            torch.save(tokenized_inputs['input_ids'],os.path.join(args.preprocess_test_dir,"input_ids.pt"))
            torch.save(tokenized_inputs['attention_mask'],os.path.join(args.preprocess_test_dir,"attention_mask.pt"))

        return tokenized_inputs
    
    def load_preprocess_test_data(self,args):
        ids = torch.load(os.path.join(args.preprocess_test_dir,"id.pt"))
        input_ids = torch.load(os.path.join(args.preprocess_test_dir,"input_ids.pt"))
        attention_mask = torch.load(os.path.join(args.preprocess_test_dir,"attention_mask.pt"))

        dataset = TensorDataset(input_ids,attention_mask)
        inputs = ObjDict(
                test_dataset = dataset,
                input_ids = input_ids,
                attention_mask = attention_mask,
                ids = ids,
                )
        return inputs
