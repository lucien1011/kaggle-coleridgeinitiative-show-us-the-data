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

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def make_label(input_ids,dataset_ids,length):
    start_index = 1
    dataset_length = len(dataset_ids)
    seq_length = len(input_ids)
    found_indices = []

    while start_index + dataset_length < seq_length:
        if not all([input_ids[start_index+i] == dataset_ids[i] for i in range(dataset_length)]):
            found_indices.append(start_index)
            start_index += dataset_length
        else:
            found = True
            break

    output = [0 for _ in range(length)]
    if found_indices:
        for start_index in found_indices:
            for i in range(dataset_length):
                if i == 0:
                    output[start_index+i] = 1
                elif i == dataset_length - 1:
                    output[start_index+i] = 3
                else:
                    output[start_index+i] = 2

    return output

def compute_metrics(preds,labels,num_classes):
    probs = softmax(preds.logits).flatten(0,1)
    labels_flatten = labels.flatten(0,1)
    return {
        "accuracy": accuracy(probs,labels_flatten,num_classes=num_classes,average='macro',),
        #"auroc": auroc(probs,labels_flatten),
        "f1": f1(probs,labels_flatten,num_classes=num_classes,average='macro',),
        "precision": precision(probs,labels_flatten,num_classes=num_classes,average='macro',),
        "recall": recall(probs,labels_flatten,num_classes=num_classes,average='macro',),
        #"specificity": specificity(probs,labels,num_classes=num_classes),
    }

class TokenMultiClassifierPipeline(Pipeline):

    def preprocess(self,args):
        if args.load_preprocess:
            return self.load_preprocess_train_data(args)
        else:
            return self.create_preprocess_train_data(args)

    def load_preprocess_train_data(self,args):
        input_ids = torch.load(os.path.join(args.preprocess_train_dir,"input_ids.pt"))
        attention_mask = torch.load(os.path.join(args.preprocess_train_dir,"attention_mask.pt"))
        labels = torch.load(os.path.join(args.preprocess_train_dir,"multilabels.pt"))

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

        self.print_header()
        print("Tokenize text ")
        print("Dataframe shape: ",df.shape)
        self.print_header()
        tokenized_inputs = tokenizer(df['text'].tolist(),padding='max_length',max_length=512,truncation=True,return_overflowing_tokens=True,return_tensors='pt',)

        self.print_header()
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
        tokenized_inputs = tokenizer(df['text'].tolist(),padding='max_length',max_length=512,truncation=True,return_overflowing_tokens=True,return_tensors="pt")
        tokenized_inputs['id'] = df['id'].tolist()

        if args.preprocess_test_dir:
            mkdir_p(args.preprocess_test_dir)
            torch.save(tokenized_inputs['id'],os.path.join(args.preprocess_test_dir,"id.pt"))
            torch.save(tokenized_inputs['input_ids'],os.path.join(args.preprocess_test_dir,"input_ids.pt"))
            torch.save(tokenized_inputs['attention_mask'],os.path.join(args.preprocess_test_dir,"attention_mask.pt"))

        return tokenized_inputs
    
    def load_preprocess_test_data(self,args):
        #ids = torch.load(os.path.join(args.preprocess_test_dir,"id.pt"))
        input_ids = torch.load(os.path.join(args.preprocess_test_dir,"input_ids.pt"))
        attention_mask = torch.load(os.path.join(args.preprocess_test_dir,"attention_mask.pt"))

        dataset = TensorDataset(input_ids,attention_mask)
        inputs = ObjDict(
                test_dataset = dataset,
                input_ids = input_ids,
                attention_mask = attention_mask,
                #ids = ids,
                )
        return inputs
