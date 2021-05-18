import os
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
cos = CosineSimilarity(dim=2,eps=1e-6)

def make_label(input_ids,dataset_ids,length):
    start_index = 1
    dataset_length = len(dataset_ids)
    seq_length = len(input_ids)
    found_indices = []

    while start_index + dataset_length < seq_length:
        if not all([input_ids[start_index+i] == dataset_ids[i] for i in range(dataset_length)]):
            start_index += 1 
        else:
            found_indices.append(start_index)
            found = True
            start_index += dataset_length

    output = [0 for _ in range(length)]
    if found_indices:
        for start_index in found_indices:
            for i in range(dataset_length):
                output[start_index+i] = 1
                #if i == 0:
                #    output[start_index+i] = 1
                #elif i == dataset_length - 1:
                #    output[start_index+i] = 3
                #else:
                #    output[start_index+i] = 2

    return output

class TokenMultiClassifierPipeline(Pipeline):

    @classmethod
    def compute_metrics(cls,preds,labels,num_classes):
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

    def preprocess(self,args):
        if args.load_preprocess:
            return self.load_preprocess_train_data(args)
        else:
            return self.create_preprocess_train_data(args)

    def load_preprocess_train_data(self,args):
        input_ids = torch.load(os.path.join(args.preprocess_train_dir,args.input_ids_name))
        attention_mask = torch.load(os.path.join(args.preprocess_train_dir,args.attention_mask_name))
        labels = torch.load(os.path.join(args.preprocess_train_dir,args.labels_name))

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
            tokenized_dataset = tokenizer(df['dataset'][int(tokenized_inputs['overflow_to_sample_mapping'][i])])
            labels.append(make_label(tokenized_inputs.input_ids[i],tokenized_dataset['input_ids'][1:-1],len(tokenized_inputs.attention_mask[i]),))
        tokenized_inputs['labels'] = torch.tensor(labels)

        self.print_header()
        print("Saving")
        self.print_header()
        if args.preprocess_train_dir:
            mkdir_p(args.preprocess_train_dir)
            torch.save(tokenized_inputs['input_ids'],os.path.join(args.preprocess_train_dir,args.input_ids_name))
            torch.save(tokenized_inputs['attention_mask'],os.path.join(args.preprocess_train_dir,args.attention_mask_name))
            torch.save(tokenized_inputs['overflow_to_sample_mapping'],os.path.join(args.preprocess_train_dir,args.overflow_to_sample_mapping_name))
            torch.save(tokenized_inputs['labels'],os.path.join(args.preprocess_train_dir,args.labels_name))
        
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
        tokenized_inputs = tokenizer(
                df['text'].tolist(),
                padding='max_length',
                max_length=512,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt"
                )
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

    def evaluate(self,inputs,model,args):
        from torch.utils.data import DataLoader, RandomSampler 
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
                        min_context_sims.append(self.calculate_min_context_similarity(model,dataset_emb,pred_label))
            self.print_header()
            print("minimum context similarity / number of prediction: {cos} / {npred:d}".format(
                cos=str(torch.mean(torch.stack(min_context_sims))),
                npred=len(min_context_sims),
                )
                )
            self.print_header()


    @classmethod
    def calculate_min_context_similarity(cls,model,dataset_emb,pred_label):
        """
        dataset_emb: [ number of dataset names , sequence_length , embedding_dimension ]
        pred_label: [ number of predicted tokens ]
        model: embedding model
        """
        pred_length = len(pred_label)
        pred_length = pred_label.numel()
        seq_length = int(dataset_emb.size(1))
        pred_label = torch.nn.functional.pad(pred_label,pad=(0,seq_length-pred_length))
        pred_emb = model(input_ids=torch.unsqueeze(pred_label,axis=0),output_hidden_states=True).hidden_states[-1]
        sim_matrix = cos(dataset_emb,pred_emb)
        min_context_sim = torch.min(torch.sqrt(torch.sum(sim_matrix**2,axis=1)))
        return min_context_sim

