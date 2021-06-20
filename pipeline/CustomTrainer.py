import os
import copy
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader,TensorDataset
from torchmetrics.functional import accuracy,auroc,f1,precision,recall

from nlptools.Trainer import Trainer
from nlptools.general_utils.mkdir_p import mkdir_p
from postprocessing.tools import split_list,calculate_precision_from_pred_batch
from metrics.kaggle import jaccard_similarity

rcdatasets = list(pickle.load(open("storage/input/rcdataset/data_sets_unique_map.p","rb")).keys())

selected_keywords = [
        "study",
        "Study",
        "studies",
        "Studies",
        "statistics",
        "Statistics",
        "project",
        "Project",
        "projects",
        "Projects",
        "program",
        "Program",
        "programs",
        "Programs",
        "National",
        "national",
        "Cohort",
        "cohort",
        "Report",
        "Reports",
        "report",
        "reports",
        "Data",
        "data",
        ]

class CustomTrainer(Trainer):
    @classmethod
    def patch_train_batch(cls,batch):
        return {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "sample_weight": batch[3],
                "labels": batch[-1],
                }

    @classmethod
    def patch_val_batch(cls,batch):
        return {"input_ids": batch[0],"attention_mask": batch[1],"labels": torch.maximum(batch[2],batch[-1]),}

    def semisupervise_train(self,inputs,model,args):
        mkdir_p(args.output_dir)

        self.logger = self.make_logger(
                "semisupervise_train",self.log_level,
                os.path.join(args.output_dir,self.name+".log"),
                console=True,
                )

        self.print_header_message(
                "\n".join([
                    "Write to dir: "+args.output_dir,
                    "Number of iteration: {:d}".format(args.niter),
                    ])
                    )
        
        for i in range(args.niter):
            self.logger.info("Iteration {it:d}".format(it=i))
            self.logger.info("Train")
            itername = "iteration{:d}".format(i)
            train_args = copy.deepcopy(args)
            train_args.output_dir = os.path.join(args.output_dir,itername)
            train_outputs = self.train(inputs,model,train_args,logname=itername)
            self.logger.info("Label propagation")
            train_pred_labels = self.label_propagation(inputs.train_dataset,model,args.thresholds[i],args.val_batch_size,args.device,matched=args.matched_datasets,tokenizer=args.tokenizer,)
            val_pred_labels = self.label_propagation(inputs.val_dataset,model,args.thresholds[i],args.val_batch_size,args.device,matched=args.matched_datasets,tokenizer=args.tokenizer,)
            inputs.train_dataset = self.merge_labels(inputs.train_dataset,train_pred_labels)
            inputs.val_dataset = self.merge_labels(inputs.val_dataset,val_pred_labels)

    @classmethod
    def match_to_datasets(cls,tokenizer,pred_labels,input_ids,keywords=selected_keywords):
        nbatch = len(pred_labels)
        matched_labels = []
        for ind in range(nbatch):
            split_ids,split_indices = split_list(input_ids[ind]*pred_labels[ind],return_indices=True)
            matched_indices = []
            pred_tokens = tokenizer.batch_decode(split_ids,skip_special_tokens=True)
            for i,ps in enumerate(pred_tokens):
                if any([k in pred_tokens for k in keywords]):
                    matched_indices.extend(split_indices[i])
                #for ts in datasets:
                #    if jaccard_similarity(ps,ts) > 0.5:
                #        matched_indices.extend(split_indices[i])
            tmp = copy.deepcopy(pred_labels[ind])
            for match_idx in matched_indices:
                tmp[match_idx] = 1
            matched_labels.append(tmp.unsqueeze(axis=0))
        return torch.cat(matched_labels,axis=0)

    @classmethod
    def merge_labels(cls,dataset,pred_labels):
        return TensorDataset(*[t for t in dataset.tensors[:-1]]+[torch.maximum(dataset.tensors[-1],pred_labels)])

    def label_propagation(self,dataset,model,threshold,batch_size,device,matched=False,tokenizer=None):
        model.eval()
        dataloader = DataLoader(dataset,batch_size)
        output_labels = []
        for step,batch in enumerate(tqdm(dataloader)):
            batch_val = {k:v.to(device) for k,v in self.patch_train_batch(batch).items()}
            with torch.no_grad():
                outputs = model(**batch_val)
                pred_labels = (torch.nn.functional.softmax(outputs.logits,dim=2)[:,:,1] > threshold).long()
                if matched:
                    pred_labels = self.match_to_datasets(tokenizer,pred_labels,batch_val['input_ids'])
            output_labels.append(pred_labels.to('cpu'))
        return torch.cat(output_labels,axis=0)
    
    def compute_metrics(self,outputs,batch,args,threshold=0.5,num_classes=2,average=None,):
        pred_labels = (torch.nn.functional.softmax(outputs.logits,dim=2)[:,:,1] > threshold).long()

        pred_labels_flatten = pred_labels.flatten(0,1)
        labels_flatten = batch["labels"].flatten(0,1)

        tp,fp = calculate_precision_from_pred_batch(batch['input_ids'],batch['attention_mask'],pred_labels,args.tokenizer,rcdatasets)
        
        return {
            "accuracy": accuracy(pred_labels_flatten,labels_flatten,num_classes=num_classes,average='micro',).float(),
            "f1": f1(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1].float(),
            "precision": precision(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1].float(),
            "recall": recall(pred_labels_flatten,labels_flatten,num_classes=num_classes,average=average,)[-1].float(),
            "ntf": labels_flatten.sum().float(),
            "rcdataset_tp": tp,
            "rcdataset_fp": fp,
            "rcdataset_precision": tp / (tp + fp),
        }
