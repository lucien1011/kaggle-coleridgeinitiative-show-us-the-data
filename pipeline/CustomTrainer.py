import os
import copy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader,TensorDataset

from nlptools.Trainer import Trainer
from nlptools.general_utils.mkdir_p import mkdir_p

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
        return {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}

    def semisupervise_train(self,inputs,model,args):
        mkdir_p(args.output_dir)

        self.logger = self.make_logger(
                self.name,self.log_level,
                os.path.join(args.output_dir,self.name".log"),
                console=True,
                )

        self.print_header_message("Write to dir: "args.output_dir)

        for i in range(args.niter):
            self.logger.info("Iteration {it:d}".format(it=i))
            self.logger.info("Train")
            train_outputs = self.train(inputs,model,args)
            self.logger.info("Label propagation")
            pred_labels = self.label_propagation(inputs,model,args.threshold,args.val_batch_size,args.device)
            inputs.train_dataset = self.merge_labels(inputs.train_dataset,pred_labels)

    @classmethod
    def merge_labels(cls,dataset,pred_labels):
        return TensorDataset(*[t for t in dataset.tensors[:-1]][torch.maximum(dataset.tensors[-1],pred_labels)])

    def label_propagation(self,inputs,model,threshold,batch_size,device):
        model.eval()
        dataloader = DataLoader(inputs.train_dataset,batch_size)
        output_labels = []
        for step,batch in enumerate(tqdm(dataloader)):
            batch_val = {k:v.to(device) for k,v in self.patch_train_batch(batch).items()}
            with torch.no_grad():
                outputs = model(**batch_val)
                pred_labels = (torch.nn.functional.softmax(outputs.logits,dim=2)[:,:,1] > threshold).long()
            output_labels.append(pred_labels.to('cpu'))
        return torch.cat(output_labels,axis=0)

