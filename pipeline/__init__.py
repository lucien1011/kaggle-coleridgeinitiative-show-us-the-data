import os
import torch
import logging
import random
import time
import numpy as np
import pickle

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from utils.mkdir_p import mkdir_p

logger = logging.getLogger(__name__)

class Pipeline(object):
    def __init__(self):
        self.start_times = {} 

    def predict(self):
        pass

    def start_count_time(self,key):
        if key not in self.start_times:
            self.start_times[key] = time.time()

    def print_elapsed_time(self,key):
        if key in self.start_times:
            elapsed_time = time.time() - self.start_times[key]
            print("Time used: {time:4.2f} seconds".format(time=elapsed_time))

    @classmethod
    def save_if_not_exist(cls,dataset,path):
        if not os.path.exists(path):
            torch.save(dataset,path)

    @classmethod
    def compute_metrics(cls,preds,labels):
        return {}

    @classmethod
    def print_header(cls,header_length=50):
        print("*"*header_length)

    @classmethod
    def set_seed(cls,args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    @classmethod
    def print_message(cls,m):
        cls.print_header()
        print(m)
        cls.print_header()

    @classmethod
    def get_model_checkpts(cls,input_dir,key):
        return [c for c in os.listdir(input_dir) if key in c and os.path.isdir(os.path.join(input_dir,c))]

    @classmethod
    def patch_train_batch(cls,batch):
        return {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}

    @classmethod
    def patch_test_batch(cls,batch):
        return {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}

    @classmethod
    def custom_print_in_validation(cls,batch,preds,labels):
        pass

    def tokenize_df(self,tokenizer,texts,tokenize_args):
        self.print_header()
        self.start_count_time("Tokenize")
        print("Tokenize text ")
        print("Dataframe shape: ",len(texts))
        self.print_header()
        tokenized_inputs = tokenizer(texts,**tokenize_args)
        self.print_elapsed_time("Tokenize")
        self.print_header()
        return tokenized_inputs

    def train(self,inputs,model,args):
        model.to(args.device)
        if not args.train_batch_size:
            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(inputs.train_dataset)
        val_sampler = RandomSampler(inputs.val_dataset)
        train_dataloader = DataLoader(inputs.train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        val_dataloader = DataLoader(inputs.val_dataset, sampler=val_sampler, batch_size=args.val_batch_size)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay},
                    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
                                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate, 
                betas=args.betas,
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay,
                )
        num_warmup_steps = args.warmup_steps if args.warmup_steps >= 1 else int(t_total*args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(inputs.train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch",)
        self.set_seed(args)
        training_log = {}
        mkdir_p(args.output_dir)
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration",)
            tr_loss_per_epoch = 0.0
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                batch_train = self.patch_train_batch(batch)

                outputs = model(**batch_train)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                tr_loss_per_epoch += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        model.eval()
                        batch_val = tuple(t.to(args.device) for t in iter(val_dataloader).next())
                        batch_val = self.patch_test_batch(batch_val)

                        with torch.no_grad():
                            preds_val = model(**batch_val)
                            val_loss = preds_val[0]
                            preds_train = model(**batch_train)
                            metrics_train = self.compute_metrics(preds_train,batch_train['labels'],num_classes=model.num_labels)
                            metrics_val = self.compute_metrics(preds_val,batch_val['labels'],num_classes=model.num_labels)

                        self.custom_print_in_validation(batch_val,preds_val,batch_val['labels'])

                        tqdm.write("*"*100)
                        tqdm.write("global step {global_step}".format(global_step=global_step))
                        tqdm.write("*"*100)
                        tqdm.write(" | ".join([
                            "train loss {train_loss:4.4f}".format(train_loss=loss.item()),
                            ] + ["train {metric} {value:4.4f}".format(metric=name,value=value) for name,value in metrics_train.items()]
                            ))
                        tqdm.write(" | ".join([
                            "val loss {val_loss:4.4f}".format(val_loss=val_loss),
                            ] + ["val {metric} {value:4.4f}".format(metric=name,value=value) for name,value in metrics_val.items()]
                            ))
                        training_log[global_step] = {"train_loss": loss.item(), "train_metrics": metrics_train, "val_loss": val_loss, "val_metrics": metrics_val,}
                        pickle.dump(training_log,open(os.path.join(args.output_dir,"training_log.p"),"wb"))

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, "checkpoint-step-{}".format(global_step))
                        self.save_model(model,args,output_dir) 
                        logger.info("Saving model checkpoint (per training step) to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            
            output_dir = os.path.join(args.output_dir,"checkpoint-epoch-{}".format(epoch))
            self.save_model(model,args,output_dir) 
            logger.info("Saving model checkpoint (per epoch) to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    @classmethod
    def save_model(cls,model,args,output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
