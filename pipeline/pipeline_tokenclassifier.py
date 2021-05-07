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
from tqdm import tqdm, trange

from pipeline import Pipeline
from utils.objdict import ObjDict

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def make_label(input_ids,dataset_ids,length):
    start_index = 1
    dataset_length = len(dataset_ids)-2
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

    def preprocess(self,args):
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
        df = pd.read_csv(args.input_csv_path)
        tokenized_inputs = tokenizer(df['sentence'].tolist(), padding=True, truncation=True, return_tensors="pt")
        labels = []
        for i,dataset in enumerate(df['dataset']):
            tokenized_dataset = tokenizer(dataset)
            
            labels.append(make_label(tokenized_inputs.input_ids[i],tokenized_dataset['input_ids'][1:-1],len(tokenized_inputs.attention_mask[i]),))
    
        tokenized_inputs['labels'] = torch.tensor(labels)

        dataset = TensorDataset(tokenized_inputs['input_ids'],tokenized_inputs['attention_mask'],tokenized_inputs['labels'])
        train_size = int(args.train_size * len(dataset))
        val_size = int(args.val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        
        inputs = ObjDict(
                train_dataset = train_dataset,
                val_dataset = val_dataset,
                test_dataset = test_dataset,
                )
        return inputs

    def train(self,inputs,model,args):
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
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
        set_seed(args)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration",)
            tr_loss_per_epoch = 0.0
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                batch_train = {"input_ids": batch[0],"attention_mask": batch[1],"labels": batch[2]}

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
                        batch_val = tuple(t.to(args.device) for t in iter(val_dataloader).next())
                        batch_val = {"input_ids": batch_val[0],"attention_mask": batch_val[1],"labels": batch_val[2]}
                        val_loss = model(**batch_val)[0]
                        tqdm.write("| train loss {train_loss:4.2f} |  val loss {val_loss:4.2f} |".format(train_loss=loss.item(),val_loss=val_loss))

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def save(self):
        pass
        #self.cfg.model.save_pretrained(self.cfg.saved_model_path)
        #torch.save(self.cfg.model,self.cfg.saved_model_path)

    def custom_train(self):
        pass
