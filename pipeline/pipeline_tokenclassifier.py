import os
import pickle
import numpy as np

from datasets import load_from_disk
from transformers import Trainer

from pipeline import Pipeline

class TokenClassifierPipeline(Pipeline):
    def set_cfg(self,cfg):
        self.cfg = cfg

    def read_np_dir(self):
        self.processed_dataset = load_from_disk(os.path.join(self.cfg.input_dataset_dir,self.cfg.input_dataset_name))
        self.dataset_dict = self.processed_dataset['train'].train_test_split(test_size=self.cfg.train_test_split)

    def train(self):
        trainer = Trainer(
            self.cfg.model,
            self.cfg.train_args,
            train_dataset=self.dataset_dict['train'],
            eval_dataset=self.dataset_dict['test'],
            data_collator=self.cfg.data_collator,
            tokenizer=self.cfg.tokenizer,
            compute_metrics=self.cfg.compute_metrics,
        )
        trainer.train()

    def save(self):
        self.cfg.model.save_pretrained(self.cfg.saved_model_path)

    def custom_train(self):
        pass
