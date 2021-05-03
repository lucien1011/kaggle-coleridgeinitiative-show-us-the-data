import os
import pickle
import numpy as np

from pipeline import Pipeline

class ClassifierPipeline(Pipeline):
    def set_cfg(self,cfg):
        self.cfg = cfg

    def read_np_dir(self):
        self.x_train = np.load(self.cfg.x_train_path,allow_pickle=True,)
        self.y_train = np.load(self.cfg.y_train_path)
        self.sample_weight_train = np.load(self.cfg.sample_weight_train_path)

        self.x_val = np.load(self.cfg.x_val_path,allow_pickle=True,)
        self.y_val = np.load(self.cfg.y_val_path)
        self.sample_weight_val = np.load(self.cfg.sample_weight_val_path) 

        self.x_test = np.load(self.cfg.x_test_path,allow_pickle=True,)
        self.y_test = np.load(self.cfg.y_test_path)
        self.sample_weight_test = np.load(self.cfg.sample_weight_test_path)

    def train(self):
        self.cfg.model.compile(
                optimizer=self.cfg.optimizer,
                loss=self.cfg.loss,
                metrics=self.cfg.metrics,
                )
        self.cfg.history = self.cfg.model.fit(
                x=self.x_train,
                y=self.y_train,
                sample_weight=self.sample_weight_train,
                validation_data=(self.x_val,self.y_val),
                epochs=self.cfg.epochs,
                callbacks=self.cfg.callbacks,
                batch_size=self.cfg.batch_size,
                )

    def save(self):
        if not self.cfg.checkpoint:
            self.cfg.model.save(self.cfg.saved_model_path)
        pickle.dump(self.cfg.history.history,open(self.cfg.saved_history_path,"wb"))

    def custom_train(self):
        pass
