import os
import pickle
import numpy as np

from pipeline import Pipeline

class ClassifierPipeline(Pipeline):
    def set_cfg(self,cfg):
        self.cfg = cfg

    def read_np_dir(self):
        self.x_train = np.load(os.path.join(self.cfg.input_np_dir,"x_train.npy"),allow_pickle=True,)
        self.y_train = np.load(os.path.join(self.cfg.input_np_dir,"y_train.npy"))
        self.sample_weight_train = np.load(os.path.join(self.cfg.input_np_dir,"sample_weight_train.npy"))

        self.x_val = np.load(os.path.join(self.cfg.input_np_dir,"x_val.npy"),allow_pickle=True,)
        self.y_val = np.load(os.path.join(self.cfg.input_np_dir,"y_val.npy"))
        self.sample_weight_val = np.load(os.path.join(self.cfg.input_np_dir,"sample_weight_val.npy")) 

        self.x_test = np.load(os.path.join(self.cfg.input_np_dir,"x_test.npy"),allow_pickle=True,)
        self.y_test = np.load(os.path.join(self.cfg.input_np_dir,"y_test.npy"))
        self.sample_weight_test = np.load(os.path.join(self.cfg.input_np_dir,"sample_weight_test.npy"))

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
                #batch_size=512,
                )

    def save(self):
        self.cfg.model.save(self.cfg.saved_model_path)
        pickle.dump(self.cfg.history,open(self.cfg.saved_history_path,"wb"))

    def custom_train(self):
        pass
        #for epoch in range(0):
        #    
        #    print("Epoch ",epoch)
        #   
        #    idx_batch = np.random.randint(0, x_train.shape[0], 512)
        #
        #    with tf.GradientTape() as tape:
        #        y_train_pred = model(x_train[idx_batch])
        #        train_loss = loss(y_train[idx_batch],y_train_pred)
        #    grads = tape.gradient(train_loss,model.trainable_variables)
        #    optimizer.apply_gradients(zip(grads,model.trainable_variables))
        #    y_val_pred = model(x_val)
        #    val_loss = loss(y_val,y_val_pred)
        #    
        #    print("Loss: ",train_loss,val_loss)
        #
        #    m_train.update_state(y_train[idx_batch],y_train_pred)
        #    m_val.update_state(y_val,y_val_pred)
        #    print("AUC: ",m_train.result().numpy(),m_val.result().numpy())
