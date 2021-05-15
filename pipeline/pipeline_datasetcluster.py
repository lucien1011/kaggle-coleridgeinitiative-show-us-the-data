import os
import torch
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline import Pipeline
from utils.mkdir_p import mkdir_p
from utils.objdict import ObjDict

cos = torch.nn.CosineSimilarity(dim=2,eps=1e-6)

class DatasetClusterPipeline(Pipeline):

    def load_preprocess_train_data(self,args):

        similarity = np.load(os.path.join(args.preprocess_train_dir,"similarity.npy"))
        embedding = np.load(os.path.join(args.preprocess_train_dir,"embedding.npy"))
        input_ids = np.load(os.path.join(args.preprocess_train_dir,"input_ids.npy"))
        attention_mask = np.load(os.path.join(args.preprocess_train_dir,"attention_mask.npy"))

        inputs = ObjDict(
                similarity = similarity,
                embedding = embedding,
                input_ids = input_ids,
                attention_mask = attention_mask,
                )
        return inputs

    def create_preprocess_train_data(self,args):

        tokenizer = args.tokenizer
        df = pd.read_csv(args.train_csv_path)

        self.print_header()
        print("Tokenize text ")
        print("Dataframe shape: ",df.shape)
        self.print_header()

        tokenized_inputs = tokenizer(df['dataset'].unique().tolist(),padding=True,truncation=True,return_tensors='pt',)
        with torch.no_grad():
            embedding = args.embedding_model(**tokenized_inputs)

        emb_ref = torch.broadcast_to(torch.unsqueeze(embedding.last_hidden_state[0,:,:],axis=0),embedding.last_hidden_state.shape)
        similarity = cos(emb_ref,embedding.last_hidden_state)

        self.print_header()
        print("Saving")
        self.print_header()
        if args.preprocess_train_dir:
            mkdir_p(args.preprocess_train_dir)
            np.save(open(os.path.join(args.preprocess_train_dir,"similarity.npy"),"wb"),similarity.detach().numpy(),)
            np.save(open(os.path.join(args.preprocess_train_dir,"embedding.npy"),"wb"),embedding.last_hidden_state.detach().numpy(),)
            np.save(open(os.path.join(args.preprocess_train_dir,"input_ids.npy"),"wb"),tokenized_inputs['input_ids'],)
            np.save(open(os.path.join(args.preprocess_train_dir,"attention_mask.npy"),"wb"),tokenized_inputs['attention_mask'],)

    def train(self,inputs,model,args):
        
        #x = np.reshape(inputs.embedding,(inputs.embedding.shape[0],inputs.embedding.shape[1]*inputs.embedding.shape[2]))
        x = inputs.similarity
        variances = []
        silhouettes = []
        outputs = []
        Ks = range(2,args.nKs+1)
        for k in Ks:
            self.print_header()
            print("Fitting ",k)
            result = KMeans(n_clusters=k,random_state=args.random_state,max_iter=args.max_iter).fit(x)
            variances.append(result.inertia_)
            silhouettes.append(metrics.silhouette_score(x,result.labels_,metric='euclidean'))

        fig, ax = plt.subplots(2,1,figsize=(15, 8))
        ax[0].plot(Ks,variances)
        ax[0].set_ylabel("Inertia ( Total Distance )")
        ax[0].set_xlabel("K Value")
        ax[0].set_xticks(Ks)
        mkdir_p(args.output_dir)

        ax[1].plot(Ks,silhouettes)
        ax[1].set_ylabel("Silhouette score")
        ax[1].set_xlabel("K Value")
        ax[1].set_xticks(Ks)
        
        mkdir_p(args.output_dir)
        fig.savefig(os.path.join(args.output_dir,"inertia_silhouette_vs_k.png"))
