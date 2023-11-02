from .clustering_pipeline import ClusteringPipeline
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from .utils_preprocessing import load_traceset,traceset_to_textset,traceset_to_pattern_one_hot,load_spmf_files
from .attentionisallyouneed_v2.aiay_pre import load_dataset,TracesetDataset
from .attentionisallyouneed_v2.aiay_model import TransformerModel
from .attentionisallyouneed_v2.aiay_train import train,evaluate,get_embeddings
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np


import copy
import torch
import time
import math

class AIAYPlus(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.arch=kwargs.get("arch", "NoArch")
        self.freq=kwargs.get("freq", 0.1)
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        max_length=32
        traceset,len_sentences=load_dataset(self.dataset_name,max_length=32)
        # Example usage

        # Create the dataset with vocabulary building
        dataset = TracesetDataset(traceset)

        vocab=dataset.build_vocab()
        # print(vocab.get_stoi())

        batch_size = 32
        bptt=batch_size


        validation_split = .2
        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)

        dataset=TracesetDataset(dataset)
        # eval_dataset=TracesetDataset(eval_list)
        train_loader=DataLoader(dataset,batch_size=batch_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ntokens = len(vocab)  # size of vocabulary
        emsize = 16    # embedding dimension
        d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # number of heads in nn.MultiheadAttention
        dropout = 0.1  # dropout probability
        pad_idx=vocab.vocab.get_stoi()['<pad>']
        model = TransformerModel(pad_idx,ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
        model.load_state_dict(torch.load("./data/models/"+self.dataset_name+"_transformers.pt"))
        model.eval()

        e=get_embeddings(model, train_loader,bptt,device,len_sentences,ntokens,None)

        e=e.reshape(e.shape[0],-1)
        
        X,_=load_spmf_files(filepath=self.filepath,traceset=execution_traces_agilkia_format,dataset_name=self.dataset_name,freq=self.freq)
        X = StandardScaler().fit_transform(X)


        return np.concatenate((X,e),axis=1)

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        y=model.fit_predict(preprocessed_execution_traces)
        
        self.model=model
        return y





