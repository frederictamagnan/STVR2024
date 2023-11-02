from attentionisallyouneed_v2.aiay_pre import load_dataset,TracesetDataset
from attentionisallyouneed_v2.aiay_model import TransformerModel
from attentionisallyouneed_v2.aiay_train import train,evaluate,get_embeddings
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np


import copy
import torch
import time
import math

def inf(dataset_name):
    max_length=32
    traceset,len_sentences=load_dataset(dataset_name,max_length=32)
    # Example usage

    # Create the dataset with vocabulary building
    dataset = TracesetDataset(traceset)

    vocab=dataset.build_vocab()
    print(vocab.get_stoi())

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
    model.load_state_dict(torch.load("./data/models/"+dataset_name+"_transformers.pt"))
    model.eval()

    e=get_embeddings(model, train_loader,bptt,device,len_sentences,ntokens,None)
    return e.reshape(e.shape[0],-1)
if __name__=='__main__':
    # datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
    datasets=["scanette_100043-steps"]

    for dataset_name in datasets:
        inf(dataset_name)