from attentionisallyouneed_v2.aiay_pre import load_dataset,TracesetDataset
from attentionisallyouneed_v2.aiay_model import TransformerModel
from attentionisallyouneed_v2.aiay_train import train,evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np


import copy
import torch
import time
import math

def train_aiyn(dataset_name):
    max_length=32
    traceset,len_sentences=load_dataset(dataset_name,max_length=32)
    # Example usage
    train_list, eval_list = train_test_split(traceset, test_size=0.2, random_state=42)
    print(type(traceset))
    print(type(train_list))
    print(traceset[0])
    print(train_list[0])
    # Create the dataset with vocabulary building
    dataset = TracesetDataset(traceset)

    # Access the vocabulary
    vocab=dataset.build_vocab()
    print(vocab.get_stoi())

    # Example of using the dataset
    batch_size = 32
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    bptt=batch_size


    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
    #                                         sampler=train_sampler)
    # eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                                 sampler=valid_sampler)

    # train_list, eval_list = train_test_split(traceset, test_size=0.2, random_state=42)
    dataset=TracesetDataset(dataset)
    # eval_dataset=TracesetDataset(eval_list)
    train_loader=DataLoader(dataset,batch_size=batch_size)
    

    # train_list, val_list = train_test_split(traceset, test_size=0.2, random_state=42)
    # batch_size = 32
    
    # eval_batch_size = 32
    # train_data=data_process(traceset=train_list,vocab=vocab)
    # train_data = batchify(train_data, batch_size,len_sentences)

    # eval_data=data_process(traceset=val_list,vocab=vocab)
    # eval_data = batchify(eval_data, batch_size,len_sentences)

    # bptt = len_sentences-1
    # print("len_sentences",len_sentences,"bptt",len_sentences-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntokens = len(vocab)  # size of vocabulary
    emsize = 16    # embedding dimension
    d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.1  # dropout probability
    pad_idx=vocab.vocab.get_stoi()['<pad>']
    model = TransformerModel(pad_idx,ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    lr = 2.5 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = 10
    best_model = None
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("count",count_parameters(model))
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model,train_loader,bptt,optimizer,scheduler,criterion,ntokens,device,epoch,vocab)
        # val_loss = evaluate(model, eval_loader,bptt,device,len_sentences,ntokens,criterion)
        # val_ppl = math.exp(val_loss)
        # elapsed = time.time() - epoch_start_time
        # print('-' * 89)
        # print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        #     f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        # print('-' * 89)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = copy.deepcopy(model)
        # torch.save(model.state_dict(), "./data/models/"+dataset_name+"_transformers.pt")

        scheduler.step()

if __name__=="__main__":
    datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
    # datasets=["scanette_100043-steps"]

    for dataset_name in datasets:
        train_aiyn(dataset_name)