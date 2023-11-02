from attentionisallyouneed.aiay_pre import load_dataset,process,data_process,batchify,MyDataset
from attentionisallyouneed.aiay_model import TransformerModel
from attentionisallyouneed.aiay_train import train,evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import copy
import torch
import time
import math

def train_aiyn(dataset_name):
    max_length=32
    traceset,len_sentences=load_dataset(dataset_name,max_length=32)

    vocab=process(traceset)

    train_list, val_list = train_test_split(traceset, test_size=0.2, random_state=42)
    batch_size = 32
    
    eval_batch_size = 32
    train_data=data_process(traceset=train_list,vocab=vocab)
    train_data = batchify(train_data, batch_size,len_sentences)

    eval_data=data_process(traceset=val_list,vocab=vocab)
    eval_data = batchify(eval_data, batch_size,len_sentences)

    bptt = len_sentences-1
    print("len_sentences",len_sentences,"bptt",len_sentences-1)
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
        train(model,train_data,len_sentences,bptt,optimizer,scheduler,criterion,ntokens,device,epoch)
        val_loss = evaluate(model, eval_data,bptt,device,len_sentences,ntokens,criterion)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            # torch.save(model.state_dict(), "./data/models/"+dataset_name+"_transformers.pt")

        scheduler.step()

if __name__=="__main__":
    # datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
    datasets=["scanette_100043-steps"]

    for dataset_name in datasets:
        train_aiyn(dataset_name)