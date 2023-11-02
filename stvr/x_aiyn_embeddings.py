from attentionisallyouneed.aiay_pre import load_dataset,process,data_process,batchify
from attentionisallyouneed.aiay_model import TransformerModel
from attentionisallyouneed.aiay_train import train,evaluate,get_batch,generate_square_subsequent_mask
from sklearn.model_selection import train_test_split
import copy
import torch
import time
import math
import sys
def get_embeddings(dataset_name):
    max_length=32
    traceset,len_sentences=load_dataset(dataset_name,max_length=32)
    vocab=process(traceset)
    print(len(traceset))
    # train_list, val_list = train_test_split(traceset, test_size=0.2, random_state=42)
    batch_size = 32
    print("batch size is",batch_size)
    train_data=data_process(traceset=traceset,vocab=vocab)
    print("first len",len(train_data))
    train_data = batchify(train_data, batch_size,len_sentences)
    print("second len",len(train_data))
    print("len_sentences",len_sentences,"bptt",len_sentences-1)


    bptt = len_sentences-1

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
    print("len train data",train_data.size(0))
    with torch.no_grad():
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        num_batches = len(train_data) // len_sentences
        all_embeddings = []
        # for batch, i in enumerate(range(0, train_data.size(0) - 1, len_sentences)):
        #     data, targets = get_batch(train_data, i,bptt)
        #     batch_size = data.size(0)
        #     if batch_size != bptt:  # only on last batch
        #         print("WARNING")
        #         sys.exit(1)
        #         src_mask = src_mask[:batch_size, :batch_size]
        #     embeddings = model(data, src_mask)
        for i in range(0, train_data.size(0) - 1, len_sentences):
            data, targets = get_batch(train_data, i,bptt)
            print("datasize",data.size())
            batch_size = data.size(0)
            if batch_size != bptt:
                break
                src_mask = src_mask[:batch_size, :batch_size]
            embeddings = model(data, src_mask)
            all_embeddings.append(embeddings)
        
        print("nb batcg final",i)
        concatenated_embeddings = torch.cat(all_embeddings, dim=0)
        numpy_embeddings = concatenated_embeddings.numpy()
        print(numpy_embeddings.shape)
if __name__=='__main__':
    dataset_name="scanette_100043-steps"
    get_embeddings(dataset_name=dataset_name)