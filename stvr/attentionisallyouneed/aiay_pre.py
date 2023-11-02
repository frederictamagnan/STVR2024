from utils_preprocessing import load_traceset,traceset_to_textset
from torch.utils.data import Dataset
import json
import torch
from torchtext.vocab import build_vocab_from_iterator


def cut_long_sentences(sentences, max_length):
    return [sentence[:max_length] for sentence in sentences]


def load_dataset(dataset_name,max_length=32):
    traceset=load_traceset("./data/datasets/",dataset_name)
    
    
    
    
    traceset=traceset_to_textset(traceset,format="lst")
    traceset=cut_long_sentences(traceset,max_length=max_length)
    l=[]
    for trace in traceset:
        
        l.append(len(trace[1:-1]))

    maximum=max(l)
    print("maximum",maximum)
    len_sentences=maximum+2
    l=[]
    i=0
    for i,trace in enumerate(traceset):
        traceset[i]=['<sos>']+trace[1:-1]+['<eos>']+['<pad>' for i in range(maximum-len(trace[1:-1]))]
    
    # print(len(traceset[0]))
    # print(len(traceset[1]))
    #Verify that all the sessions are well padded
    return traceset,len_sentences


class TracesetDataset(Dataset):
    def __init__(self, traceset):
        self.traceset = traceset
    def __len__(self):
        return len(self.traceset)
    
    def __getitem__(self, index):
        
        return " ".join(self.traceset[index])


def sentences_generator(dataset):
    for i in range(len(dataset)):
        yield dataset[i]

def process(traceset):
    dataset = TracesetDataset(traceset)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    tokenizer=lambda x: x.split(" ")
    vocab = build_vocab_from_iterator(map(tokenizer, sentences_generator(dataset)))
    # print(vocab.vocab.get_itos())
    # print(vocab.vocab.get_stoi())
    return vocab

def data_process(traceset,vocab):
    """Converts raw text into a flat Tensor."""
    dataset = TracesetDataset(traceset)
    iterator=sentences_generator(dataset)
    tokenizer=lambda x: x.split(" ")
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in iterator]
    print(torch.cat(tuple(filter(lambda t: t.numel() > 0, data))))
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data,bsz,len_sentences) :
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_sentences = data.size(0)//len_sentences
    sentences_per_batch=nb_sentences//bsz
    seq_len=len_sentences*sentences_per_batch
    print(seq_len,"seq_len")
    print(nb_sentences,"nb sentences keeped")
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


