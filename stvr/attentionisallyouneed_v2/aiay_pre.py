from .utils import load_traceset,traceset_to_textset
from torch.utils.data import Dataset
import json
import torch
from torchtext.vocab import build_vocab_from_iterator

import random
def cut_long_sentences(sentences, max_length):
    return [sentence[:max_length] for sentence in sentences]


def load_dataset(dataset_name,max_length=32,split=False):
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


    if split:
        random.shuffle(traceset)

        # Calculate the split index based on proportions
        split_index = int(0.8 * len(traceset))

        # Divide the list into two parts
        training_set = traceset[:split_index]
        validation_set = traceset[split_index:]

        return training_set,validation_set,len_sentences
    return traceset,len_sentences


import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

class TracesetDataset(Dataset):
    def __init__(self, sentences, vocab=None):
        self.sentences = sentences
        self.tokenizer=lambda x: x.split(" ")
        self.vocab=vocab
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        if self.vocab is None:
            sentence = self.sentences[index]
            return sentence
        else:
            source = self.sentences[index]
            target = source[1:]
            source=source[:-1]
            source_tensor = torch.tensor(self.vocab(source), dtype=torch.long)
            target_tensor = torch.tensor(self.vocab(target), dtype=torch.long)
            return source_tensor, target_tensor

    @staticmethod
    def sentences_generator(dataset):
        for i in range(len(dataset)):
            yield " ".join(dataset[i])

    def build_vocab(self):
        sentences_iterator = self.sentences_generator(self)
        
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, sentences_iterator))
        return self.vocab




