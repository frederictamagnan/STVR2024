from .clustering_pipeline import ClusteringPipeline
from .utils_preprocessing import traceset_to_textset
from sklearn.cluster import AgglomerativeClustering
import numpy as np

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

class AgglutinatePipeline(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.model=None
        self.freq=kwargs.get("freq", 0.1)

    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        textset=traceset_to_textset(execution_traces_agilkia_format)

        X=np.zeros((len(textset),len(textset)))
        for i,sentence1 in enumerate(textset):
            for j,sentence2 in enumerate(textset):
                X[i,j]=self.similarity(sentence1,sentence2)
            if i % 10==0:
                print(i)
        return X
                
        
        
    def fit_predict(self, preprocessed_execution_traces,k):
        model=AgglomerativeClustering(n_clusters=k,affinity='precomputed',linkage='single')
        model.fit(preprocessed_execution_traces)
        return model.fit_predict(preprocessed_execution_traces)

    def similarity(self,trace1,trace2):
        
        s_trace1=set(trace1)
        s_trace2=set(trace2)
        a=len(s_trace1.difference(s_trace2))
        b=len(s_trace2.difference(s_trace1))
        c=len(s_trace1.intersection(s_trace2))
        return (a+b)/(a+b+c)
        
    




