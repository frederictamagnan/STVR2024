from .clustering_pipeline import ClusteringPipeline
from .utils_preprocessing import traceset_to_textset
from pyclustering.cluster.kmedoids import kmedoids
import numpy as np
from math import sqrt

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

class KMedoidsPipeline(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        textset=traceset_to_textset(execution_traces_agilkia_format)
        lang=Lang("traces")
        for sentence in textset:
            lang.addSentence(sentence)
        
        listset=[indexesFromSentence(lang,sentence) for sentence in textset]
        X=np.zeros((len(textset),len(textset)))
        for i,sentence1 in enumerate(listset):
            for j,sentence2 in enumerate(listset):
                X[i,j]=self.similarity(sentence1,sentence2)
            if i % 10==0:
                print(i)
        return X
                
        
        
    def fit_predict(self, preprocessed_execution_traces,k):
        initial_medoids=list(range(0,k))
        kmedoids_instance=kmedoids(preprocessed_execution_traces,initial_medoids,data_type='distance_matrix')
        kmedoids_instance.process()
        clusters_list=kmedoids_instance.get_clusters()
        self.model=kmedoids_instance
        labels=[0 for i in range(len(preprocessed_execution_traces))]
        for k,cluster in enumerate(clusters_list):
            for idx in cluster:
                labels[idx]=k
        return labels

    def similarity(self,trace1,trace2):
        
        raw_trace1=trace1
        raw_trace2=trace2
        
        raw_trace1=raw_trace1[:min(len(trace1),len(trace2))]
        raw_trace2=raw_trace2[:min(len(trace1),len(trace2))]
        
        arr_trace1=np.array(raw_trace1)
        arr_trace2=np.array(raw_trace2)
        
        
        comparison=arr_trace1!=arr_trace2
        equal_array=1*comparison.all()
        return sqrt(np.sum(equal_array)**2+(len(trace1)-len(trace2))**2)
        
    



