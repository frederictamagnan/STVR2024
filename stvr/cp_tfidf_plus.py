from .clustering_pipeline import ClusteringPipeline
from .sample_heuristic import SampleHeuristic
from .utils_preprocessing import traceset_to_textset,textset_to_bowarray
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils_preprocessing import load_traceset,traceset_to_textset,traceset_to_pattern_one_hot,load_spmf_files
import numpy as np
class TfIdfPlus(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.freq=kwargs.get("freq", 0.1)
        self.model=None
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        
        textset=traceset_to_textset(execution_traces_agilkia_format)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(textset)
        print(type(X))
        X=X.toarray()
        W,_=load_spmf_files(filepath=self.filepath,traceset=execution_traces_agilkia_format,dataset_name=self.dataset_name,freq=self.freq)
        W= StandardScaler().fit_transform(W)
        return np.concatenate((X,W),axis=1)

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        # model.fit(preprocessed_execution_traces)
        return model.fit_predict(preprocessed_execution_traces)

class Sampling(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels,**kwargs):

        listset=traceset_to_textset(execution_traces_agilkia_format,format='lst')
        
        #arrange execution traces index by cluster id in a dict
        idx_by_c=defaultdict(list)
        for idx,c in enumerate(cluster_labels):
            idx_by_c[c].append(idx)
        tests_idx=[]
        #sample one trace index for each cluster
        for c,list_of_idx in idx_by_c.items():
            tests_idx.append(sample(list_of_idx,1)[0])
        testset=[]
        #extract the corresponding traces in a test set
        for idx in tests_idx:
            testset.append(listset[idx])
        return testset,tests_idx



