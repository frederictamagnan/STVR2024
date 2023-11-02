from .clustering_pipeline import ClusteringPipeline
from .utils_preprocessing import traceset_to_textset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils_preprocessing import load_traceset,traceset_to_textset,traceset_to_pattern_one_hot,load_spmf_files
import gensim
import numpy as np

class KmeansW2vPlus(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.freq=kwargs.get("freq", 0.1)
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        listset=traceset_to_textset(execution_traces_agilkia_format,format='lst')
        # print(listset[0])
        model = gensim.models.Word2Vec(sentences=listset,vector_size=10,window=5,min_count=1)
        means=[]
        for seq in listset:
            vecs=[model.wv[elt] for elt in seq]
            means.append(np.mean(vecs,axis=0))
        e=np.array(means)
        
        # return X
        X,_=load_spmf_files(filepath=self.filepath,traceset=execution_traces_agilkia_format,dataset_name=self.dataset_name,freq=self.freq)
        X = StandardScaler().fit_transform(X)
        return np.concatenate((X,e),axis=1)
    
    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        y=model.fit_predict(preprocessed_execution_traces)
        self.model=model
        return  y



