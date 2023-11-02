from .clustering_pipeline import ClusteringPipeline
from .sample_heuristic import SampleHeuristic
from .utils_preprocessing import load_traceset,traceset_to_textset,traceset_to_pattern_one_hot,load_spmf_files

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample

class KmeansPipelinePatternOneHot(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.freq=kwargs.get("freq", 0.1)
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        
        
        X,_=load_spmf_files(filepath=self.filepath,traceset=execution_traces_agilkia_format,dataset_name=self.dataset_name,freq=self.freq)
        X = StandardScaler().fit_transform(X)
        return X

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        # model.fit(preprocessed_execution_traces)
        y=model.fit_predict(preprocessed_execution_traces)
        self.model=model
        return y

class Sampling(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):

        # listset=traceset_to_textset(execution_traces_agilkia_format,start_end_token_creator=lambda i:(list(),list()),format='lst')
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



