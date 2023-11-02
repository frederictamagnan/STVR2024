from .clustering_pipeline import ClusteringPipeline
from .sample_heuristic import SampleHeuristic
from .utils_preprocessing import traceset_to_textset
from random import sample,randint
from pprint import pprint

class NoClustering(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.model=None
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        return execution_traces_agilkia_format

    def fit_predict(self, preprocessed_execution_traces,k):
        n=len(preprocessed_execution_traces)
        random_list = [randint(0, k) for _ in range(n)]
        
        return random_list

class OnlySampling(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels,**kwargs):
        nb_of_tests=cluster_labels[0]
    
        cluster_labels=[0 for i in range(len(execution_traces_agilkia_format))]
        listset=traceset_to_textset(execution_traces_agilkia_format,format='lst')
        tests_idx=sample(list(range(len(execution_traces_agilkia_format))),nb_of_tests)
        testset=[]
        for idx in tests_idx:
            testset.append(listset[idx])

        return testset



