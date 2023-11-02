from abc import ABC, abstractmethod

class ClusteringPipeline(ABC):

    

    @abstractmethod
    def preprocessor(self,execution_traces_agilkia_format,**kwargs):
        pass
    
    @abstractmethod
    def fit_predict(self,preprocessed_execution_traces,k):
        pass

