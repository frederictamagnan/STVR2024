from .clustering_pipeline import ClusteringPipeline
from sklearn.cluster import KMeans
from .utils_vae import split_dataset,main_test
import numpy as np
class VAEEncodingKmeans(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
        self.arch=kwargs.get("arch", "NoArch")
        self.freq=kwargs.get("freq", "NoFreq")
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        
        
        # split_dataset(execution_traces_agilkia_format,self.dataset_name)
        #parameters : dataset + arch
        z=main_test(self.dataset_name,self.arch)


      


        return z


    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        y=model.fit_predict(preprocessed_execution_traces)
        self.model=model
        return y





