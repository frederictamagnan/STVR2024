
from ucdc.ucdc import UsageCoverageDrivenClustering
from ucdc.utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray

from ucdc.kmeans_with_sampling import Sampling
from ucdc.agglutinate import AgglutinatePipeline
from ucdc.kmedoids_pipeline import KMedoidsPipeline
from ucdc.kmeans_with_w2v import KmeansW2v
from ucdc.no_clustering_only_sampling import NoClustering,OnlySampling
from ucdc.kmeans_with_best_usage_choice import KmeansPipeline,BestUsageChoice
from ucdc.kmeans_with_sampling import Sampling
import json

from utils_experiments import load_config_dict

import matplotlib.pyplot as plt
from math import sqrt
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# import warnings
# warnings.filterwarnings("ignore")

class UCDCExperiment:
    def __init__(self,name_experiment,result_path="./results/json/results",):
        self.config_dict=load_config_dict()
        self.results={}
        self.result_path=result_path+name_experiment+'.json'

    
    def experiment(self,clustering_pip,sample_heuri):
        tests_d={}
        for i,dataset_name in enumerate(self.config_dict.keys()):
            n_clusters=[8,3,72]
            datapath=self.config_dict[dataset_name]['datapath']
            traceset=load_traceset(datapath,dataset_name)
            u=UsageCoverageDrivenClustering(clustering_pip,sample_heuri,execution_traces_agilkia_format=traceset)
            tests,usage,missing_n_grams,present_n_grams=u.inference(n_clusters[i])
            tests_d[dataset_name]={}
            tests_d[dataset_name]['tests']=tests
            tests_d[dataset_name]['usage']=usage
            tests_d[dataset_name]['present_n_grams']=present_n_grams
        with open(self.result_path, "w") as outfile:
                json.dump(tests_d, outfile)
    def frequent_n_grams(present_n_grams):
        pass
            
    


if __name__=='__main__':
    

    results_path=[]
    # pipelines=[("agglutinate",AgglutinatePipeline,Sampling)]
    
    pipelines=[("kmeansW2v",KmeansW2v,BestUsageChoice)]
    for pipeline in pipelines:
        name_experiment,clustering_pip,sample_heuri=pipeline
        ucdce=UCDCExperiment(name_experiment+"_inference_BUC")
        ucdce.experiment(clustering_pip(),sample_heuri())
        
    

    
