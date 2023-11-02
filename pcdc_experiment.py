
from stvr.pcdc import PatternCoverageDrivenClustering
from stvr.utils_preprocessing import load_traceset
from stvr.cp_kmeans_with_sampling import Sampling
from stvr.cp_agglutinate import AgglutinatePipeline
from stvr.cp_kmedoids_pipeline import KMedoidsPipeline
from stvr.cp_pattern_encoding_kmeans import KmeansPipelinePatternOneHot
from stvr.cp_kmeans_with_w2v import KmeansW2v
from stvr.cp_no_clustering_only_sampling import NoClustering,OnlySampling
from stvr.cp_kmeans_with_sampling import Sampling
from stvr.cp_vae_encoding_kmeans import VAEEncodingKmeans
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

class PCDCExperiment:
    def __init__(self,name_experiment,result_path="./results/json/results"):
        self.config_dict=load_config_dict()
        self.results={}
        self.result_path=result_path+name_experiment+'.json'
        
    
    def experiment(self,clustering_pip,sample_heuri):
        for dataset_name in self.config_dict.keys():
            datapath=self.config_dict[dataset_name]['datapath']
            rcd=self.config_dict[dataset_name]["range"]
            experiments=self.config_dict[dataset_name]["experiments"]

            traceset=load_traceset(datapath,dataset_name)
            freq=self.config_dict[dataset_name]["freq"]
            
            clustering_pip.freq=freq
            clustering_pip.dataset_name=dataset_name
            clustering_pip.filepath=datapath
            u=PatternCoverageDrivenClustering(clustering_pipeline=clustering_pip,sample_heuristic=sample_heuri,execution_traces_agilkia_format=traceset,freq=freq,filepath=datapath,dataset_name=dataset_name)
            self.results[dataset_name]=u.finetuning_stage(epsilon=0.05,range_clusters=range(*rcd),repeat_experiments=experiments)
            with open(self.result_path, "w") as outfile:
                json.dump(self.results, outfile)
    
    def plot_multiple(self,results,metadata=""):
        
        
        datasets=list(results[list(results.keys())[0]].keys())
        colors=['b','r','k','g','c','m','y']
        for dataset in datasets:
            plt.figure()
           
            color_index=0
            for experiment,result in results.items():
                print(result)
                c=colors[color_index]
                plot_results=result[dataset]['usage_coverage_by_clusters']
                x=[a for a,b,c in plot_results]
                y=[b for a,b,c in plot_results]
                z=[c for a,b,c in plot_results]
                w=[1.96*c/sqrt(30) for a,b,c in plot_results]

                sns.set()

                
                mean_1 = np.array(y)
                std_1 = np.array(w)
                # plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
                plt.plot(x, mean_1, 'b-', label=experiment,color=c)
                plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=c, alpha=0.2)
                
                color_index+=1
                # plt.legend(title='Usage Coverage per # of clusters/tests',prop={'size': 8})
                plt.legend(prop={'size': 8.25})

                plt.show()

            # plt.savefig("./results/fig/experiments_"+dataset+"_final_2.png",format="png")
            plt.savefig("./results/fig/aaa_experiments_"+dataset+"_final_2"+metadata+".png",format="png")
        
        datasets=list(results[list(results.keys())[0]].keys())
        colors=['b','r','k','g','c','m','y']
        for dataset in datasets:
            plt.figure()
           
            color_index=0
            for experiment,result in results.items():
                print(result)
                c=colors[color_index]
                plot_results=result[dataset]['distance_by_clusters']
                x=[a for a,b,c in plot_results]
                y=[b for a,b,c in plot_results]
                z=[c for a,b,c in plot_results]
                w=[1.96*c/sqrt(30) for a,b,c in plot_results]

                sns.set()

                
                mean_1 = np.array(y)
                std_1 = np.array(z)
                # plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
                plt.plot(x, mean_1, 'b-', label=experiment,color=c)
                plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=c, alpha=0.2)
                
                color_index+=1
                # plt.legend(title='Usage Coverage per # of clusters/tests',prop={'size': 8})
                plt.legend(prop={'size': 8.25})

                plt.show()

            # plt.savefig("./results/fig/experiments_"+dataset+"_final_2.png",format="png")
            plt.savefig("./results/fig/aaa_experiments_distance_"+dataset+"_final_2"+metadata+".png",format="png")


if __name__=='__main__':
    import torch
    
    pipelines=[("KmeansPattern",KmeansPipelinePatternOneHot,Sampling)]
    pipelines+=[("NoClusteringOnlySampling",NoClustering,OnlySampling)]
    
    pipelines+=[("kmeansW2v",KmeansW2v,Sampling)]

    
    experiment=True
    plot=False
    meta="1304"
    if experiment:
        results_path=[]
        # pipelines=[("agglutinate",AgglutinatePipeline,Sampling)]
        
        for pipeline in pipelines:
            name_experiment,clustering_pip,sample_heuri=pipeline
            pcdce=PCDCExperiment(name_experiment=name_experiment+meta)
            pcdce.experiment(clustering_pip(),sample_heuri())
            results_path.append((name_experiment,pcdce.result_path))

    
