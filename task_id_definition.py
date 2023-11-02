from utils_experiments import load_config_dict


from stvr.pcdc_v2 import PatternCoverageDrivenClustering
from stvr.utils_preprocessing import load_traceset
from stvr.cp_kmeans_with_sampling import Sampling
from stvr.cp_agglutinate import AgglutinatePipeline
from stvr.cp_kmedoids_pipeline import KMedoidsPipeline
from stvr.cp_pattern_encoding_kmeans import KmeansPipelinePatternOneHot
from stvr.cp_kmeans_with_w2v import KmeansW2v
from stvr.cp_no_clustering_only_sampling import NoClustering,OnlySampling
from stvr.cp_kmeans_with_sampling import Sampling
from stvr.cp_vae_encoding_kmeans import VAEEncodingKmeans

from stvr.cp_kmeans_with_centroids_choice import CentroidsChoice,KmeansPipeline
from pprint import pprint
from stvr.cp_aiay_encoding_kmeans_v2 import AIAY
from stvr.cp_aiay_encoding_plus_v2 import AIAYPlus
from stvr.cp_kmeans_with_w2v_plus import KmeansW2vPlus
from stvr.cp_vae_encoding_plus import VAEEncodingKmeansPlus
# from stvr.cp_kmeans_with_best_usage_choice import BUC
from stvr.cp_kmeans_with_best_usage_choice_v3 import BUC
from stvr.cp_tfidf import TfIdf
from stvr.cp_tfidf_plus import TfIdfPlus
from stvr.cp_bow_plus import BowPlus
# from stvr.cp_pattern_encoding_kmeans import KmeansPipelinePatternOneHot
import sys
import json
import uuid
# from stvr.utils_vae import argss,testargs
from stvr.utils_vae import Hyperparameters

datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","scanette_100043-steps"]

freq_dict={
    "spree_5000_session_wo_responses_agilkia":{"freq":0.5},
    "femto_booking_agilkia_v6":{"freq":0.06},
    "scanette_100043-steps":{"freq":0.2},
    "teaming_execution":{"freq":0.005}
}


vae_dict={dataset_name:{'arch':'VAE',"freq":freq_dict[dataset_name]['freq']} for dataset_name in datasets}
ae_dict={dataset_name:{"arch":"AE","freq":freq_dict[dataset_name]['freq']} for dataset_name in datasets}
daae_dict={dataset_name:{"arch":"DAAE","freq":freq_dict[dataset_name]['freq']} for dataset_name in datasets}

# print(vae_dict)
pipelines=[]


pipelines+=[("BOW|Kmeans|RS",KmeansPipeline,Sampling,vae_dict)]
pipelines+=[("BOW|Kmeans|BUC",KmeansPipeline,BUC,vae_dict)]
pipelines+=[("BOW+POH|Kmeans|BUC",BowPlus,BUC,vae_dict)]
pipelines+=[("BOW+POH|Kmeans|RS",BowPlus,Sampling,vae_dict)]
pipelines+=[("TermFreq|Kmeans|RS",TfIdf,Sampling,vae_dict)]
pipelines+=[("TermFreq|Kmeans|BUC",TfIdf,BUC,vae_dict)]
pipelines+=[("TermFreq+POH|Kmeans|RS",TfIdfPlus,Sampling,vae_dict)]
pipelines+=[("TermFreq+POH|Kmeans|BUC",TfIdfPlus,BUC,vae_dict)]
pipelines+=[("-|AHC|RS",AgglutinatePipeline,Sampling,vae_dict)]
pipelines+=[("-|Baseline|RS",NoClustering,OnlySampling,freq_dict)]
pipelines+=[("-|Kmedoids|RS",KMedoidsPipeline,Sampling,freq_dict)]
pipelines+=[("W2V|Kmeans|RS",KmeansW2v,Sampling,vae_dict)]
pipelines+=[("W2V|Kmeans|BUC",KmeansW2v,BUC,vae_dict)]
pipelines+=[("W2V+POH|Kmeans|RS",KmeansW2vPlus,Sampling,vae_dict)]
pipelines+=[("W2V+POH|Kmeans|BUC",KmeansW2vPlus,BUC,vae_dict)]
pipelines+=[("AE|Kmeans|RS",VAEEncodingKmeans,Sampling,ae_dict)]
pipelines+=[("AE|Kmeans|BUC",VAEEncodingKmeans,BUC,ae_dict)]
pipelines+=[("AE+POH|Kmeans|RS",VAEEncodingKmeansPlus,Sampling,ae_dict)]
pipelines+=[("AE+POH|Kmeans|BUC",VAEEncodingKmeansPlus,BUC,ae_dict)]
pipelines+=[("VAE|Kmeans|RS",VAEEncodingKmeans,Sampling,vae_dict)]
pipelines+=[("VAE|Kmeans|BUC",VAEEncodingKmeans,BUC,vae_dict)]
pipelines+=[("VAE+POH|Kmeans|BUC",VAEEncodingKmeansPlus,BUC,vae_dict)]
pipelines+=[("VAE+POH|Kmeans|RS",VAEEncodingKmeansPlus,Sampling,vae_dict)]
pipelines+=[("DAAE|Kmeans|RS",VAEEncodingKmeans,Sampling,daae_dict)]
pipelines+=[("DAAE|Kmeans|BUC",VAEEncodingKmeans,BUC,daae_dict)]
pipelines+=[("DAAE+POH|Kmeans|RS",VAEEncodingKmeansPlus,Sampling,vae_dict)]
pipelines+=[("DAAE+POH|Kmeans|BUC",VAEEncodingKmeansPlus,BUC,daae_dict)]
pipelines+=[('TF|Kmeans|RS',AIAY,Sampling,ae_dict)]
pipelines+=[('TF|Kmeans|BUC',AIAY,BUC,ae_dict)]
pipelines+=[('TF+POH|Kmeans|BUC',AIAYPlus,BUC,ae_dict)]
pipelines+=[('TF+POH|Kmeans|RS',AIAYPlus,Sampling,ae_dict)]

pipelines+=[("POH|Kmeans|RS",KmeansPipelinePatternOneHot,Sampling,vae_dict)]
pipelines+=[("POH|Kmeans|BUC",KmeansPipelinePatternOneHot,BUC,vae_dict)]


pipelines+=[("AEplusSampling",VAEEncodingKmeansPlus,Sampling,ae_dict)]

pipelines+=[('transformersCentroid',AIAY,CentroidsChoice,ae_dict)]
def max_task():
    config_dict_exp=load_config_dict()['experiments']
    i=1
    for dataset,v in config_dict_exp.items():
        for nb_cluster in range(*v['range']):
            for pipeline in pipelines:
                i+=1    
    return i

def return_task(task_id):
    config_dict_exp=load_config_dict()['experiments']
    i=0
    for dataset,v in config_dict_exp.items():
        for nb_cluster in range(*v['range']):
            for pipeline in pipelines:
                if i==task_id:
                    return dataset,nb_cluster,pipeline
                else:
                    i+=1    
    print("max",i)   
    return None,None,None


def compute(dataset_name,cluster_nb,pipeline_tuple):
    if dataset_name is None:
        print('no task fo this id')
        return 0
    config_dict=load_config_dict()
    config_dict_exp=config_dict['experiments']
    config_dict_hp=config_dict['hp']
    filepath=config_dict_exp[dataset_name]['datapath']
    freq=config_dict_exp[dataset_name]['freq']
    traceset=load_traceset(filepath,dataset_name)

    name_exp=pipeline_tuple[0]
    print(pipeline_tuple[3])
    print(pipeline_tuple[3][dataset_name])
    clustering_pipeline=pipeline_tuple[1](dataset_name=dataset_name,filepath=filepath,**pipeline_tuple[3][dataset_name])
    sample_heuristic=pipeline_tuple[2]()
    nb_repetition=config_dict_hp['nb_repetition']
    p=PatternCoverageDrivenClustering(name_exp=name_exp,clustering_pipeline=clustering_pipeline,sample_heuristic=sample_heuristic,execution_traces_agilkia_format=traceset,filepath=filepath,dataset_name=dataset_name,freq=freq,cluster_nb=cluster_nb)
    
    output_data=p.compute(repeat_experiments=nb_repetition)
    print(output_data)
    filename = f'output_{uuid.uuid4()}.jsonl'
    with open(filename, 'w') as f:
        for obj in output_data:
            json.dump(obj, f)
            f.write('\n')


task_id=0

# task_id = int(sys.argv[1])

a,b,c=return_task(task_id)

compute(a,b,c)

