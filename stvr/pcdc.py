from statistics import mean,stdev
from .pattern_coverage import PatternCoverage
import time
class PatternCoverageDrivenClustering:
    
    def __init__(self,clustering_pipeline,sample_heuristic,execution_traces_agilkia_format,filepath,dataset_name,freq=0.1,):
        
        
        self.clustering_pipeline = clustering_pipeline
        self.sample_heuristic=sample_heuristic
        self.execution_traces_agilkia_format=execution_traces_agilkia_format
        
        self.uc=PatternCoverage(reference_traceset=self.execution_traces_agilkia_format,freq=freq,dataset_name=dataset_name,filepath=filepath)
        
        

    def finetuning_stage(self,range_clusters=range(2,20),epsilon=0.05,repeat_experiments=20):
        usage_coverage_stats=0
        recorded_usage_coverage_stats=[]
        
        distance_stats=0
        recorded_distance_stats=[]
        results={}
        b=time.time()
        X=self.clustering_pipeline.preprocessor(self.execution_traces_agilkia_format)
        
        for nb_clusters in range_clusters:
            print("nb cluster "+ str(nb_clusters))
            usage_experiments=[]
            distance_experiments=[]
            for experiment in range(repeat_experiments):
                print("--------nb experiment :  "+ str(experiment))
                tests,tests_idx,clusters=self.clustering_and_heuristic(X,nb_clusters)
                usage_experiments.append(self.compute_pattern_usage(tests))
                distance_experiments.append(self.compute_pattern_distance(clusters))
            usage_coverage_stats=(mean(usage_experiments),stdev(usage_experiments))
            distance_stats=(mean(distance_experiments),stdev(distance_experiments))
            print("--usage coverage: "+ str(usage_coverage_stats))
            print("--distance: "+ str(distance_stats))
            recorded_usage_coverage_stats.append(usage_coverage_stats)
            recorded_distance_stats.append(distance_stats)
            if 1-usage_coverage_stats[0]<epsilon:
                epsilon=-100
                results['best_nb_of_clusters']=nb_clusters
        
        results['usage_coverage_by_clusters']=[(a,b,c) for (a,(b,c)) in zip(list(range_clusters),recorded_usage_coverage_stats)]
        results['distance_by_clusters']=[(a,b,c) for (a,(b,c)) in zip(list(range_clusters),recorded_distance_stats)]
        results['time']=time.time()-b
        return  results
    
    def inference(self, nb_clusters):

        b=time.time()
        X=self.clustering_pipeline.preprocessor(self.execution_traces_agilkia_format)
        tests,tests_idx=self.clustering_and_heuristic(X,nb_clusters)
        usage=self.compute_pattern_usage(tests)
        return  tests,tests_idx
        
    
    def clustering_and_heuristic(self,X,nb_clusters):
        cluster_labels=self.clustering_pipeline.fit_predict(X,nb_clusters)
        tests,tests_idx=self.sample_heuristic.tests_extraction(self.execution_traces_agilkia_format,cluster_labels)
        return tests,tests_idx,cluster_labels 
    
    def compute_usage(self,tests):
        usage,missing_n_grams,present_n_grams=self.uc.usage(tests)
        return usage
    
    def compute_usage_and_ngrams(self,tests):
        usage,missing_n_grams,present_n_grams=self.uc.usage(tests)
        return usage,missing_n_grams,present_n_grams
    
    def compute_pattern_usage(self,tests):
        return self.uc.pattern_usage(tests)
    def compute_pattern_distance(self,clusters):
        return self.uc.pattern_distance(clusters)