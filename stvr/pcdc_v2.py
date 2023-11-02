from statistics import mean,stdev
from .pattern_coverage import PatternCoverage
import time
class PatternCoverageDrivenClustering:
    
    def __init__(self,name_exp,clustering_pipeline,sample_heuristic,execution_traces_agilkia_format,filepath,dataset_name,cluster_nb,freq=[0.1,0.2]):
        self.name_exp = name_exp
        self.clustering_pipeline = clustering_pipeline
        self.sample_heuristic=sample_heuristic
        self.execution_traces_agilkia_format=execution_traces_agilkia_format
        self.dataset_name=dataset_name
        self.pc={}
        for f in freq:
            self.pc[f]=PatternCoverage(reference_traceset=self.execution_traces_agilkia_format,freq=f,dataset_name=dataset_name,filepath=filepath)
        self.freq=freq
        self.cluster_nb=cluster_nb
    def compute(self,repeat_experiments=20):

        b=time.time()
        X=self.clustering_pipeline.preprocessor(self.execution_traces_agilkia_format)      
        usage_experiments={}
        distance_experiments={}
        for f in self.freq:
            usage_experiments[f]=[]
            distance_experiments[f]=[]

        for experiment in range(repeat_experiments):
            print("--------nb experiment :  "+ str(experiment))
            # kwargs={'model':self.clustering_pipeline.model,"X":X}
            tests,clusters=self.clustering_and_heuristic(X,self.cluster_nb)

            for f in self.freq:
                usage_experiments[f].append(self.compute_pattern_usage(tests,self.pc[f]))
                print(usage_experiments[f])
                distance_experiments[f].append(self.compute_pattern_distance(clusters,self.pc[f]))
        output_data=[]
        for f in self.freq:
            d={
                "name_exp":self.name_exp,
                "cluster_nb" : self.cluster_nb,
                "clustering_pipeline":type(self.clustering_pipeline).__name__,
                "sample_heuristic":type(self.sample_heuristic).__name__,
                "pattern_coverage":(mean(usage_experiments[f]),stdev(usage_experiments[f])),
                "pattern_coverage_raw_data":usage_experiments[f],
                "distance_experiments":(mean(distance_experiments[f]),stdev(distance_experiments[f])),
                "time":time.time()-b,
                "coverage_freq":f,
                "dataset_name":self.dataset_name
            }
            output_data.append(d)
        return  output_data
    
    def clustering_and_heuristic(self,X,nb_clusters):
        cluster_labels=self.clustering_pipeline.fit_predict(X,nb_clusters)
        # print("hep",type(self.clustering_pipeline.model))
        kwargs={'model':self.clustering_pipeline.model,'X':X,"lst_patterns":self.pc[min(self.freq)].lst_patterns}
        # print(kwargs)
        tests=self.sample_heuristic.tests_extraction(execution_traces_agilkia_format=self.execution_traces_agilkia_format,cluster_labels=cluster_labels,**kwargs)
        return tests,cluster_labels 
    
    def compute_pattern_usage(self,tests,pc):
        return pc.pattern_usage(tests)
    def compute_pattern_distance(self,clusters,pc):
        return pc.pattern_distance(clusters)
    
