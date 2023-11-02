from .clustering_pipeline import ClusteringPipeline
from .sample_heuristic import SampleHeuristic
from .utils_preprocessing import traceset_to_textset,textset_to_bowarray
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample
from .utils_preprocessing import traceset_to_textset,textset_to_spmf,id_to_words,pattern_in_sentence,load_spmf_files
from copy import deepcopy
from .pattern_coverage import PatternCoverage
class KmeansPipeline(ClusteringPipeline):
    
    def __init__(self,dataset_name,filepath,**kwargs):
        self.dataset_name=dataset_name
        self.filepath=filepath
    def preprocessor(self, execution_traces_agilkia_format,**kwargs):
        
        textset=traceset_to_textset(execution_traces_agilkia_format)
        X,voc=textset_to_bowarray(textset) 
        X = StandardScaler().fit_transform(X)
        
        return X

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        # model.fit(preprocessed_execution_traces)
        return model.fit_predict(preprocessed_execution_traces)

class BUC(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels,**kwargs):
        self.lst_patterns=kwargs.get("lst_patterns", None)
        listset=traceset_to_textset(execution_traces_agilkia_format,format='lst')
        
        #arrange execution traces index by cluster id in a dict
        idx_by_c=defaultdict(list)
        for idx,c in enumerate(cluster_labels):
            idx_by_c[c].append(idx)
        
        #determine the patterns for each cluster
        patterns_by_c_d={}
        for k  in idx_by_c.keys():
            patterns_by_c_d[k]={}
        for cluster_id,list_traces_idx in idx_by_c.items():
            for idx_pattern,pattern in enumerate(self.lst_patterns):
                for trace_idx in list_traces_idx:
                    if pattern_in_sentence(pattern[0],listset[trace_idx]):
                        
                        if idx_pattern in patterns_by_c_d[cluster_id]:
                            patterns_by_c_d[cluster_id][idx_pattern]+=1
                        else:
                            patterns_by_c_d[cluster_id][idx_pattern]=1
        patterns_by_c={}
        for k  in idx_by_c.keys():
            patterns_by_c[k]=[]
        for k,v in patterns_by_c_d.items():
            for k1,v1 in patterns_by_c_d[k].items():
                # print(k1,v1)
                tf = v1/len(idx_by_c[k])
                idf=(len(listset)/self.lst_patterns[k1][1])
                patterns_by_c[k].append((self.lst_patterns[k1][0],tf*idf))
                # if tf*idf !=  1:
                #     print( tf*idf)
            # if len(patterns_by_c[cluster_id])!=0:
            #     patterns_by_c[cluster_id]=sorted(patterns_by_c[cluster_id], key=lambda tup: tup[1],reverse=True)



        tests=[]
        
        for cluster_id,list_traces_idx in idx_by_c.items():
            pattern_coverage=PatternCoverage(None,None,None,None,None,raw=True,lst_patterns=patterns_by_c[cluster_id])
            candidates=[]
            for trace_idx in list_traces_idx:
                candidates.append(listset[trace_idx])
            score_candidates=[]
            for candidate in candidates:
                score_candidates.append((candidate,pattern_coverage.pattern_usage(candidate)))
            sorted_score_candidates=sorted(score_candidates, key=lambda x: x[1],reverse=True)

            # frozen_candidates=[None]
            # frozen_candidates_idx=[]
            # candidates_idx=list(range(len(candidates)))
            # idx_pattern=0
            # while len(candidates)>0 and idx_pattern<len(patterns_by_c[cluster_id]):
                
            #     wrong_indexes=[]
            #     frozen_candidates=deepcopy(candidates)
            #     frozen_candidates_idx=deepcopy(candidates_idx)
            #     for idx,trace in enumerate(candidates):
            #         if pattern_in_sentence(patterns_by_c[cluster_id][idx_pattern][0],trace):
            #             wrong_indexes.append(idx)
            #     for index in sorted(wrong_indexes, reverse=True):
            #         del candidates[index]
            #         del candidates_idx[index]
                
            #     idx_pattern+=1
            
            # print("found "+str(len(frozen_candidates))+" candidates for cluster "+str(cluster_id)+" and reached the "+str(idx_pattern)+"-th n gram")
            tests+=[sorted_score_candidates[0][0]]
        
        print(tests)
        print('len',len(tests))
        return tests