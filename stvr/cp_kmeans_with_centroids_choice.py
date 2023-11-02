from .clustering_pipeline import ClusteringPipeline
from .sample_heuristic import SampleHeuristic
from .utils_preprocessing import traceset_to_textset,textset_to_bowarray
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample
import numpy as np
from scipy.cluster.vq import vq
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
        y=model.fit_predict(preprocessed_execution_traces)
        # model.fit(preprocessed_execution_traces)
        self.model=model
        return y

class CentroidsChoice(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels,**kwargs):

        listset=traceset_to_textset(execution_traces_agilkia_format,format='lst')
        
        # #arrange execution traces index by cluster id in a dict
        # idx_by_c=defaultdict(list)
        # for idx,c in enumerate(cluster_labels):
        #     idx_by_c[c].append(idx)
        # tests_idx=[]
        # #sample one trace index for each cluster
        # for c,list_of_idx in idx_by_c.items():
        #     tests_idx.append(sample(list_of_idx,1)[0])
        # testset=[]
        # #extract the corresponding traces in a test set
        # for idx in tests_idx:
        #     testset.append(listset[idx])
        
        tf_matrix= kwargs.get("X", None)
        m_km= kwargs.get("model", None)
        print(m_km)
        num_clusters=max(cluster_labels)+1
        m_clusters=cluster_labels


        centers = np.array(m_km.cluster_centers_)

        closest_data = []
        for i in range(num_clusters):
            center_vec = centers[i]
            data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

            one_cluster_tf_matrix = np.zeros( (  len(data_idx_within_i_cluster) , centers.shape[1] ) )
            for row_num, data_idx in enumerate(data_idx_within_i_cluster):
                one_row = tf_matrix[data_idx]
                one_cluster_tf_matrix[row_num] = one_row
            
            closest, _ = vq(center_vec.reshape(1,-1), one_cluster_tf_matrix)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
            data_id =closest_data_row_num

            closest_data.append(data_id)

        closest_data = list(set(closest_data))

        assert len(closest_data) == num_clusters
        tests_idx=closest_data
        testset=[]
        #extract the corresponding traces in a test set
        for idx in tests_idx:
            testset.append(listset[idx])
        return testset,tests_idx




