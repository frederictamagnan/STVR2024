from spmf import Spmf
from .utils_preprocessing import traceset_to_textset,textset_to_spmf,id_to_words,pattern_in_sentence,load_spmf_files
import numpy as np
from sklearn.metrics import davies_bouldin_score
import os
class PatternCoverage:
    def __init__(self,reference_traceset,dataset_name,filepath,freq=0.1,spmf_bin_location_dir="./stvr/", raw=False,lst_patterns=None):
        self.reference_traceset=reference_traceset
        
    
        # textset=traceset_to_textset(self.reference_traceset,format='str')
        # voc=textset_to_spmf(textset,filepath="./data/",filename="_spmf_dataset_v2.txt")
        # print(os.getcwd())
        # spmf = Spmf("ClaSP", input_filename="./data/_spmf_dataset_v2.txt",
        #         output_filename="./results/output.txt", arguments=[freq,False],spmf_bin_location_dir='./stvr/')

        # spmf.run()
        # df=spmf.to_pandas_dataframe(pickle=True)
        # voc_=inv_map = {v: k for k, v in voc.items()}
        
        # self.lst_patterns=[]
        # for index, row in df.iterrows():
        #     self.lst_patterns.append((id_to_words(row['pattern'],voc_), row['sup']))
            
        # #added for Davies Bouldin
        # textset=traceset_to_textset(self.reference_traceset,format='lst')
        # lst_encoding=[]
        # for i,trace in enumerate(textset):
        #     lst_encoding.append([0 for elt in self.lst_patterns])
        #     for j,pattern in enumerate(self.lst_patterns):
        #         if pattern_in_sentence(pattern[0],trace):
        #             lst_encoding[i][j]=1

        # self.X = np.array(lst_encoding)
        if not(raw):
            self.X,self.lst_patterns=load_spmf_files(filepath=filepath,traceset=reference_traceset,dataset_name=dataset_name,freq=freq,spmf_bin_location_dir=spmf_bin_location_dir)
        else:
            print('test')
            self.lst_patterns=lst_patterns

    def pattern_distance(self,clusters):
        return davies_bouldin_score(self.X,clusters)
        
    def pattern_usage(self,candidate_traceset):

        global_sum=sum([elt[1] for elt in self.lst_patterns])
        candidate_sum=0
        
        for pattern in self.lst_patterns:
            for trace in candidate_traceset:
                if pattern_in_sentence(pattern[0],trace):
                    candidate_sum+=pattern[1]
                    break
        return candidate_sum/global_sum
