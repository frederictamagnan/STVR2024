import json
from agilkia import TraceSet
from sklearn.feature_extraction.text import CountVectorizer
from itertools import groupby
from spmf import Spmf
import numpy as np
import json
from os.path import exists
def load_from_dict(cls,data) -> 'TraceSet':
    """Load traces from the given file.
    This upgrades older trace sets to the current version if possible.
    """
    if isinstance(data, dict) and data.get("__class__", None) == "TraceSet":
        return cls.upgrade_json_data(data)
    else:
        raise Exception("unknown JSON file format: " + str(data)[0:60])

def load_traceset(datapath,dataset_name):


    with open(datapath+dataset_name+'.json') as json_file:
        data = json.load(json_file)

    traceset=load_from_dict(TraceSet,data)
    return traceset

def traceset_to_textset(traceset,start_end_token_creator=lambda i:('<sos>','<eos>'),format='str'):
    textset=[]
    for i,tr in enumerate(traceset):
        sos,eos=start_end_token_creator(i)
        if format=='str':
            textset.append(sos+' '+' '.join([ev.action for ev in tr.events])+' '+eos)
        elif format=='lst':
            textset.append([sos]+[ev.action for ev in tr.events]+[eos])
        else:
            raise ValueError('not implemented')
    return textset

def textset_to_bowarray(textset,vocabulary_provided=None):
    if vocabulary_provided is None:
        count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
        bowarray=count_vect_actions.fit_transform(textset)
        return bowarray.toarray(),count_vect_actions.vocabulary_
    else:
        count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
        count_vect_actions.vocabulary_=vocabulary_provided
        bowarray=count_vect_actions.transform(textset)
        return bowarray.toarray(),count_vect_actions.vocabulary_

def textset_to_one_hot(textset):
    count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
    bowarray=count_vect_actions.fit_transform(textset)
    one_hot=bowarray!=0
    return one_hot.toarray().astype(int),count_vect_actions.vocabulary_

def get_a_session(i,traceset_global):
    return [ev.action for ev in traceset_global[i].events]

def traceset_to_pattern_one_hot(traceset,dataset_name,filepath="./data/",freq=0.1,spmf_bin_location_dir="./stvr/"):
    textset=traceset_to_textset(traceset,format='str')
    # print(textset)
    filename="spmf_dataset_v3_"+dataset_name+"_"+str(freq)
    voc=textset_to_spmf(textset,filepath=filepath,filename=filename)
    spmf = Spmf("ClaSP", input_filename=filepath+filename,
            output_filename=filepath+filename+".txt", arguments=[freq,False],spmf_bin_location_dir=spmf_bin_location_dir)

    spmf.run()
   
    df=spmf.to_pandas_dataframe(pickle=True)
    voc_=inv_map = {v: k for k, v in voc.items()}
    # print(df)
    lst_patterns=[]
    for index, row in df.iterrows():
        lst_patterns.append((id_to_words(row['pattern'],voc_), row['sup']))
    
    
    textset=traceset_to_textset(traceset,format='lst')
    lst_encoding=[]
    for i,trace in enumerate(textset):
        lst_encoding.append([0 for elt in lst_patterns])
        for j,pattern in enumerate(lst_patterns):
            if pattern_in_sentence(pattern[0],trace):
                lst_encoding[i][j]=1
    filename_npy="spmf_one_hot_"+dataset_name+"_"+str(freq)+".npy"
    filename_lst="spmf_lst_patterns_"+dataset_name+"_"+str(freq)+".json"

    d={}
    d['lst_patterns']=lst_patterns
    arr = np.array(lst_encoding)
    np.save(filepath+filename_npy,arr)
    with open(filepath+filename_lst, "w") as outfile:
        json.dump(d, outfile)
    return arr,lst_patterns

def intersperse(lst, item):
        result = [item] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result

def merge_sentence(sentence):
    result = []
    for key, group in groupby(sentence):
        count = len(list(group))
        if count == 1:
            result.append(key)
        else:
            result.append(key)
    return result


def load_spmf_files(filepath,traceset,dataset_name,freq):
    filename_npy="spmf_one_hot_"+dataset_name+"_"+str(freq)+".npy"
    filename_lst="spmf_lst_patterns_"+dataset_name+"_"+str(freq)+".json"
    if exists(filepath+filename_npy) and exists(filepath+filename_lst):
        spmf_one_hot=np.load(filepath+filename_npy)
        with open(filepath+filename_lst) as json_file:
            spmf_lst_patterns_dict = dict(json.load(json_file))
        spmf_lst_patterns=spmf_lst_patterns_dict['lst_patterns']
    
    else:
        spmf_one_hot,spmf_lst_patterns=traceset_to_pattern_one_hot(traceset,dataset_name,filepath=filepath,freq=freq,spmf_bin_location_dir="./stvr/")
        
    return spmf_one_hot,spmf_lst_patterns

def textset_to_spmf(textset,filepath=None,filename="_spmf_dataset.txt"):
    count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
    count_vect_actions.fit(textset)
    
    idset=[[count_vect_actions.vocabulary_[word] for word in sentence.split(' ')] for sentence in textset]
    idset=[merge_sentence(sentence) for sentence in idset]
    idset=[intersperse(sentence,-1)+[-1,-2] for sentence in idset]
    idset=[[str(word) for word in sentence] for sentence in idset]
    
    if filepath is None:
        filepath="./data/"
    with open(filepath+filename, 'w') as f:
        for line in idset:
            f.write(" ".join(line))
            f.write('\n')
    return count_vect_actions.vocabulary_
    
def id_to_words(sentence,voc):
    s=[]
    # for word in sentence:
    #     s+=[word.split(" ")]
    p=[voc[int(word)] for word in sentence]
    
    return p

def pattern_in_sentence(pattern,sentence):
    small_list=pattern
    big_list=sentence
    i=0
    for elt in big_list:
        if small_list[i]==elt:
            i=i+1
            
            if i>=len(small_list):
                break
    if i==len(small_list):
        return True
    else:
        return False