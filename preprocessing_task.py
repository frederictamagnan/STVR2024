from utils_experiments import load_config_dict
from stvr.utils_preprocessing import traceset_to_pattern_one_hot,load_traceset
import sys

def get_task():
    config_dict=load_config_dict()
    config_dict_exp=config_dict['experiments']
    i=1
    for dataset_name,v in config_dict_exp.items():
        
        config_dict_exp=config_dict['experiments']
        config_dict_hp=config_dict['hp']
        filepath=config_dict_exp[dataset_name]['datapath']
        freqs=config_dict_exp[dataset_name]['freq']
        traceset=load_traceset(filepath,dataset_name)
        for freq in freqs:
            # spmf_one_hot,spmf_lst_patterns=traceset_to_pattern_one_hot(traceset,dataset_name,filepath=filepath,freq=freq,spmf_bin_location_dir="./stvr/")
            yield traceset,dataset_name,filepath,freq,"./stvr/",i
            i+=1
def get_max():
    for elt in get_task():
        pass
    return list(elt)[-1]

def do_task(task_id):
    m=get_max()
    if task_id>m:
        return 0
    for elt in get_task():
        print(elt)
        if list(elt)[-1]==task_id:
            traceset,dataset_name,filepath,freq,_,i=elt
            spmf_one_hot,spmf_lst_patterns=traceset_to_pattern_one_hot(traceset,dataset_name,filepath=filepath,freq=freq,spmf_bin_location_dir="./stvr/")
            print(freq)
            return 0

task_id = int(sys.argv[1])
do_task(task_id)
