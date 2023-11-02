from utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray
import json
def split_dataset(datapath,dataset_name):
    traceset=load_traceset(datapath,dataset_name)
    
    l=len(traceset)
    train_split=int(l*0.8)
    test_split=int(l*0.9)
    
    with open(datapath+"/dataset/"+dataset_name+"_train.txt", 'w') as f:
            for trace in traceset[:train_split]:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    with open(datapath+"/dataset/"+dataset_name+"_valid.txt", 'w') as f:
            for trace in traceset[train_split:test_split]:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    with open(datapath+"/dataset/"+dataset_name+"_test.txt", 'w') as f:
            for trace in traceset[test_split:]:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    with open(datapath+"/dataset/"+dataset_name+"_full.txt","w") as f:
            for trace in traceset:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')

def load_config_dict(filepath='./config/config.json'):
    with open(filepath) as json_file:
        config_dict = dict(json.load(json_file))
    return config_dict

config_dict=load_config_dict()
config_dict_exp=config_dict['experiments']
for dataset_name,v in config_dict_exp.items():
    filepath=config_dict_exp[dataset_name]['datapath']
    split_dataset(filepath,dataset_name)