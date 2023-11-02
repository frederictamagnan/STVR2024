from text_autoencoders.train import main
import json
class HyperparametersTraining:
    def __init__(self,model_name,datapath,rootpath,h_dict):
        self.arch = model_name
        self.div = 10
        self.name = "scanette_100043-steps"
        self.train = datapath + self.name + "_train.txt"
        self.valid = datapath + self.name + "_valid.txt"
        self.save_dir = rootpath + "checkpoints_test/" + self.name + "/" + self.arch
        
        for key, value in h_dict.items():
            setattr(self, key, value)

def load_vae_dict(filepath='./config/config_vae.json'):
    with open(filepath) as json_file:
        config_dict = dict(json.load(json_file))
    return config_dict

hpt=HyperparametersTraining("vae_scanette","./data/datasets/","./data/",load_vae_dict()['vae'])
main(hpt)