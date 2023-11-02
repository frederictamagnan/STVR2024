from text_autoencoders.train import main
import json
import optuna
import joblib

class hp():
    def __init__(dataset_name,arch):
        self.arch=arch
        self.dataset_name = dataset_name
        self.train = datapath + self.dataset_name + "_train.txt"
        self.valid = datapath + self.dataset_name + "_valid.txt"
        self.save_dir = rootpath + "checkpoints/" + self.dataset_name + "/" + self.arch
        self.vocab_size=10000
        self.dim_z=trial.suggest_int("dim_emb",32,128)
        self.dim_emb=trial.suggest_int("dim_emb",64,256)
        self.dim_h=trial.suggest_int("dim_h",256,1024)
        self.nlayers=1
        self.dim_d=trial.suggest_int("dim_h",64,256)
        self.lambda_kl=0
        self.lambda_adv=0
        self.lampda_p=0
        self.noise=[0,0,0,0]
        self.dropout=0.5
        self.lr=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        self.epochs=50
        self.batch_size=256
        if self.arch="AE":
            self.model_type="dae"
        elif self.arch="VAE":
            self.model_type="vae"
            self.lambda_kl=0.1
        elif self.arch="DAAE":
            self.lambda_adv=10
            self.noise=[0.3,0,0,0]

        



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


def define_model(trial):
    hpt=HyperparametersTraining("vae_scanette","./data/datasets/","./data/",load_vae_dict()['vae'])
    hpt.nlayers=trial.suggest_int("nlayers",1 , 3)
    hpt.dim_emb=trial.suggest_int("dim_emb",64,256)
    hpt.lr=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    return hpt
def objective(trial):
    hpt=define_model(trial)
    return main(hpt)

study = optuna.create_study(directions=["minimize"])
study.optimize(objective, n_trials=10, timeout=300)
joblib.dump(study, "./data/studies/study.pkl")
print("Number of finished trials: ", len(study.trials))
# hpt=HyperparametersTraining("vae_scanette","./data/datasets/","./data/",load_vae_dict()['vae'])
# main(hpt)
