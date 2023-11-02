from typing import Any
from text_autoencoders.train import main
import json
import optuna
import joblib

from utils_vae import Hyperparameters

class TrainingHandler:
    def __init__(self,dataset_name,arch,datapath) -> None:
        self.hp=Hyperparameters(dataset_name,arch,datapath)


    def define_model(self,trial):
        # self.hp.dim_z=trial.suggest_int("dim_z",32,128)
        # self.hp.dim_emb=trial.suggest_int("dim_emb",64,256)
        # self.hp.dim_h=trial.suggest_int("dim_h",256,1024)
        # self.hp.dim_d=trial.suggest_int("dim_d",64,256)
        # self.hp.dim_z=trial.suggest_int("dim_z",32,128)
        self.hp.dim_z=trial.suggest_int("dim_z",32,128,step=32)
        self.hp.dim_emb=trial.suggest_int("dim_emb",64,512,step=56)
        self.hp.dim_h=trial.suggest_int("dim_h",128,1024,step=112)
        self.hp.dim_d=trial.suggest_int("dim_d",32,512,step=56)
        self.hp.lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        # self.hp.batch_size=trial.suggest_int("batch_size",64,256,step=64)
        # self.hp.epochs=trial.suggest_int("epochs",4,10)

        return self.hp
    
    def objective(self,trial):
        try:
            hp=self.define_model(trial)
            print("Trial Number:", trial.number)
            print("Params:", trial.params)
            return main(hp,trial)
        except RuntimeError as e:
            print("error")
            raise optuna.TrialPruned()
    
    
    def main_optuna(self):
        
        self.study = optuna.create_study(directions=["minimize"],pruner=optuna.pruners.MedianPruner())
        # self.study = optuna.create_study(directions=["minimize"])
        self.study.optimize(self.objective, n_trials=15)
        joblib.dump(self.study, "./data/studies/study_"+self.hp.dataset_name+"_"+self.hp.arch+".pkl")
        print("Number of finished trials: ", len(self.study.trials))

if __name__=="__main__":

    datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
    archs=["AE","VAE","DAAE"]
    # archs=["DAAE"]
    for arch in archs:
        for dataset_name in datasets:
            th=TrainingHandler(dataset_name,arch,"./data/")
            th.main_optuna()
