from attentionisallyouneed_v2.aiay_pre import load_dataset,TracesetDataset
from attentionisallyouneed_v2.aiay_model import TransformerModel
from attentionisallyouneed_v2.aiay_train import train,evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import optuna
import numpy as np
import joblib

import copy
import torch
import time
import math

class TrainingHandler:
    def __init__(self,dataset_name,datapath) -> None:
        self.dataset_name=dataset_name
        self.datapath=datapath
        max_length=32
        full_set,len_sentences=load_dataset(dataset_name,max_length=32,split=False)
        training_set,validation_set,len_sentences=load_dataset(dataset_name,max_length=32,split=True)
        # Example usage
        

        full_set = TracesetDataset(full_set)
        training_set = TracesetDataset(training_set)
        validation_set = TracesetDataset(validation_set)
        print(len(validation_set))
        self.vocab=full_set.build_vocab()
        # print(self.vocab.get_stoi())
        self.ntokens = len(self.vocab) 

        batch_size = 32

        self.bptt=batch_size


        validation_split = .2
        shuffle_dataset = True
        random_seed= 42

        dataset_size = len(full_set)


        training_set.build_vocab()
        validation_set.build_vocab()
        self.train_loader=DataLoader(training_set,batch_size=batch_size)
        self.val_loader=DataLoader(validation_set,batch_size=batch_size)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pad_idx=self.vocab.vocab.get_stoi()['<pad>']


    def define_model(self,trial):
        emsize = trial.suggest_int("emsize",16,64,step=16)    # embedding dimension
        d_hid = trial.suggest_int("d_hid",64,256,step=64)    # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = trial.suggest_int("nlayers",1,3,step=1)   # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        # nhead = trial.suggest_int("nhead",1,2,step=1)    # number of heads in nn.MultiheadAttention
        nhead=trial.suggest_categorical("nhead",[1,2,4])# number of heads in nn.MultiheadAttention
        dropout = trial.suggest_float("dropout", 0.1, 0.5)  # dropout probability
        self.model = TransformerModel(self.pad_idx,self.ntokens, emsize, nhead, d_hid, nlayers, dropout).to(self.device)
    
    def objective(self,trial):
        try:
            self.define_model(trial)
            print("Trial Number:", trial.number)
            print("Params:", trial.params)
            return self.train_aiyn(trial)
        except RuntimeError as e:
            print("error")
            raise optuna.TrialPruned()
    
    def train_aiyn(self,trial):
 
        self.ntokens = len(self.vocab)  # size of vocabulary

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        lr = 2.5 # learning rate
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        best_val_loss = float('inf')
        epochs = 30
        best_model = None
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("count",count_parameters(self.model))
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            loss=train(self.model,self.train_loader,self.bptt,optimizer,scheduler,criterion,self.ntokens,self.device,epoch,self.vocab)
            val_loss = evaluate(self.model, self.val_loader,self.bptt,self.device,None,self.ntokens,criterion)
            trial.report(val_loss,epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                torch.save(best_model.state_dict(), "./data/models/"+self.dataset_name+"_transformers_optuna"+str(trial.number)+".pt")

            scheduler.step()
        return val_loss
    
    def main_optuna(self):
        
        self.study = optuna.create_study(directions=["minimize"],pruner=optuna.pruners.MedianPruner())
        # self.study = optuna.create_study(directions=["minimize"])
        self.study.optimize(self.objective, n_trials=30)
        joblib.dump(self.study, "./data/studies/study_"+self.dataset_name+"_transformers.pkl")
        print("Number of finished trials: ", len(self.study.trials))


if __name__=='__main__':
    datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
    for dataset_name in datasets:
        th=TrainingHandler(dataset_name,"./data/")
      
        th.main_optuna()

