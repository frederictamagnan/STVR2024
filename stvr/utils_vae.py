import os
import joblib
from .text_autoencoders.train import main
from .text_autoencoders.vocab import Vocab
from .text_autoencoders.model import *
from .text_autoencoders.utils import *
from .text_autoencoders.batchify import get_batches
from .text_autoencoders.train import evaluate

def split_dataset(traceset,dataset_name):
    l=len(traceset)
    train_split=int(l*0.8)
    test_split=int(l*0.9)
    
    with open("./data/"+dataset_name+"_train.txt", 'w') as f:
            for trace in traceset[:train_split]:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    with open("./data/"+dataset_name+"_valid.txt", 'w') as f:
            for trace in traceset[train_split:test_split]:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    with open("./data/"+dataset_name+"_test.txt", 'w') as f:
            for trace in traceset[test_split:]:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    with open("./data/"+dataset_name+"_full.txt", 'w') as f:
            for trace in traceset:
                line=" ".join([ev.action for ev in trace.events])
                f.write(line)
                f.write('\n')
    
class Hyperparameters:
    def __init__(self,dataset_name,arch,datapath):
        self.arch=arch
        self.dataset_name = dataset_name
        self.train = datapath+"/datasets/" + self.dataset_name + "_train.txt"
        self.valid = datapath+"/datasets/" + self.dataset_name + "_valid.txt"
        self.save_dir = datapath + "checkpoints/" + self.dataset_name + "/" + self.arch
        self.datapath=datapath
        self.vocab_size=200
        self.nlayers=1
        self.lambda_kl=0
        self.lambda_adv=0
        self.lambda_p=0
        self.noise=[0,0,0,0]
        self.dropout=0.3
        self.epochs=15
        self.batch_size=256
        if self.arch=="AE":
            self.model_type="dae"
        elif self.arch=="VAE":
            self.model_type="vae"
            self.lambda_kl=0.1
        elif self.arch=="AAE":
            self.model_type="aae"
            self.lambda_adv=10
        elif self.arch=="LAAE":
            self.model_type="aae"
            self.lambda_adv=10
            self.lambda_p=0.01    
        elif self.arch=="DAAE":
            self.model_type="aae"
            self.lambda_adv=10
            self.noise=[0.3,0,0,0]
        self.seed=1111
        self.no_cuda="store_true"
        self.load_model=None
        self.log_interval=100

class InferenceHyperparameters:
     def __init__(self,hp,dataset_name,arch) -> None:
            self.hp=hp
            self.dataset_name=dataset_name
            self.arch=arch
    
            for attr in dir(self.hp):
                if not callable(getattr(self.hp, attr)) and not attr.startswith("__"):
                    setattr(self, attr, getattr(self.hp, attr))
            study = joblib.load("./data/studies/study_"+dataset_name+"_"+arch+".pkl")
            params=study.best_trial.params
            for key, value in params.items():
                setattr(self, key, value)
            
            if hasattr(self, 'learning_rate'):
                self.lr=self.learning_rate

            self.checkpoint = self.datapath + "checkpoints/" + self.dataset_name + "/" + self.arch
            self.model_name='model_trial_'+str(study.best_trial.number)+'.pt'
            self.data=self.datapath+"/datasets/"+self.dataset_name+"_full.txt"
            # self.checkpoint="./models/"+"checkpoints/"+self.name+"/"+self.arch
            self.sample=10
            self.output="sample"  
            self.n=10  
            self.load_model=""

            self.max_len=30
            self.dec="sample"
            self.enc='mu'
            self.m=100


def get_model(path,args,vocab,device):
    ckpt = torch.load(path)
    train_args = args
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

def encode(sents,args,vocab,device,model):    
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        if args.enc == 'mu':
            zi = mu
        else:
            zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def main_test(dataset_name,arch):
    hp=Hyperparameters(datapath='./data/',dataset_name=dataset_name,arch=arch)

    args=InferenceHyperparameters(hp,dataset_name,arch)
    vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, args.model_name),args,vocab,device)

    sents = load_sent(args.data)
    z = encode(sents,args,vocab,device,model)
    return z