from torch import nn, Tensor
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import sys
import numpy as np
from torch.distributions.categorical import Categorical
import math
from pprint import pprint

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def get_batch(source, i,bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    # seq_len = min(bptt, len(source) - 1 - i)
    
    data = source[i:i+bptt]
    target = source[i+1:i+1+bptt].reshape(-1)
    return data, target

def train(model: nn.Module,dataloader,bptt,optimizer,scheduler,criterion,ntokens,device,epoch,vocab) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 50
    start_time = time.time()
    num_batches = len(dataloader)
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    
    for batch_idx, batch in enumerate(dataloader):
        
        source,target=batch
        source=source.transpose(0, 1)
        targets=target.transpose(0, 1).reshape(-1)
        batch_size = source.size(0)
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(source, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.5f} | ppl {ppl:8.5f}')
            total_loss = 0
            print(inference(model,bptt,1,30,vocab,ntokens,device))
    return cur_loss



def evaluate(model, eval_loader,bptt,device,len_sentences,ntokens,criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
         for batch_idx, batch in enumerate(eval_loader):
            source,target=batch
            source=source.transpose(0, 1)
            targets=target.transpose(0, 1).reshape(-1)
            batch_size = source.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(source, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_loader) - 1)

def get_embeddings(model, eval_loader,bptt,device,len_sentences,ntokens,criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    embeddings = []
    with torch.no_grad():
         for batch_idx, batch in enumerate(eval_loader):
            source,target=batch
            source=source.transpose(0, 1)
            targets=target.transpose(0, 1).reshape(-1)
            batch_size = source.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model.encode(source, src_mask)
            # print(output.size())
            embeddings.append(output.cpu().numpy())

    concatenated_embeddings = np.concatenate(embeddings, axis=1)


    return concatenated_embeddings.transpose(1,0,2)

def inference(model,bptt,nb_of_seq,len_max,vocab,ntokens,device):
    s=nn.Softmax(dim=0)
    model.eval()  # turn on evaluation mode

    src_mask = generate_square_subsequent_mask(bptt).to(device)
    list_seq=[]

    for j in range(nb_of_seq):

      curr_seq=""
      blank=[]
      blank.append(['<sos>'])
      for i in range(bptt-1):
        blank.append(['<pad>'])
      l=0
      next_token=None
      while next_token != '<eos>' and l<len_max:
          data = [torch.tensor(vocab(item), dtype=torch.long) for item in blank]
          data=torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
          data = torch.unsqueeze(data, dim=-1)
          data=data.to(device)
          # data = batchify(data, 1)
          with torch.no_grad():
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            # output_flat[i][0]=0
            proba=s(output_flat[l])

            m=Categorical(proba)
            sampled_id=m.sample()
            next_token=vocab.vocab.get_itos()[sampled_id]
          curr_seq+=" "+next_token
          blank[l+1]=[next_token]
          l+=1
      list_seq.append(curr_seq)
    # for s in list_seq:
    #   print(s)
    return list_seq