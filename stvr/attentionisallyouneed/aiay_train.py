from torch import nn, Tensor
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import sys

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

def train(model: nn.Module,train_data,len_sentences,bptt,optimizer,scheduler,criterion,ntokens,device,epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    print("len train data",train_data.size(0))
    num_batches = len(train_data) // len_sentences
    print("rest",len(train_data)%len_sentences)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, len_sentences)):
        data, targets = get_batch(train_data, i,bptt)
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch
            print("WARNING")
            sys.exit(1)
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.5f} | ppl {ppl:8.5f}')
            total_loss = 0

def evaluate(model, eval_data,bptt,device,len_sentences,ntokens,criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, len_sentences):
            data, targets = get_batch(eval_data, i,bptt)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)