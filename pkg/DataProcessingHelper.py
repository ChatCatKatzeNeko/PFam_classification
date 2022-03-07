# basic import
import pandas as pd
import numpy as np
import glob

# TAPE
from tape import TAPETokenizer

# Pytorch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch


def read_datafiles(path):
    '''
    read and concat all files under the given path
    '''
    fileList = glob.glob(path+'*')
    df = pd.DataFrame()
    for f in fileList:
        df = pd.concat([df,pd.read_csv(f)], axis=0, ignore_index=True)
    return df



# Create a function to tokenize a set of sequences
def prepare_data(sequences, maxLen):
    """
    Perform required preprocessing steps for pretrained BERT-base model.
    INPUT
    sequences (array-like): Array of sequences to be processed.
    tokenizer (TAPETokenizer obj): a TAPE tokenizer instance
    OUTPUTS
    tokenIdx (torch.Tensor): Tensor of token indices to be fed
    """
    # to be returned
    tokenIdx = np.zeros((len(sequences), maxLen+2))

    # Load the BERT-Protein tokenizer
    tokenizer = TAPETokenizer(vocab='iupac')

    # loop through all sequences
    for i,seq in enumerate(sequences):
        # (1) Tokenize the sentence
        # (2) Truncate/Pad sentence to MAX_LEN
        # (3) Map tokens to their IDs
        # (4) Return a tensor of outputs
        
        # (1)
        tokens = tokenizer.tokenize(seq)
        # (2)
        # turncation
        if len(tokens) > maxLen:
            tokens = tokens[:maxLen]
        # padding
        elif len(tokens) < maxLen:
            tokens += ["<pad>"] * (maxLen-len(tokens))
        else:
            pass
        # (3)
        tokens = tokenizer.add_special_tokens(tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # fill in the output
        tokenIdx[i,:] = token_ids
    
    # (4) Convert to tensor
    tokenIdx = torch.tensor(tokenIdx,dtype=int)

    return tokenIdx


def create_data_loader(inputs, labels, batchSize):
    """
    return a data loader
    """
    data = TensorDataset(inputs,labels)
    sampler = RandomSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batchSize)