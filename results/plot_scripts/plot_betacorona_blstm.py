#!/usr/bin/env python
"""
BLSTM model with FCN layer. 
The ESM model is utilized to embed the real world RBD datasets, which is 
then used in the BLSTM model. The last hidden states of the model is then
saved into a pickle file, which is utilized for clustering analysis and 
also a classification model. 
"""

import os
import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
from typing import Union
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

class RBDVariantDataset(Dataset):
    """ Real world RBD sequence dataset, embedded with ESM model. """

    def __init__(self, csv_file:str, device:str):
        self.df = pd.read_csv(csv_file, sep=',', header=0)
        self.device = device

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]["seq_id"], self.df.iloc[idx]["variant"], self.df.iloc[idx]["host"], self.df.iloc[idx]["sequence"]

class BLSTM(nn.Module):
    """ Bidirectional LSTM. """

    def __init__(self,
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size,    # The number of features in hidden layer of CN.
                 device):            # Device ('cpu' or 'cuda')
        super().__init__()
        self.device = device

        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)           

        # FCN
        if lstm_bidirectional:
            self.fcn = nn.Sequential(nn.Linear(2 * lstm_hidden_size, fcn_hidden_size),
                                     nn.ReLU())
        else:
            self.fcn = nn.Sequential(nn.Linear(lstm_hidden_size, fcn_hidden_size),
                                     nn.ReLU())

        # FCN output layer, for prediction (singular value)
        self.out = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n.detach()
        c_n.detach()
        lstm_final_out = lstm_out[:, -1, :]
        lstm_final_state = lstm_final_out.to(self.device)
        fcn_out = self.fcn(lstm_final_state)
        #prediction = self.out(fcn_out)

        return fcn_out

def run_model(model, tokenizer, embedder, data_loader, batch_size, device, save_as):
    """
    Call the BLSTM model to generate hidden states.

    Parameters:
    - model: BLSTM model
    - tokenizer: AutoTokenizer from Transformers, used for ESM model
    - embedder: ESM model from Transformers
    - data_loader: data loader of the dataset being embedded
    - batch_size: number of items per batch
    - device: cuda or cpu, currently set for cuda
    - save_as: what to save the pickle file as
    """
    model.eval()

    # Set the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          total = len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    all_seq_ids, all_hosts, all_variants, all_embeddings = [], [], [], []

    for i, batch_data in data_iter:
        seq_ids, hosts, variants, seqs = batch_data

        # Add 2 to max_length to account for additional tokens added to beginning and end by ESM
        tokenized_seqs = tokenizer(seqs,return_tensors="pt", padding="max_length", truncation=True, max_length=225).to(device)
        embedder.eval()

        with torch.no_grad():
            esm_hidden_states = embedder(**tokenized_seqs).last_hidden_state
            blstm_hidden_states = model(esm_hidden_states)
            
            for seq_id, host, variant, embedding in zip(seq_ids, hosts, variants, blstm_hidden_states):
                all_seq_ids.append(seq_id)
                all_hosts.append(host)
                all_variants.append(variant)
                all_embeddings.append(embedding.cpu().numpy())

    embedding_matrix = np.vstack(all_embeddings)
    print(embedding_matrix.shape)

    # Save data to a pickle file
    with open(save_as, 'wb') as f:
        pickle.dump((all_seq_ids, all_hosts, all_variants, all_embeddings), f)

if __name__=='__main__':

    # Set device
    device = torch.device("cpu")
    batch_size = 64

    # Data file
    full_csv_file = "/data/spike_ml/valerie/Spike_NLP/data/betacoronavirus_seq.csv"
    full_seq_dataset = RBDVariantDataset(full_csv_file, device)
    torch.manual_seed(0)
    full_seq_loader = DataLoader(full_seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # ESM embedding setup
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    embedder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)

    # BLSTM Model setup
    lstm_input_size = 320     
    lstm_hidden_size = 320
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 320
    model = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size, device).to(device)

    save_as = full_csv_file.replace(".csv", "betacoronavirus.pkl")
    run_model(model, tokenizer, embedder, full_seq_loader, batch_size, device, save_as)
