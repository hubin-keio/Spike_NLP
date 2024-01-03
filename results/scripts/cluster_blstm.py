#!/usr/bin/env python
"""
BLSTM model with FCN layer. 
The ESM model is utilized to embed the datasets, which is 
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
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index
from pnlp.model.language import ProteinMaskedLanguageModel, BERT

class RBDVariantDataset(Dataset):
    """ Real world RBD sequence dataset, embedded with ESM model. """

    def __init__(self, csv_file:str, device:str):
        self.df = pd.read_csv(csv_file, sep=',', header=0)
        self.device = device

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]['seq_id'], self.df.iloc[idx]['variant'], self.df.iloc[idx]['sequence']

class AlphaSeqDataset(Dataset):
    """ AlphaSeq sequence dataset, embedded with ESM model. """

    def __init__(self, csv_file:str, device:str):
        self.df = pd.read_csv(csv_file, sep=',', header=0)
        self.device = device

        self.df = self.df[['POI', 'Sequence', 'Mean_Affinity']]
        self.df['variant'] = pd.qcut(self.df['Mean_Affinity'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4']) # Quantile based on mean affinity
        self.df = self.df.rename(columns={'POI':'seq_id', 'Sequence':'sequence'})

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]['seq_id'], self.df.iloc[idx]['variant'], self.df.iloc[idx]['sequence']

class BLSTM(nn.Module):
    """ Bidirectional LSTM. Output is embedding layer, not prediction value."""

    def __init__(self,
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size):   # The number of features in hidden layer of CN.
        super().__init__()

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

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n.detach()
        c_n.detach()
        lstm_final_out = lstm_out[:, -1, :]
        lstm_final_state = lstm_final_out.to(x.device)
        fcn_out = self.fcn(lstm_final_state)

        return fcn_out

class BERT_BLSTM(nn.Module):
    """" BERT protein language model. """
    
    def __init__(self, bert: BERT, blstm:BLSTM, vocab_size: int):
        super().__init__()

        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)
        self.blstm = blstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)
        error_1 = self.mlm(x) # error from masked language
        error_2 = self.blstm(x) # error from regession

        return error_2

def run_bert_model(model, tokenizer, data_loader, batch_size, device, save_as):
    """
    Call the BERT-BLSTM model to generate hidden states.

    Parameters:
    - model: BERT-BLSTM model
    - tokenizer: AutoTokenizer from Transformers, used for ESM model
    - data_loader: data loader of the dataset being embedded
    - batch_size: number of items per batch
    - device: cuda or cpu, currently set for cuda
    - save_as: what to save the pickle file as
    """
    model = model.to(device)
    model.eval()

    # Set the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          total = len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    all_seq_ids, all_variants, all_embeddings = [], [], []

    for i, batch_data in data_iter:
        seq_ids, variants, seqs = batch_data

        tokenized_seqs = tokenizer(seqs).to(device)

        with torch.no_grad():
            blstm_hidden_states = model(tokenized_seqs)
            print(blstm_hidden_states.shape)
            
            for seq_id, variant, embedding in zip(seq_ids, variants, blstm_hidden_states):
                all_seq_ids.append(seq_id)
                all_variants.append(variant)
                all_embeddings.append(embedding.cpu().numpy())

    embedding_matrix = np.vstack(all_embeddings)
    print(embedding_matrix.shape)

    # Save data to a pickle file
    with open(save_as, 'wb') as f:
        pickle.dump((all_seq_ids, all_variants, all_embeddings), f)


def run_esm_model(model, tokenizer, embedder, data_loader, batch_size, device, save_as):
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
    model = model.to(device)
    model.eval()

    # Set the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          total = len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    all_seq_ids, all_variants, all_embeddings = [], [], []

    for i, batch_data in data_iter:
        seq_ids, variants, seqs = batch_data

        # Add 2 to max_length to account for additional tokens added to beginning and end by ESM
        # 225 for RBD, 251 for AlphaSeq
        tokenized_seqs = tokenizer(seqs,return_tensors='pt', padding='max_length', truncation=True, max_length=225).to(device) 
        embedder.eval()

        with torch.no_grad():
            esm_hidden_states = embedder(**tokenized_seqs).last_hidden_state
            blstm_hidden_states = model(esm_hidden_states)
            print(blstm_hidden_states.shape)
            
            for seq_id, variant, embedding in zip(seq_ids, variants, blstm_hidden_states):
                all_seq_ids.append(seq_id)
                all_variants.append(variant)
                all_embeddings.append(embedding.cpu().numpy())

    embedding_matrix = np.vstack(all_embeddings)
    #print(embedding_matrix.shape)

    # Save data to a pickle file
    with open(save_as, 'wb') as f:
        pickle.dump((all_seq_ids, all_variants, all_embeddings), f)

if __name__=='__main__':

    # Set device
    device = torch.device('cuda:0')
    batch_size = 64

    # Data file
    data_dir = os.path.join(os.path.dirname(__file__), f'../data')
    full_csv_file = os.path.join(data_dir, 'spikeprot0528.clean.uniq.noX.RBD_variants.csv')
    full_seq_dataset = RBDVariantDataset(full_csv_file, device)
    # full_csv_file = os.path.join(data_dir, 'clean_avg_alpha_seq.csv')
    # full_seq_dataset = AlphaSeqDataset(full_csv_file, device)

    torch.manual_seed(0)
    full_seq_loader = DataLoader(full_seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # == BERT BLSTM CLUSTERING ==
    # Load saved model BERT-BLSTM
    results_dir = os.path.join(os.path.dirname(__file__), f'../results/run_results/bert_blstm')
    # model_pth = os.path.join(results_dir, 'bert_blstm-dms_binding-2023-11-22_23-03/bert_blstm-dms_binding-2023-11-22_23-03_train_84420_test_21105.model_save')
    model_pth = os.path.join(results_dir, 'bert_blstm-dms_expression-2023-11-22_23-05/bert_blstm-dms_expression-2023-11-22_23-05_train_93005_test_23252.model_save')
    saved_state = torch.load(model_pth, map_location='cuda')
    state_dict = saved_state['model_state_dict']

    # For loading from ddp models, they have 'module.bert.' or 'module.mlm.' in keys of state_dict
    # Also need separated out for each corresponding model part
    bert_state_dict = {key[len('bert.'):]: value for key, value in state_dict.items() if key.startswith('bert.')}
    mlm_state_dict = {key[len('mlm.'):]: value for key, value in state_dict.items() if key.startswith('mlm.')}
    blstm_state_dict = {key[len('blstm.'):]: value for key, value in state_dict.items() if key.startswith('blstm.')}

    # BERT input
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    embedding_file = os.path.join(results_dir, '../bert_blstm_esm/bert_blstm_esm-dms_expression-2023-12-21_21-01/esm_embeddings_320_dim.pth')
    bert.embedding.load_pretrained_embeddings(embedding_file, no_grad=False)
    #bert.load_state_dict(bert_state_dict)

    # BLSTM input
    lstm_input_size = 320
    lstm_hidden_size = 320
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = 320
    blstm = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size)
    #blstm.load_state_dict(blstm_state_dict)

    # BERT_BLSTM input
    vocab_size = len(token_to_index)
    model = BERT_BLSTM(bert, blstm, vocab_size)
    #model.mlm.load_state_dict(mlm_state_dict)

    base_filename = os.path.basename(full_csv_file).replace('.csv', '_clustering_bert_blstm_esm.pkl')
    save_as = os.path.join(f'{data_dir}/pickles', base_filename)
    run_bert_model(model, tokenizer, full_seq_loader, batch_size, device, save_as)

    # == ESM BLSTM CLUSTERING ==
    # # ESM embedding setup
    # tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    # embedder = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D').to(device)

    # base_filename = os.path.basename(full_csv_file).replace('.csv', '_clustering_esm_blstm.pkl')
    # save_as = os.path.join(f'{data_dir}/pickles', base_filename)
    # run_esm_model(blstm, tokenizer, embedder, full_seq_loader, batch_size, device, save_as)

   
