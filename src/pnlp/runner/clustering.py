#!/usr/bin/env python
"""
Clustering analysis

This script utilize the BERT model with best pretraining parameters to generate
the hidden states of all the sequences of SARS-CoV-2 RBD sequences, which will be
used for clustering and evolution studies.
"""

import os
import sys
import time
import datetime
import random
import pickle
from datetime import date
from typing import Union
import logging
import numpy as np
import tqdm
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT

logger = logging.getLogger(__name__)

class BERT_Runner:
    """ BERT model runner """

    def __init__(self,
                 run_dir: str,
                 csv_name: str,
                 parameter_file: str,         # pth file from best training model.
                 vocab_size:int,
                 embedding_dim:int,           # BERT parameters. Should come from best training model.
                 dropout: float,              # BERT parameters. Should come from best training model.
                 max_len: int,                # BERT parameters. Should come from best training model.
                 mask_prob: float,            # BERT parameters. Should come from best training model.
                 n_transformer_layers:int,    # BERT parameters. Should come from best training model.
                 n_attn_heads: int,           # BERT parameters. Should come from best training model.
                 batch_size: int,             # Learning parameters
                 device: str='cpu'):

        self.parameter_file = parameter_file
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.n_transformer_layers = n_transformer_layers
        self.n_attn_heads = n_attn_heads
        self.batch_size = batch_size
        self.device = device

        # Initialize model
        # Only BERT model is used to get the hidden state
        self.tokenizer = ProteinTokenizer(max_len, mask_prob)
        self.model = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
        self.model.to(self.device)

        # Create the file name
        self.save_as = csv_name.replace(".csv", "_cluster_320_embedding.pkl")

    def load_parameters(self):
        """
        Load parameters from the best pre-training model.
        Note that only parameters up to the BERT model is needed. Those for the language model are not needed.
        """
        saved_state = torch.load(self.parameter_file, map_location=self.device)
        saved_hyperparameters = saved_state['hyperparameters']
        state_dict = saved_state['model_state_dict']

        if saved_hyperparameters["device"] != self.device:
            msg = f'Map saved status from {saved_hyperparameters["device"]} to {self.device}.'
            logger.warning(msg)

        hyperparameters = ['vocab_size','embedding_dim','dropout','max_len','mask_prob',
                           'n_transformer_layers','n_attn_heads','batch_size']

        for hp in hyperparameters:
            assert getattr(self, hp) == saved_hyperparameters[hp], f"{hp} mismatch"

        # For loading from ddp models, need to remove 'module.bert.' in state_dict
        if 'ddp' in self.parameter_file:
            prefix = 'module.bert.'
            state_dict = {i[len(prefix):]: j for i, j in state_dict.items() if prefix in i[:len(prefix)]}

        self.model.load_state_dict(state_dict)
        random.setstate(saved_state['random_state'])

    def run(self, seq_data:Dataset, max_batch: Union[int, None]):
        """
        Call the BERT model to generate hidden states.

        Parameters:
        seq_data: a dataloader feeding protein sequences.
        max_batch: maximum number of batches to run. If not defined, all avaiable batches from train_data will be used.
        """
        if not max_batch:
            max_batch = len(seq_data)

        logger.info(f"Loading saved pth file: {self.parameter_file}")
        self.load_parameters()

        logger.info("Running BERT")
        start_time = time.time()
        self.model.eval()

        # Set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(seq_data),
                              total = len(seq_data),
                              bar_format='{l_bar}{r_bar}')

        all_seq_ids, all_variants, all_embeddings = [], [], []

        for i, batch_data in data_iter:
            if max_batch > 0 and i >= max_batch:
                break

            seq_ids, variants, seqs = batch_data
            tokenized_seqs = self.tokenizer(seqs)
            tokenized_seqs = tokenized_seqs.to(self.device)  # input tokens with masks

            with torch.no_grad():
                hidden_states = self.model(tokenized_seqs)
                embeddings = hidden_states.cpu().numpy()
                
                for seq_id, variant, embedding in zip(seq_ids, variants, embeddings):
                    if embedding.shape == (223, 320):
                        all_seq_ids.append(seq_id)
                        all_variants.append(variant)
                        all_embeddings.append(embedding)

        # Save data to a pickle file
        with open(self.save_as, 'wb') as f:
            pickle.dump((all_seq_ids, all_variants, all_embeddings), f)

class VariantSeqDataset(Dataset):

    def __init__(self, csv_file:str):
        self.df = pd.read_csv(csv_file, sep=',', header=0)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return [self.df.iloc[idx]["seq_id"],
                self.df.iloc[idx]["variant"],
                self.df.iloc[idx]["sequence"]]

if __name__=="__main__":

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(os.path.dirname(__file__), f'../../../results/clustering/clustering-{date_hour_minute}')
    os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    log_file = os.path.join(run_dir, f'clustering_{date_hour_minute}.log')
    logging.basicConfig(
        filename = log_file,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p')
    logger.setLevel(logging.DEBUG)  # To disable the matplotlib font_manager logs.

    # Dataset
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data/spike_variants')
    csv_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants.csv")
    seq_dataset = VariantSeqDataset(csv_file)
    
    # -= HYPERPARAMETERS =-
    embedding_dim = 320 # 768
    dropout=0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 10 # 12
    hidden = embedding_dim

    batch_size = 64
    n_test_batches = -1

    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len, dropout)
    vocab_size = len(token_to_index)

    USE_GPU = True
    device = torch.device("cuda:1" if torch.cuda.is_available() and USE_GPU else "cpu")

    torch.manual_seed(0) # Dataloader uses its own random number generator.
    seq_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    # best_pth = "../../../results/ddp_runner/ddp-2023-08-16_08-41/ddp-2023-08-16_08-41_best_model_weights.pth" # 768
    best_pth = "../../../results/ddp_runner/ddp-2023-10-06_20-16/ddp-2023-10-06_20-16_best_model_weights.pth" # 320
    parameter_file = os.path.join(os.path.dirname(__file__), best_pth)  # pick the pth with the best accuracy.

    runner = BERT_Runner(run_dir=run_dir, csv_name=csv_file, parameter_file=parameter_file, vocab_size=vocab_size, 
                         embedding_dim=embedding_dim, dropout=dropout, max_len=max_len, 
                         mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                         n_attn_heads=attn_heads, batch_size=batch_size, device=device)

    logger.info(f'Run log file located in this directory: {run_dir}')
    logger.info(f'Using device: {device}')
    logger.info(f'Total seqs: {len(seq_dataset)}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'embedding_dim: {embedding_dim}')
    logger.info(f'dropout: {dropout}')
    logger.info(f'max_len: {max_len}')
    logger.info(f'mask_prob: {mask_prob}')
    logger.info(f'n_transformer_layers: {n_transformer_layers}')
    logger.info(f'n_attn_heads: {attn_heads}')
    logger.info(f'hidden: {hidden}')

    runner.run(seq_loader, n_test_batches)