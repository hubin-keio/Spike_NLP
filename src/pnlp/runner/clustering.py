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
from datetime import date
from os import path
from typing import Union
import logging
import numpy as np
import tqdm
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT
from pnlp.plots import plot_run

logger = logging.getLogger(__name__)

class BERT_Runner:
    """
    BERT model runner
    """

    def __init__(self,
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
        now = datetime.datetime.now()
        date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
        save_path = os.path.join(os.path.dirname(__file__), f'../../../results/clustering_{date_hour_minute}')
        self.save_as = os.path.join(save_path, date_hour_minute)
        self.run_result_csv = ''.join([self.save_as, '_results.csv'])

    def load_parameters(self):
        """
        Load parameters from the best pre-training model.
        Note that only parameters up to the BERT model is needed. Those for the language model are not needed.

        """
        map_location = self.device
        saved_state = torch.load(self.parameter_file, map_location=map_location)
        saved_hyperparameters = saved_state['hyperparameters']
        if saved_hyperparameters["device"] != self.device:
            msg = f'Map saved status from {saved_hyperparameters["device"]} to {self.device}.'
            logger.warning(msg)

        assert self.vocab_size == saved_hyperparameters['vocab_size']
        assert self.embedding_dim == saved_hyperparameters['embedding_dim']
        assert self.dropout == saved_hyperparameters['dropout']
        assert self.max_len == saved_hyperparameters['max_len']
        assert self.mask_prob == saved_hyperparameters['mask_prob']
        assert self.n_transformer_layers == saved_hyperparameters['n_transformer_layers']
        assert self.n_attn_heads == saved_hyperparameters['n_attn_heads']
        assert self.batch_size == saved_hyperparameters['batch_size']

        # For loading from ddp models, they have 'module.bert.' in keys of state_dict
        # Need to remove that part to work
        state_dict = saved_state['model_state_dict']
        load_ddp = False
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k[:11] == 'module.bert':
                load_ddp = True
                new_state_dict[k[12:]] = v   # remove 'module.bert.'
            else: break
            
        if load_ddp: state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
        random.setstate(saved_state['random_state'])

    def run(self, seq_data:SeqDataset, max_batch: Union[int, None]):
        """
        Call the BERT model to generate hidden states.

        Parameters:
        seq_data: a dataloader feeding protein sequences.
        max_batch: maximum number of batches to run. If not defined, all avaiable batches from train_data will be used.
        """
        WRITE = False

        if not max_batch:
            max_batch = len(seq_data)

        logger.info(f"Loading saved pth file: {self.parameter_file}")
        self.load_parameters()

        logger.info("Running BERT")
        start_time = time.time()
        self.model.eval()

        MASK_TOKEN_IDX = token_to_index['<MASK>']

        # Set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(seq_data),
                              total = len(seq_data),
                              bar_format='{l_bar}{r_bar}')

        for i, batch_data in data_iter:
            if max_batch > 0 and i >= max_batch:
                break

            seq_ids, seqs = batch_data
            tokenized_seqs = self.tokenizer(seqs)
            tokenized_seqs = tokenized_seqs.to(self.device)  # input tokens with masks

            with torch.no_grad():
                hidden_status = self.model(tokenized_seqs) 
                print(hidden_status)

                # TODO:
                # Need to save seq_ids and self.model.hidden in a csv file. each entry starts with a seq_id.

        if WRITE:
            plot_run.plot_run(self.run_result_csv, save=True)
            logger.info(f'Run result saved to {os.path.basename(self.run_result_csv)}')

if __name__=="__main__":

    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(os.path.dirname(__file__), f'../../../results/clustering_{date_hour_minute}')
    os.makedirs(run_dir, exist_ok = True)

    # Add logging configuration
    log_file = os.path.join(run_dir, f'clustering_{date_hour_minute}.log')
    logging.basicConfig(
        filename = log_file,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p')
    logger.setLevel(logging.DEBUG)  # To disable the matplotlib font_manager logs.

    # Data loader
    db_file = path.abspath(path.dirname(__file__))
    db_file = path.join(db_file, '../../../data/SARS_CoV_2_spike_noX_RBD.db')
    seq_dataset = SeqDataset(db_file, "train")  # FIXEME: needs to include both training and testing

    # -= HYPERPARAMETERS =-
    embedding_dim = 768
    dropout=0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12
    hidden = embedding_dim

    batch_size = 64
    n_test_baches = 5

    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)
    vocab_size = len(token_to_index)

    USE_GPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")

    torch.manual_seed(0) # Dataloader uses its own random number generator.
    seq_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_pth = "../../../results/ddp-2023-08-16_08-41/ddp-2023-08-16_08-41_best_model_weights.pth"
    parameter_file = os.path.join(os.path.dirname(__file__), best_pth)  # pick the pth with the best accuracy.

    runner = BERT_Runner(parameter_file=parameter_file, vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout,
                         max_len=max_len, mask_prob=mask_prob, n_transformer_layers=n_transformer_layers,
                         n_attn_heads=attn_heads, batch_size=batch_size, device=device)

    logger.info(f'Run results located in this directory: {run_dir}')
    logger.info(f'Using device: {device}')
    logger.info(f'Data set: {os.path.basename(db_file)}')
    logger.info(f'Total seqs: {len(seq_dataset)}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'embedding_dim: {embedding_dim}')
    logger.info(f'dropout: {dropout}')
    logger.info(f'max_len: {max_len}')
    logger.info(f'mask_prob: {mask_prob}')
    logger.info(f'n_transformer_layers: {n_transformer_layers}')
    logger.info(f'n_attn_heads: {attn_heads}')
    logger.info(f'hidden: {hidden}')

    runner.run(seq_loader, n_test_baches)