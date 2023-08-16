#!/usr/bin/env python
""" BLSTM Model Utilizing Pretrained Model and DMS Dataset. """

import os
import re
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from collections import OrderedDict

from pnlp.db.dataset import SeqDataset, initialize_db
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT

class DMS_Dataset(Dataset):
    """ Binding Dataset """
    
    def __init__(self, csv_file:str, refseq:str, model:nn.Module):
        """
        Load sequence label and binding data from csv file and generate full
        sequence using the label and refseq.

        csv_file: a csv file with sequence label and kinetic data.
        refseq: reference sequence (wild type sequence)

        The csv file has this format (notice that ka_sd is not always a number):

        A22C_R127G_E141D_L188V, log10Ka=8.720000267028809, ka_sd=nan
        N13F, log10Ka=10.358182907104492, ka_sd=0.05153989791870117
        V71K_P149L_N157T, log10Ka=6.0, ka_sd=nan
        """
        def _load_csv():
            labels = []
            log10_ka = []
            try:
                with open(csv_file, 'r') as fh:
                    for line in fh.readlines():
                        [label, affinity, _] = line.split(',')
                        affinity = np.float32(affinity.split('=')[1])
                        labels.append(label)
                        log10_ka.append(affinity)
            except FileNotFoundError:
                print(f'File not found error: {csv_file}.', file=sys.stderr)
                sys.exit(1)
            return labels, log10_ka

        self.csv_file = csv_file
        self.refseq = list(refseq)
        self.model = model
        self.labels, self.log10_ka = _load_csv()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
            seq = self._label_to_seq(label)
            features = self.embed(seq)
            return label, features, self.log10_ka[idx]
        
        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)

    def _label_to_seq(self, label: str) -> str:
        """Generate sequence based on reference sequence and mutation label."""
        seq = self.refseq.copy()
        p = '([0-9]+)'
        if '_' in label:
            for mutcode in label.split('_'):
                [ori, pos, mut] = re.split(p, mutcode)
                pos = int(pos)-1    # use 0-based counting
                assert self.refseq[pos].upper() == ori
                seq[pos] = mut.upper()
            seq = ''.join(seq)
            return seq

        if label=='wildtype': return ''.join(seq)

        [ori, pos, mut] = re.split(p, label)
        pos = int(pos)-1    # use 0-based counting
        assert self.refseq[pos] == ori
        seq[pos] = mut.upper()
        seq = ''.join(seq)
        return seq

    def embed(self, seq:str):
        """ Embed sequence. """
        mask_prob = 0
        max_len = 280
        tokenizer = ProteinTokenizer(max_len, mask_prob)
        input_seqs = tokenizer.forward(self.refseq)

        with torch.no_grad():
            last_hidden_states = self.model(input_seqs)[0]

        return last_hidden_states

def load_model(model_pth):
    """ Load in the Spike_NLP model parameters. """

    # -= SPIKE HYPERPARAMETERS =-
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 768
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 12

    # Init spike model
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    embedder = NLPEmbedding(embedding_dim, max_len,dropout)
    vocab_size = len(token_to_index)
    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    model = ProteinLM(bert, vocab_size)

    # Load pretrained spike model
    saved_state = torch.load(model_pth, map_location='cuda')
    state_dict = saved_state['model_state_dict']

    # For loading from ddp models, they have 'module' in keys of state_dict
    load_ddp = False
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            load_ddp = True
            new_state_dict[k[7:]] = v   # remove 'module'
        else: break
    if load_ddp: state_dict = new_state_dict

    # Load pretrained state_dict
    model.load_state_dict(state_dict)
    return model

if __name__=="__main__":

    DATA_DIR = os.path.join(os.path.join(os.path.dirname(__file__), '../../../data'))
    RESULTS_DIR = os.path.join(os.path.join(os.path.dirname(__file__), '../../../results'))
    
    # Data file, reference sequence
    dms_csv = os.path.join(DATA_DIR, 'mutation_binding_Kds.csv')
    refseq = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

    # Load pretrained spike model
    model_pth = os.path.join(RESULTS_DIR, 'ddp-2023-08-14_15-10/ddp-2023-08-14_15-10_best_model_weights.pth')
    model = load_model(model_pth)

    # Dataset, training and test dataset set up
    dataset = DMS_Dataset(dms_csv, refseq, model)
    for idx in range(5):
        label, features, log10_ka = dataset[idx]
        print("Label:", label)
        print("Features shape:", features.shape)
        print("Log10 Ka:", log10_ka)
        print()




