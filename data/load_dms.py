#!/usr/bin/env python
""" DMS Dataset Embedder """

import os
import re
import sys
import torch
import pickle
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
    
    def __init__(self, csv_file:str, refseq:str, model:nn.Module, device: str):
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
        self.device = device
        self.labels, self.log10_ka = _load_csv()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
            log10_ka = self.log10_ka[idx]
            seq = self._label_to_seq(label)
            features = self.embed(seq)
            return label, log10_ka, seq, features

        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            return None

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
        tokenized_seq = tokenizer(seq)
        tokenized_seq = tokenized_seq.to(self.device)

        with torch.no_grad():
            last_hidden_states = self.model(tokenized_seq)
            reshaped_hidden_states = last_hidden_states.view(-1, 768).cpu()

        return reshaped_hidden_states

class PKL_Loader(Dataset):
    """ Binding Dataset """
    
    def __init__(self, pickle_file:str, device:str):
        """
        Load sequence label, binding data, and embeddings from pickle file.

        pickle_file: a pickle file with seq_id, log10_ka, seq, and embedding data.
                     NOTE: the embedding tensors are saved as cpu
        device: cuda or cpu
        """
        with open(pickle_file, 'rb') as f:
            dms_list = pickle.load(f)
        
        self.device = device
        self.labels = [entry['seq_id'] for entry in dms_list]
        self.log10_ka = [entry['log10_ka'] for entry in dms_list]
        self.embeddings = [entry['embedding'] for entry in dms_list]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            label = self.labels[idx]
            features = self.embeddings[idx].to(self.device)
            targets = self.log10_ka[idx]
            
            return label, features, targets
        
        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)

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
    model = bert

    # Load pretrained spike model into bert
    saved_state = torch.load(model_pth, map_location='cuda')
    state_dict = saved_state['model_state_dict']
    load_ddp = False
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k[:11] == 'module.bert':
            load_ddp = True
            new_state_dict[k[12:]] = v   # remove 'module.bert.'
        else: break
        
    if load_ddp: state_dict = new_state_dict

    # Load pretrained state_dict
    model.load_state_dict(state_dict)
    return model

def generate_embedding_pickle(csv_file:str, dms_dataset:Dataset):
    """
    Embed the sequences from the DMS dataset. This will include 4 specific items:
        - Sequence Identifier (seq_id): the specific mutations to perform to the reference sequence, identifier
            - ex: A22C_R127G_E141D_L188V
        - Log10 Ka (log10_ka): log10 ka value associated with the sequence
            ex: 8.720000267028809
        - Sequence (seq): sequence post transformation of the reference sequence to contain the 
            corresponding  mutation as noted in the sequence label
            - ex: NITNLCPFGEVFNATRFASVYCWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFGKSNLKPFERDISTDIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELVHAPATVCGPKKST
        - Embedded Sequence (embedding): sequence embedding ran through the BERT model utilizing the 
            best pretrained masked protein language model, tensor format.
            ex: not putting the output, but should be of shape: torch.Size([, 768])
    """
    embedded_file = csv_file.replace('.csv', '_embedded.pkl')
    dms_list = []

    for i, item in enumerate(dms_dataset):
        if item == None:
            break

        seq_id, log10_ka, seq, embedding = item                
        entry = {"seq_id": seq_id,
                 "log10_ka": log10_ka,
                 "seq": seq,
                 "embedding": embedding}
        dms_list.append(entry)
        print(f"{i}: {entry['seq_id']}, {entry['embedding'].shape}", flush=True)

    with open(embedded_file, 'wb') as f:
        pickle.dump(dms_list, f)
            
if __name__=="__main__":

    data_dir = os.path.dirname(__file__)
    results_dir = os.path.join(data_dir, '../results')
    
    # Data file, reference sequence
    dms_train_csv = os.path.join(data_dir, 'dms/mutation_binding_Kds_train.csv')
    dms_test_csv = os.path.join(data_dir, 'dms/mutation_binding_Kds_test.csv')
    refseq = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

    # Load pretrained spike model
    model_pth = os.path.join(results_dir, 'ddp_runner/ddp-2023-08-16_08-41/ddp-2023-08-16_08-41_best_model_weights.pth')
    model = load_model(model_pth)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = model.to(device)  

    # Dataset, training and test dataset pickles
    print("Making train pickle.")
    train_dataset = DMS_Dataset(dms_train_csv, refseq, model, device)
    generate_embedding_pickle(dms_train_csv, train_dataset)
    print("\nMaking test pickle.")
    test_dataset = DMS_Dataset(dms_test_csv, refseq, model, device)
    generate_embedding_pickle(dms_test_csv, test_dataset)

    # Test the pickle loader
    print("\nTesting pickle loader")
    embedded_train_pkl = dms_train_csv.replace('.csv', '_embedded.pkl')
    embedded_test_pkl = dms_test_csv.replace('.csv', '_embedded.pkl')

    train_pkl_loader = PKL_Loader(embedded_train_pkl, device)
    test_pkl_loader = PKL_Loader(embedded_test_pkl, device)

    print("Loaded training dataset from pickle:")
    for i in range(5):  # adjust the range as needed to print a few examples
        print(train_pkl_loader[i])

    print("\nLoaded test dataset from pickle:")
    for i in range(5):  # adjust the range as needed to print a few examples
        print(test_pkl_loader[i])