#!/usr/bin/env python
""" DMS Dataset Embedder """

import os
import re
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from collections import OrderedDict
from pnlp.embedding.tokenizer import ProteinTokenizer
from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT

class DMS_Dataset(Dataset):
    """ Binding Dataset """
    
    def __init__(self, csv_file:str, model_pth:str, refseq:str, device: str, embed_method:str):
        """
        Load sequence label and binding data from csv file and generate full
        sequence using the label and refseq.

        csv_file: a csv file with sequence label and kinetic data.
        refseq: reference sequence (wild type sequence)
        device: cuda or cpu
        embed_method: "rbd_learned", "rbd_bert", "esm", "one_hot"

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
        self.model_pth = model_pth
        self.labels, self.log10_ka = _load_csv()
        self.refseq = list(refseq)
        self.device = device
        self.embed_method = embed_method

        if embed_method == "rbd_learned":
            self.embedder = load_nlp_embedder(self.model_pth).to(self.device) 
        elif embed_method == "rbd_bert":
            self.embedder = load_bert_embedder(self.model_pth).to(self.device) 
        elif embed_method == "esm":
            self.embedder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(self.device) 

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

        def _rbd_learned(seq: str) -> torch.Tensor:
            """ Embed sequence feature using rbd learned embeddings. """

            mask_prob = 0
            max_len = 280
            tokenizer = ProteinTokenizer(max_len, mask_prob)
            tokenized_seq = tokenizer(seq).to(self.device)

            with torch.no_grad():
                last_hidden_states, _ = self.embedder(tokenized_seq)
                reshaped_hidden_states = last_hidden_states.squeeze(1).cpu() # torch.Size([x, 1, y]) -> torch.Size([x, y])

            return reshaped_hidden_states

        def _rbd_bert(seq:str) -> torch.Tensor:
            """ Embed sequence feature using rbd learned embeddings and BERT. """

            mask_prob = 0
            max_len = 280
            tokenizer = ProteinTokenizer(max_len, mask_prob)
            tokenized_seq = tokenizer(seq).to(self.device)

            with torch.no_grad():
                last_hidden_states = self.embedder(tokenized_seq)
                reshaped_hidden_states = last_hidden_states.squeeze(1).cpu() # torch.Size([x, 1, y]) -> torch.Size([x, y])

            return reshaped_hidden_states

        def _esm(seq:str) -> torch.Tensor:
            """ Embed sequence feature using ESM model. """

            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            tokenized_seq = tokenizer(seq,return_tensors="pt").to(self.device)
            self.embedder.eval()

            with torch.no_grad():
                last_hidden_states = self.embedder(**tokenized_seq).last_hidden_state
                reshaped_hidden_states = last_hidden_states.squeeze(0)
            
            return reshaped_hidden_states

        def _one_hot(seq: str) -> torch.Tensor:
            """ Embed sequence feature using one-hot. """

            mask_prob = 0
            max_len = 280
            tokenizer = ProteinTokenizer(max_len, mask_prob)
            tokenized_seq = tokenizer(seq).to(self.device)

            with torch.no_grad():
                embedding = nn.functional.one_hot(tokenized_seq).float().squeeze(1).cpu()
            
            return embedding

        if self.embed_method == 'rbd_learned':
            return _rbd_learned(seq)
        elif self.embed_method == 'rbd_bert':
            return _rbd_bert(seq)
        elif self.embed_method == 'one_hot':
            return _one_hot(seq)
        elif self.embed_method == 'esm':
            return _esm(seq)
        else:
            print(f'Undefined embedding method: {self.embed_method}', file=sys.stderr)
            sys.exit(1)

class PKL_Loader(Dataset):
    """ Binding Dataset """
    
    def __init__(self, pickle_file:str, device:str):
        """
        Load sequence label, binding data, and embeddings from pickle file.

        pickle_file: a pickle file with seq_id, log10_ka, seq, and embedding data.
                     NOTE: the embedding tensors are saved as cpu
        device: cuda or cpu
        """
        def _load_dms():
            with open(pickle_file, 'rb') as f:
                dms_list = pickle.load(f)
            
                self.labels = [entry['seq_id'] for entry in dms_list]
                self.log10_ka = [entry['log10_ka'] for entry in dms_list]
                self.embeddings = [entry['embedding'] for entry in dms_list]
 
        self.device = device
        _load_dms()

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

def load_nlp_embedder(model_pth):
    """ Load in the rbd learned Spike_NLP model parameters to NLP embedder. """

    saved_state = torch.load(model_pth, map_location='cuda')
    state_dict = saved_state['model_state_dict']

    # For loading from ddp models, they have 'module.' in keys of state_dict
    prefix = 'module.'
    state_dict = {i[len(prefix):]: j for i, j in state_dict.items() if prefix in i[:len(prefix)]}

    # Embedder set up
    max_len = 280
    embedding_dim = 320
    dropout = 0.1

    embedder = NLPEmbedding(embedding_dim, max_len, dropout)
    embedding_weights = state_dict['bert.embedding.token_embedding.weight']
    
    with torch.no_grad():
        embedder.token_embedding.weight = nn.Parameter(embedding_weights)

    return embedder

def load_bert_embedder(model_pth):
    """ Load in the rbd learned Spike_NLP model parameters to BERT embedder. """

    # Load pretrained spike model weights
    saved_state = torch.load(model_pth, map_location='cuda')
    state_dict = saved_state['model_state_dict']

    # For loading from ddp models, they have 'module.bert.' in keys of state_dict
    if 'ddp' in model_pth:
        prefix = 'module.bert.'
        state_dict = {i[len(prefix):]: j for i, j in state_dict.items() if prefix in i[:len(prefix)]}

    # BERT model hyperparameters
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10 

    embedder = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    embedder.load_state_dict(state_dict)

    return embedder

def generate_embedding_pickle(csv_file:str, dms_dataset:Dataset, embed_method:str):
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
    embedded_file = csv_file.replace('.csv', f'_{embed_method}_embedded.pkl')
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

    # Load pretrained spike model to embedder
    #model_pth = os.path.join(results_dir, 'ddp_runner/ddp-2023-08-16_08-41/ddp-2023-08-16_08-41_best_model_weights.pth') # 768 dim
    model_pth = os.path.join(results_dir, 'ddp_runner/ddp-2023-10-06_20-16/ddp-2023-10-06_20-16_best_model_weights.pth') # 320 dim
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    embed_method = "rbd_learned"
 
    # Dataset, training and test dataset pickles
    print("Making train pickle.")
    train_dataset = DMS_Dataset(dms_train_csv, model_pth, refseq, device, embed_method)
    generate_embedding_pickle(dms_train_csv, train_dataset, embed_method)
    print("\nMaking test pickle.")
    test_dataset = DMS_Dataset(dms_test_csv, model_pth, refseq, device, embed_method)
    generate_embedding_pickle(dms_test_csv, test_dataset, embed_method)

    # Test the pickle loader
    print("\nTesting pickle loader")
    embedded_train_pkl = dms_train_csv.replace('.csv', f'_{embed_method}_embedded.pkl')
    embedded_test_pkl = dms_test_csv.replace('.csv', f'_{embed_method}_embedded.pkl')

    train_pkl_loader = PKL_Loader(embedded_train_pkl, device)
    test_pkl_loader = PKL_Loader(embedded_test_pkl, device)

    print("Loaded training dataset from pickle:")
    for i in range(5):  # adjust the range as needed to print a few examples
        print(train_pkl_loader[i])

    print("\nLoaded test dataset from pickle:")
    for i in range(5):  # adjust the range as needed to print a few examples
        print(test_pkl_loader[i])
