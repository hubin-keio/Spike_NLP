#!/usr/bin/env python
"""
Clustering analysis

This script utilize the ESM model with best pretraining parameters to generate
the hidden states of all the sequences of SARS-CoV-2 RBD sequences, which will be
used for clustering, graphing, and evolution studies.
"""

import os
import torch
import pickle
import pandas as pd
import tqdm
import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel

class ESM_Runner:
    """ ESM model runner. """

    def __init__(self,
                 csv_name:str,
                 batch_size:int,
                 device:str):

        self.batch_size = batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.model.to(device)

        self.save_as = csv_name.replace(".csv", "_cluster_esm_embedding.pkl")

    def run(self, seq_data:Dataset):

        # Set the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(seq_data),
                              total = len(seq_data),
                              bar_format='{l_bar}{r_bar}')

        all_seq_ids, all_variants, all_embeddings = [], [], []

        for i, batch_data in data_iter:
            seq_ids, variants, seqs = batch_data
            tokenized_seqs = self.tokenizer(seqs, return_tensors="pt", padding="max_length", truncation=True, max_length=225)
            tokenized_seqs = tokenized_seqs.to(self.device)  # input tokens with masks
            self.model.eval()

            with torch.no_grad():
                last_hidden_states = self.model(**tokenized_seqs).last_hidden_state
                embeddings = last_hidden_states.cpu().numpy() # shape is (64, 225, 320)
                
                # This way creates significantly smaller pickle files than dictionary entries
                for seq_id, variant, embedding in zip(seq_ids, variants, embeddings):
                    if embedding.shape == (225, 320):
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

    data_dir = os.path.join(os.path.dirname(__file__), 'spike_variants')

    batch_size = 64
    torch.manual_seed(0) # Dataloader uses its own random number generator.
    
    USE_GPU = True
    device = torch.device("cuda:3" if torch.cuda.is_available() and USE_GPU else "cpu")    

    # Train dataset pickle
    train_csv_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_train.csv")
    train_seq_dataset = VariantSeqDataset(train_csv_file)
    train_seq_loader = DataLoader(train_seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_runner = ESM_Runner(train_csv_file, batch_size, device)
    train_runner.run(train_seq_loader)

    # Test dataset pickle
    test_csv_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants_test.csv")
    test_seq_dataset = VariantSeqDataset(test_csv_file)
    test_seq_loader = DataLoader(test_seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_runner = ESM_Runner(test_csv_file, batch_size, device)
    test_runner.run(test_seq_loader)

    # Full dataset pickle
    full_csv_file = os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD_variants.csv")
    full_seq_dataset = VariantSeqDataset(full_csv_file)
    full_seq_loader = DataLoader(full_seq_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    full_runner = ESM_Runner(full_csv_file, batch_size, device)
    full_runner.run(full_seq_loader)