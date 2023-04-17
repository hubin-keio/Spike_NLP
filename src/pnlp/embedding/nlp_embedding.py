"""
Methods to embed batched sequences

Author: Bin Hu
"""
import os
import math
import random
import torch
import torch.nn as nn
from typing import Tuple
from collections.abc import Sequence

class PositionalEmbedding(nn.Module):
    """
    Impement the PE function.

    The PE forward function is different from the BERT-pytorch. Here we used the original method in BERT so
    PE embeddings are added to the input embeddings and no graident tracking is used.
    """

    def __init__(self,
                 d_model: int,       # model input dimension
                 dropout: float=0.1, # dropout rate
                 max_len: int=1500): # maximum sequence length
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # adding positional embeddings on top of original embedding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class NLPEmbedding(nn.Module):
    """
    Encode batched amino acid sequences. In our model, we only need the token and position embedding
    but not the segment embedding <SEP> used in the orignal BERT.
    """
    def __init__(self,
                 embedding_dim: int,    # embedding dimensions
                 dropout: float=0.1,    # dropout rate
                 max_len: int=1500,     # maximum length of input sequence
                 mask_prob: float=0.15): # masking probability

        super().__init__()
        self.embeddng_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.__init_tokens__()
        self.token_embedding = nn.Embedding(len(self.token_to_index), embedding_dim, self.padding_idx)
        self.add_position = PositionalEmbedding(embedding_dim, dropout, max_len)

    def __init_tokens__(self):
        """
        Initialize token dictionaries
        """
        ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
        ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>', '<MASK>', '<TRUNCATED>']

        # Each sequence is added <START> and <END>. "<PAD>" are added to sequence shorten than max_len.
        ADDED_TOKENS_PER_SEQ = 2

        n_aas = len(ALL_AAS)
        self.aa_to_token = {aa: i for i, aa in enumerate(ALL_AAS)}
        additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}

        self.token_to_index = {**self.aa_to_token, **additional_token_to_index}
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}
        self.padding_idx = self.token_to_index['<PAD>']

    def batch_pad(self, batch_seqs:Sequence[str]) -> torch.Tensor:
        """
        Tokenize a sequence batch and add padding when needed.

        Returns tokenized tensor of the shape [batch_size, longest_seq].
        If the longest sequence in the batch is longer then max_len, the shape is [batch_size, max_len].

        Arguments:

        """
        tokenized_seqs = []
        for a_seq in batch_seqs:
            if len(a_seq) < self.max_len:
                a_seq = [self.aa_to_token.get(aa, self.token_to_index['<OTHER>']) for aa in a_seq]

            # if more  then max_len, we will need to truncate it and mark it <TRUNCATED>
            else:
                a_seq = [self.aa_to_token.get(aa, self.token_to_index['<OTHER>']) for aa in a_seq[:self.max_len-1]]
                a_seq.append(self.token_to_index['<TRUNCATED>'])
            tokenized_seqs.append(a_seq)

        max_in_bach = max([len(a_seq) for a_seq in tokenized_seqs])  # length of longest sequence in batch
        for _, seq in enumerate(tokenized_seqs):
            n_pad = max_in_bach - len(seq)
            if n_pad > 0:
                for p in range(n_pad):
                    seq.append(self.padding_idx)
                tokenized_seqs[_] = seq

        tokenized_seqs = torch.tensor(tokenized_seqs)

        return tokenized_seqs

    def batch_mask(self, batch_padded_seqs: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Add <MASK> token at random locations based on assigned probabilities.

        Locations of the <MASK> are the same across one batch with the exception that
        speical tokens, e.g. <PAD> <TRUNCATED> are not masked. Returns a tuple of
        (seq_ids, masked sequence tokens, original tokens)

        Parameter:

        batch_tokenized_seqs: batch of padded sequences from batch_pad method.

        """
        batch_masked = batch_padded_seqs.clone()
        seq_len = batch_padded_seqs.size(1)

        n_mask = max(int(self.mask_prob * seq_len), 1) # at least one mask

        row_idx = range(batch_padded_seqs.size(0))


        SPECIAL_TOKENS = torch.tensor([self.token_to_index[st] for st in ['<PAD>', '<TRUNCATED>']])
        MASK_IDX = self.token_to_index['<MASK>']

        masked_idx = []

        for _ in range(n_mask):
            idx = int(random.random() * seq_len)
            if not torch.any(torch.isin(batch_masked[:, idx], SPECIAL_TOKENS)):
                batch_masked[:, idx] = torch.tensor(MASK_IDX)
                masked_idx.append(idx)
        return batch_masked, masked_idx

    def load_pretrained_embeddings(self, pte:str, no_grad=True):
        """Load pretrained embeddings and prevent it from updating if no_grad is True"""
        assert(os.access(pte, os.R_OK))
        weights = torch.load(pte)
        if no_grad:
            weights.requires_grad_ = False
        self.token_embedding.weight = nn.Parameter(weights)


    def forward(self, batch_seq:Sequence[str]):  # input batch_seq is a sequence of amino acids.
        x_padded = self.batch_pad(batch_seq)
        x_padded_masked, _ = self.batch_mask(x_padded)
        x = self.token_embedding(x_padded_masked)
        x = self.add_position(x)
        return self.dropout(x)
