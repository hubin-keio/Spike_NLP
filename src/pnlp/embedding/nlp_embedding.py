"""
Methods to embed batched sequences

Author: Bin Hu
"""
import os
import math
import torch
import torch.nn as nn

from pnlp.embedding.tokenizer import ProteinTokenizer


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
    def __init__(self, embedding_dim: int, max_len:int, dropout: float=0.1):
        """
        Parameters:

        embedding_dim: embedding dimensions
        dropout: dropout rate
        """

        super().__init__()
        self.embeddng_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        tokenizer = ProteinTokenizer(0, 0.0)
        self.padding_idx = tokenizer.token_to_index['<PAD>']
        self.token_embedding = nn.Embedding(len(tokenizer.token_to_index),
                                            embedding_dim,
                                            self.padding_idx)  # prevent <PAD> embedding from updating.
        self.add_position = PositionalEmbedding(embedding_dim, dropout, max_len)


    def load_pretrained_embeddings(self, pte:str, no_grad=True):
        """Load pretrained embeddings and prevent it from updating if no_grad is True"""
        assert(os.access(pte, os.R_OK))
        weights = torch.load(pte)
        if no_grad:
            weights.requires_grad_ = False
        self.token_embedding.weight = nn.Parameter(weights)


    def forward(self, batch_token: torch.Tensor):
        """
        Sequence embedding

        Parameters:
        batch_token: Tensor of the shape (batch_size, max(longest_seq, max_len))
        """
        x = self.token_embedding(batch_token)
        x = self.add_position(x)

        (batch_size, seq_len) = batch_token.shape

        padding_masks = batch_token == self.padding_idx
        padding_masks = padding_masks.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        mask_tensor = torch.ones(batch_size, seq_len, seq_len)
        mask_tensor = mask_tensor.masked_fill(padding_masks, 0.0)

        return self.dropout(x), mask_tensor
