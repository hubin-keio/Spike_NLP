"""BERT Model"""


import copy
import torch
import torch.nn as nn

from pnlp.embedding.nlp_embedding import NLPEmbedding
from pnlp.model.transformer import TransformerBlock

class BERT(nn.Module):
    """
    BERT model
    """

    def __init__(self, 
                 embedding_dim: int,
                 dropout: float,
                 max_len: int,
                 mask_prob: float,
                 hidden: int,
                 n_transformer_layers: int,
                 attn_heads: int):
        """
        embedding_dim: dimensions of embedding
        hidden: BERT model size (used as input size and hidden size)
        n_layers: number of Transformer layers
        attn_heads: attenion heads
        dropout: dropout ratio
        """

        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.max_len = max_len
        self.mask_prob = mask_prob
        
        self.hidden  = hidden
        self.n_transformer_layers = n_transformer_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4         # 4 * hidden_size for FFN

        self.embedding = NLPEmbedding(self.embedding_dim, self.dropout, self.max_len, self.mask_prob)

        def clones(module, n):
            """Produce N identical layers"""
            return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

        self.transformer_blocks = clones(TransformerBlock(self.hidden, self.attn_heads, self.feed_forward_hidden, self.dropout), self.n_transformer_layers)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)   # batch sequences
        #TODO: Generate mask here.
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask=None)
        return x
