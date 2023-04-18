"""Attention Network"""

import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    """Single head scaled dot product attention"""
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    #sqrt(d_k) is the scaling factor.

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

# %%
class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention

    h: numer of heads
    d_model: model size

    """
    def __init__(self, h:int, d_model:int, n_linear: int=4, dropout=0.1):
        super().__init__()
        assert d_model % h == 0 # d_model/h is used as d_k and d_v

        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_linear)])  # n layers of linear model with the same input and output size
        self.output_linear = nn.Linear(d_model, d_model)    # Output lienar model. This implementation follows BERT-pytorch instead of using the last linear layer, which is found in the annotated transformer.
        self.attn = Attention() # The forward function in Attention class is called since no hooks are defined in Attention class. See __call__() and _call_impl() in nn.Module implementation.

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)    # same mask applied to all heads
        n_batches = query.size(0)

        # 1) Linear projections in batch from d_model => h x d_k
        query, key, value = [lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                             for lin, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, attn = self.attn(query, key, value, mask=mask, dropout = self.dropout)   # Returned attn is not needed since x has already been weighted by attention in Attention.forward().

        # 3) "Concat using a view and apply a final linear"
        x = (x.transpose(1, 2)
             .contiguous()
             .view(n_batches, -1, self.h * self.d_k))

        return self.output_linear(x)
