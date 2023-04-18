"""Tokenizer for protein sequences"""

import os
import math
import random
import torch
import torch.nn as nn
from typing import Tuple
from collections.abc import Sequence


class ProteinTokenier:
    def __init__(self, max_len: int, mask_prob: float):
        """
        Parameters:

        max_len: maximum length of the input sequence before it is truncated to this lenth
        mask_prob: probability of token masking. 0.0 for no masking.
        """
        self.max_len = max_len        
        self.mask_prob = mask_prob

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

    def _batch_pad(self, batch_seqs:Sequence[str]) -> torch.Tensor:
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

    def _batch_mask(self, batch_padded_seqs: torch.Tensor) -> Tuple[torch.Tensor, list]:
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

    def get_token(self, batch_seqs:Sequence[str]):
        """Get token representation for a batch of sequences and index of added <MASK>."""
        x_padded = self._batch_pad(batch_seqs)
        if self.mask_prob > 0:
            return self._batch_mask(x_padded)
        else:
            return x_padded, []
