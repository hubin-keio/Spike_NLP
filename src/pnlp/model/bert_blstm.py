"""
BLSTM with FCN layer, MLM, and BERT.
"""

import torch
import torch.nn as nn
from pnlp.model.language import ProteinMaskedLanguageModel, BERT
from blstm import BLSTM

class BERT_BLSTM(nn.Module):
    """"
    BERT protein language model
    """
    def __init__(self, bert: BERT, blstm:BLSTM, vocab_size: int, alpha:float):
        super().__init__()
        self.alpha = alpha

        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)
        self.blstm = blstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)

        # shape of x is [torch.Size([32, 201, 320])] batch_size, seq_len, embedding dim
        # bert does not like, mismatch of 201 and 320?
        # reshape??

        x = self.bert(x)
        exit()
        error_1 = self.mlm(x) # error from masked language
        error_2 = self.blstm(x) # error from regession

        print(f"MLM Error: {error_1}")
        print(f"BLSTM Error: {error_2}")

        total_error = error_1 + (self.alpha * error_2) # alpha is a weight parameter.
        return total_error