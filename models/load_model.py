import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pnlp.model.language import ProteinLM
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index, index_to_token
from pnlp.model.language import ProteinLM
from pnlp.model.bert import BERT


class AlphpaSeqDataset(Dataset):
    """Binding dataset."""
    def __init__(self, csv_file: str, model_path: str):
        """
        Load sequence label and binding data from csv file and generate full
        sequence using the label and refseq.

        csv_file: a csv file with sequence and kinetic data.
        refseq: reference sequence (wild type sequence)
        transformer: a Transfomer object for embedding features.
        """

        def _load_csv():
            labels = []
            log10_ka = []
            seqs = []
            try:
                with open(csv_file, 'r') as fh:
                    next(fh) #skip header
                    for line in fh.readlines():
                        line = line.split(',')
                        labels.append(line[0])
                        seqs.append(line[1])
                        log10_ka.append(np.float32(line[14]))
            except FileNotFoundError:
                print(f'File not found error: {csv_file}.', file=sys.stderr)
                sys.exit(1)
            return labels, seqs, log10_ka

        self.csv_file = csv_file
        self.labels, self.seqs, self.log10_ka = _load_csv()
        self._longest_seq = len(max(self.seqs,key=len))
        self.pretrained_model = load_model(model_path)
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            embedded_seq = get_embedding([self.seqs[idx]],self.pretrained_model, self._longest_seq)
            padding_size =  self._longest_seq - embedded_seq.size(0)
            features = torch.nn.functional.pad(embedded_seq,(0,0,0, padding_size))
            return self.labels[idx], features, self.log10_ka[idx]
        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)

def load_model(model_pth):
    embedding_dim = 768
    dropout=0.1
    max_len = 280
    mask_prob = 0.15
    n_transformer_layers = 12
    attn_heads = 12
    hidden = embedding_dim

    vocab_size = len(token_to_index)
    
    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, attn_heads)

    model = ProteinLM(bert, vocab_size)
    # Load the saved model
    model.load_state_dict(torch.load(model_pth,map_location=torch.device('cpu')))
    model.eval()
    
    return model

def get_embedding(seqs,model,max_len):
    
    mask_prob = 0
    tokenizer = ProteinTokenizer(max_len, mask_prob)
    input_seqs = tokenizer.forward(seqs)

    with torch.no_grad():
        last_hidden_states = model(input_seqs)[0]

    return last_hidden_states

if __name__ == '__main__':
    seqs = ['RVQPTESIVRFPNITNLCPFDEVFNATTFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSTVGGNYNYRYRLFRKSKLKPFERDISTEIYQAGNKPCNGVAGVNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF']
    model_pth = 'results/2023-05-28_02_26_model_weights.pth'
    model = load_model(model_pth)
    max_len = 250 
    output = get_embedding(seqs,model,max_len)
    
    target_size = 225
    padding_size = target_size - output.size(0)
    padded_tensor = torch.nn.functional.pad(output,(0,0,0, padding_size))
    print(output.size(), padded_tensor.size())
    
    