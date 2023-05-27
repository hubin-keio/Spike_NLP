'''Extract embedding from a PTH file'''

import os
import csv
import torch

from pnlp.embedding.tokenizer import index_to_token

def extract_ebmedding(pth: str) -> None:
    '''
    Extract embeeding from x.pth and write it to x_embeddings.csv.

    x.pth is a odel state dictionary saved by torch.save().
    '''
    state_dict = torch.load(pth)
    embedding_key = 'bert.embedding.token_embedding.weight'
    assert embedding_key in state_dict.keys()

    embeddings = state_dict[embedding_key]
    embedding_dim = embeddings.shape[1]
    embeddings = embeddings.tolist()
    vocab_size = len(index_to_token)

    # Header row
    rows = [['Token'] + [f'Dimension_{i}' for i in range(1, embedding_dim+1)]]

    # Add word and corresponding tensor values to each row
    for i in range(vocab_size):
        row = [index_to_token[i]] + embeddings[i]
        rows.append(row)

    # Write the rows to a CSV file
    output_file = pth.replace('.pth', '_embeddings.csv')
    with open(output_file, 'w', newline='\n') as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
