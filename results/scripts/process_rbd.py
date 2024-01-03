#!/usr/bin/env python
"""
Variant Distribution

Bash code used to extract the Accession IDs and Variants from 
larger meta data file:
    cat spikeprot0528.clean.uniq.noX.RBD.metadata.tsv | cut -f 5,16 | awk '{print $1, $3}' > spikeprot0528.clean.uniq.noX.RBD.metadata_variants.txt

> Extracts the Accession IDs that have Variants from the resulting
  txt file of the above command. 

> Then goes through the training and testing database to see which 
  Accession IDs with variant labels are within each database. 

> Resulting Accession IDs with no variant labels are kept track of 
  in a csv. 

> Those with variant labels are also kept track of in a csv, but 
  plotted to show distribution of the variants among Accession IDs 
  within the training and testing databases. 

Other distribution plots included, such as 
    - amino acid distribution (related to the above desc), 
    - sequence length distribution,
    - variant distribution of the highest frequency sequence length.
"""
import os
import tqdm
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from pnlp.db.dataset import SeqDataset, initialize_db
