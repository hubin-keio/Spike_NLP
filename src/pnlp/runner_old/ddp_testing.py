#!/usr/bin/env python
"""
Model runner for ESM-initialized BERT-MLM model. We utilize DistributedDataParallel.

*GPU ONLY* to avoid device assignment issues.

Usage:
    torchrun
    > --standalone: utilize single node
    > --nproc_per_node: number of processes/gpus
"""

import os
import torch
import random
import datetime
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record


# Define Dataset
class RBDDataset(Dataset):
    def __init__(self, csv_file: str):
        try:
            self.full_df = pd.read_csv(csv_file, sep=',', header=0)
        except (FileNotFoundError, pd.errors.ParserError, Exception) as e:
            print(f"Error reading in .csv file: {csv_file}\n{e}", file=sys.stderr)
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.full_df)

    def __getitem__(self, idx):
        seq_id = self.full_df['seq_id'][idx]
        sequence = self.full_df['sequence'][idx]
        random_value = random.random()  # Generate a random number
        return seq_id, sequence, random_value


@record
def main():
    # Initialize process group - DDP
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../data/rbd')

    # Run setup
    batch_size = 64
    num_workers = 8
    num_epochs = 2  # For testing over multiple epochs

    # Check if it can run on GPU - DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else sys.exit(1)
    torch.cuda.set_device(local_rank)

    # Create Dataset and DataLoader, use DistributedSampler - DDP
    seed = 0
    torch.manual_seed(seed)

    train_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_train.csv"))

    # =====================
    # NO Worker Seed Section
    # =====================
    print("\nNo, seed for workers")

    # Create DistributedSampler without worker-specific seeding
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        sampler=train_sampler
    )

    # Run for multiple epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch} (Rank {local_rank}) - No worker seeds")
        train_sampler.set_epoch(epoch)  # Ensure different shuffling per epoch
        for i, batch in enumerate(train_data_loader):
            if i >= 5:  # Only print the first 5 batches
                break
            seq_ids, sequences, random_values = batch
            print(f"[Rank {local_rank}] Epoch {epoch}, Batch {i + 1}: First 5 random values: {random_values[:5]}")

    # =====================
    # WITH Worker Seed Section
    # =====================
    print("\nYes, seed for workers")

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id + local_rank * num_workers  # Unique seed per worker and rank
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        print(f"Worker {worker_id} (Rank {local_rank}) initialized with seed: {worker_seed}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)

    # Run for multiple epochs with worker-specific seeding
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch} (Rank {local_rank}) - With worker seeds")
        train_sampler.set_epoch(epoch)  # Ensure different shuffling per epoch
        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=True, 
            num_workers=num_workers, 
            worker_init_fn=worker_init_fn, 
            pin_memory=True, 
            sampler=train_sampler
        )
        
        for i, batch in enumerate(train_data_loader):
            if i >= 5:  # Only print the first 5 batches
                break
            seq_ids, sequences, random_values = batch
            print(f"[Rank {local_rank}] Epoch {epoch}, Batch {i + 1}: First 5 random values: {random_values[:5]}")

    # Clean up - DDP
    dist.destroy_process_group()

if __name__ == '__main__':
    main()