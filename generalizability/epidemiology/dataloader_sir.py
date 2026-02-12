"""
dataloader_sir.py

Dataloader for COVID-19 epidemiology dataset (county level).

Data format:
    - Input columns: input_node{i}_t{j} for i in 0..n_counties-1, j in 0..4
    - Target columns: target_node{i}_h{j} for i in 0..n_counties-1, j in 1..15
    - Adjacency: n_counties x n_counties matrix of county connectivity

FIXED: Pre-extracts data as numpy arrays in __init__ to avoid pandas threading issues.
       This fixes the "TypeError: type 'NumpyBlock' is not subscriptable" error.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path


class EpidemiologyDataset(Dataset):
    """Dataset for COVID-19 epidemiology prediction (county level).

    Thread-safe implementation using pre-extracted numpy arrays.
    """

    def __init__(self, data_path, metadata_path=None):
        """
        Args:
            data_path: Path to CSV file (train.csv, val.csv, or test.csv)
            metadata_path: Path to metadata.json (optional, for normalization stats)
        """
        # Load DataFrame temporarily
        df = pd.read_csv(data_path)

        # Load metadata if provided
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        # Parse column structure
        self._parse_columns(df)

        # Pre-extract all data as numpy arrays (fixes pandas threading bug)
        self._extract_data(df)

        # Store length
        self._len = len(df)

        print(f"Loaded {self._len} samples")
        print(f"  Nodes (counties): {self.n_nodes}")
        print(f"  Input window: {self.input_window}")
        print(f"  Prediction horizon: {self.pred_horizon}")

        # Delete DataFrame to free memory
        del df

    def _parse_columns(self, df):
        """Parse input and target column structure."""
        # Find input columns: input_node{i}_t{j}
        input_cols = [c for c in df.columns if c.startswith('input_node')]
        target_cols = [c for c in df.columns if c.startswith('target_node')]

        # Parse node and time indices
        # input_node0_t0 -> node=0, t=0
        input_info = []
        for col in input_cols:
            parts = col.replace('input_node', '').split('_t')
            node_idx = int(parts[0])
            time_idx = int(parts[1])
            input_info.append((col, node_idx, time_idx))

        # target_node0_h1 -> node=0, h=1
        target_info = []
        for col in target_cols:
            parts = col.replace('target_node', '').split('_h')
            node_idx = int(parts[0])
            horizon_idx = int(parts[1])
            target_info.append((col, node_idx, horizon_idx))

        # Determine dimensions
        self.n_nodes = max(info[1] for info in input_info) + 1
        self.input_window = max(info[2] for info in input_info) + 1
        self.pred_horizon = max(info[2] for info in target_info)

        # Store column mappings
        self.input_cols = sorted(input_info, key=lambda x: (x[1], x[2]))
        self.target_cols = sorted(target_info, key=lambda x: (x[1], x[2]))

    def _extract_data(self, df):
        """Pre-extract all data as numpy arrays for thread-safe access.

        This avoids the pandas threading bug:
        "TypeError: type 'NumpyBlock' is not subscriptable"
        """
        n_samples = len(df)

        # Pre-allocate arrays
        self.input_data = np.zeros((n_samples, self.n_nodes, self.input_window), dtype=np.float32)
        self.target_data = np.zeros((n_samples, self.n_nodes, self.pred_horizon), dtype=np.float32)

        # Extract input data
        for col, node_idx, time_idx in self.input_cols:
            self.input_data[:, node_idx, time_idx] = df[col].values.astype(np.float32)

        # Extract target data
        for col, node_idx, horizon_idx in self.target_cols:
            self.target_data[:, node_idx, horizon_idx - 1] = df[col].values.astype(np.float32)  # h1 -> index 0

        # Extract metadata columns
        self.time_idx = df['time_idx'].values.astype(np.int64) if 'time_idx' in df.columns else np.zeros(n_samples, dtype=np.int64)
        self.year = df['year'].values.astype(np.int64) if 'year' in df.columns else np.zeros(n_samples, dtype=np.int64)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """Thread-safe data access using pre-extracted numpy arrays."""
        return {
            'input': torch.from_numpy(self.input_data[idx]),      # (n_nodes, input_window)
            'target': torch.from_numpy(self.target_data[idx]),    # (n_nodes, pred_horizon)
            'time_idx': self.time_idx[idx],
            'year': self.year[idx]
        }


def load_adjacency(adj_path):
    """Load adjacency matrix from CSV."""
    adj_df = pd.read_csv(adj_path, index_col=0)
    adj = adj_df.values.astype(np.float32)
    return adj, list(adj_df.columns)


def create_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=0
):
    """
    Create train/val/test dataloaders.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv, adjacency.csv, metadata.json
        batch_size: Batch size
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader, test_loader, adjacency, metadata
    """
    data_dir = Path(data_dir)

    # Load metadata
    metadata_path = data_dir / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load adjacency
    adj, county_ids = load_adjacency(data_dir / 'adjacency.csv')

    print("=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    # Create datasets
    print("\nTraining dataset:")
    train_dataset = EpidemiologyDataset(
        data_dir / 'train.csv',
        metadata_path
    )

    print("\nValidation dataset:")
    val_dataset = EpidemiologyDataset(
        data_dir / 'val.csv',
        metadata_path
    )

    print("\nTest dataset:")
    test_dataset = EpidemiologyDataset(
        data_dir / 'test.csv',
        metadata_path
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("\n" + "=" * 60)
    print("DATALOADER SUMMARY")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Counties: {len(county_ids)} total")
    print(f"Data level: {metadata.get('level', 'county')}")

    return train_loader, val_loader, test_loader, adj, metadata


if __name__ == "__main__":
    # Test dataloader
    data_dir = Path(__file__).parent / 'data'

    train_loader, val_loader, test_loader, adj, metadata = create_dataloaders(
        data_dir,
        batch_size=32
    )

    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input: {batch['input'].shape}")
    print(f"  Target: {batch['target'].shape}")
    print(f"\nMetadata:")
    print(f"  n_nodes: {metadata['n_nodes']}")
    print(f"  input_window: {metadata['input_window']}")
    print(f"  pred_horizon: {metadata['pred_horizon']}")
