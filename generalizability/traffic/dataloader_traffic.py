"""
dataloader_traffic.py

Dataloader for METR-LA traffic speed dataset.
Uses STAN-style CSV format with columns: input_node{i}_t{j} and target_node{i}_h{j}

IMPORTANT: Data is in ORIGINAL SCALE (mph), NOT normalized!
This is critical for physics-based models where parameters like
free-flow speed (65 mph) need to operate on real speed values.
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple


class TrafficDataset(Dataset):
    """
    Traffic speed prediction dataset.

    Loads from preprocessed CSV files with STAN-style format:
    - Input: [n_nodes, input_window] traffic speeds (mph)
    - Target: [n_nodes, pred_horizon] future speeds (mph)

    IMPORTANT: Data is kept in ORIGINAL SCALE (mph), NOT normalized!
    This is required for physics-based models where parameters like
    free-flow speed need meaningful units.
    """

    def __init__(
        self,
        csv_path: str,
        n_nodes: int,
        input_window: int,
        pred_horizon: int,
        mean: float = None,
        std: float = None
    ):
        """
        Args:
            csv_path: Path to train/val/test CSV file
            n_nodes: Number of sensor nodes
            input_window: Number of input time steps
            pred_horizon: Number of prediction time steps
            mean: Data mean (stored for reference, NOT used for normalization)
            std: Data std (stored for reference, NOT used for normalization)
        """
        self.csv_path = Path(csv_path)
        self.n_nodes = n_nodes
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.mean = mean
        self.std = std

        # Load data
        self.df = pd.read_csv(csv_path)

        # Identify input and target columns
        self.input_cols = [
            f'input_node{i}_t{j}'
            for i in range(self.n_nodes)
            for j in range(self.input_window)
        ]
        self.target_cols = [
            f'target_node{i}_h{j}'
            for i in range(self.n_nodes)
            for j in range(self.pred_horizon)
        ]

        # Verify columns exist
        missing_input = [c for c in self.input_cols if c not in self.df.columns]
        missing_target = [c for c in self.target_cols if c not in self.df.columns]
        if missing_input:
            raise ValueError(f"Missing input columns: {missing_input[:5]}...")
        if missing_target:
            raise ValueError(f"Missing target columns: {missing_target[:5]}...")

        # Pre-extract data as numpy arrays (avoids pandas threading issues)
        # Shape: [n_samples, n_nodes * input_window] -> [n_samples, n_nodes, input_window]
        self.inputs = self.df[self.input_cols].values.astype(np.float32)
        self.inputs = self.inputs.reshape(-1, self.n_nodes, self.input_window)

        # Shape: [n_samples, n_nodes * pred_horizon] -> [n_samples, n_nodes, pred_horizon]
        self.targets = self.df[self.target_cols].values.astype(np.float32)
        self.targets = self.targets.reshape(-1, self.n_nodes, self.pred_horizon)

        # Free the dataframe to save memory
        n_samples = len(self.df)
        del self.df

        # Print data scale info
        sample_values = self.inputs[:, :, :12].flatten()
        data_min = sample_values.min()
        data_max = sample_values.max()
        data_mean = sample_values.mean()
        print(f"  -> {n_samples} samples, {self.n_nodes} sensors")
        print(f"     Data scale: min={data_min:.1f}, max={data_max:.1f}, mean={data_mean:.1f} mph (ORIGINAL SCALE)")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': torch.from_numpy(self.inputs[idx]),
            'target': torch.from_numpy(self.targets[idx]),
        }


def create_dataloaders(
    data_dir: str,
    input_window: int = 12,
    pred_horizon: int = 12,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, Dict]:
    """
    Create train/val/test dataloaders for METR-LA traffic data.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv, metadata.json, adjacency.npy/npz
        input_window: Input sequence length (default: 12 = 1 hour at 5-min intervals)
        pred_horizon: Prediction horizon (default: 12 = 1 hour)
        batch_size: Batch size
        num_workers: Dataloader workers

    Returns:
        train_loader, val_loader, test_loader, adjacency, metadata
    """
    data_dir = Path(data_dir)

    print("=" * 60)
    print("LOADING METR-LA DATA (STAN-STYLE FORMAT)")
    print("=" * 60)

    # Check if required files exist
    required_files = ['train.csv', 'val.csv', 'test.csv', 'metadata.json']
    missing = [f for f in required_files if not (data_dir / f).exists()]

    if missing:
        print(f"\nMissing files: {missing}")
        if 'metadata.json' in missing:
            print("\nCreating metadata.json from adjacency and CSV files...")
            _create_metadata(data_dir, input_window, pred_horizon)
        else:
            raise FileNotFoundError(f"Missing required files: {missing}")

    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        meta = json.load(f)

    n_nodes = meta['n_nodes']
    mean = meta.get('mean', 58.47)
    std = meta.get('std', 12.62)

    print(f"\nDataset info:")
    print(f"  Nodes (sensors): {n_nodes}")
    print(f"  Input window: {input_window} ({input_window * 5} min)")
    print(f"  Prediction horizon: {pred_horizon} ({pred_horizon * 5} min)")
    print(f"  Mean: {mean:.2f} mph")
    print(f"  Std: {std:.2f} mph")
    print(f"  Data scale: ORIGINAL (mph)")

    # Load adjacency matrix (try .npz first, then .npy)
    if (data_dir / 'adjacency.npz').exists():
        adjacency = sparse.load_npz(data_dir / 'adjacency.npz')
        adjacency = adjacency.toarray()  # Convert to dense for compatibility
        print(f"\nLoaded adjacency from adjacency.npz: {adjacency.shape}")
    elif (data_dir / 'adjacency.npy').exists():
        adjacency = np.load(data_dir / 'adjacency.npy')
        print(f"\nLoaded adjacency from adjacency.npy: {adjacency.shape}")
    else:
        raise FileNotFoundError("No adjacency matrix found (adjacency.npz or adjacency.npy)")

    # Create datasets
    print("\nCreating PyTorch datasets...")

    print("  Train:")
    train_dataset = TrafficDataset(
        data_dir / 'train.csv',
        n_nodes=n_nodes,
        input_window=input_window,
        pred_horizon=pred_horizon,
        mean=mean,
        std=std
    )

    print("  Val:")
    val_dataset = TrafficDataset(
        data_dir / 'val.csv',
        n_nodes=n_nodes,
        input_window=input_window,
        pred_horizon=pred_horizon,
        mean=mean,
        std=std
    )

    print("  Test:")
    test_dataset = TrafficDataset(
        data_dir / 'test.csv',
        n_nodes=n_nodes,
        input_window=input_window,
        pred_horizon=pred_horizon,
        mean=mean,
        std=std
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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

    # Metadata dict
    metadata = {
        'n_nodes': n_nodes,
        'input_window': input_window,
        'pred_horizon': pred_horizon,
        'mean': mean,
        'std': std,
        'sensor_ids': meta.get('sensor_ids', None),
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset)
    }

    print("\n" + "=" * 60)
    print("DATALOADER SUMMARY")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Adjacency: {adjacency.shape}")
    print(f"Input shape: [{n_nodes}, {input_window}]")
    print(f"Target shape: [{n_nodes}, {pred_horizon}]")
    print(f"\nData is in ORIGINAL SCALE (mph) - physics parameters are meaningful!")

    return train_loader, val_loader, test_loader, adjacency, metadata


def _create_metadata(data_dir: Path, input_window: int, pred_horizon: int):
    """Create metadata.json from existing files."""
    # Determine n_nodes from adjacency
    if (data_dir / 'adjacency.npz').exists():
        adj = sparse.load_npz(data_dir / 'adjacency.npz')
        n_nodes = adj.shape[0]
    elif (data_dir / 'adjacency.npy').exists():
        adj = np.load(data_dir / 'adjacency.npy')
        n_nodes = adj.shape[0]
    else:
        raise FileNotFoundError("No adjacency matrix to determine n_nodes")

    # Compute stats from training data
    train_df = pd.read_csv(data_dir / 'train.csv')
    input_cols = [f'input_node{i}_t{j}' for i in range(n_nodes) for j in range(input_window)]
    values = train_df[input_cols].values.flatten()
    mean = float(values.mean())
    std = float(values.std())

    metadata = {
        'n_nodes': n_nodes,
        'input_window': input_window,
        'pred_horizon': pred_horizon,
        'mean': mean,
        'std': std,
        'data_scale': 'original',
        'data_unit': 'mph'
    }

    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata.json: {metadata}")


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
