"""
baseline_gcn.py

Graph Convolutional Network (GCN) baseline for ethnicity population prediction.

Architecture:
- Multi-layer GCN (2-3 layers)
- Node features: census features + population history
- Graph structure: spatial adjacency
- Task: Predict population at t+1 given population at t

**Windows Users:** This script automatically sets num_workers=0 on Windows to avoid
multiprocessing issues. If you want to use multiple workers, you may need to add:
    if __name__ == '__main__':
        # your code here
at the top of your script and run with the 'spawn' method.

Usage:
    python baseline_gcn.py --epochs 200 --hidden_dim 128
    python baseline_gcn.py --resume --checkpoint best_model.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import csv
from datetime import datetime
import platform


# ============================================================================
# GCN LAYERS
# ============================================================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer
    
    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_normalized):
        """
        Args:
            x: (n_nodes, in_features)
            adj_normalized: normalized adjacency matrix (sparse or dense)
        Returns:
            (n_nodes, out_features)
        """
        # Graph convolution: A @ X @ W
        if isinstance(adj_normalized, torch.sparse.FloatTensor):
            support = torch.sparse.mm(adj_normalized, x)
        else:
            support = torch.mm(adj_normalized, x)
        
        output = self.linear(support)
        output = self.dropout(output)
        
        return output


class GCNModel(nn.Module):
    """
    Multi-layer GCN for population prediction
    
    Features per node:
    - Census features (30)
    - Current population (n_ethnicities)
    - Neighbor population features (aggregated)
    
    Output:
    - Predicted population at t+1 (1 value per node-ethnicity pair)
    """
    def __init__(self, n_census_features, n_ethnicities, hidden_dim=128, 
                 n_layers=2, dropout=0.2):
        super().__init__()
        
        self.n_census_features = n_census_features
        self.n_ethnicities = n_ethnicities
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Input: census + population + other ethnicities
        input_dim = n_census_features + n_ethnicities
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim, dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))
        
        # Output layer (predicts change in population)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        print(f"\n{'='*80}")
        print("GCN MODEL ARCHITECTURE")
        print(f"{'='*80}")
        print(f"Input features: {input_dim} (census: {n_census_features}, ethnicities: {n_ethnicities})")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Number of GCN layers: {n_layers}")
        print(f"Dropout: {dropout}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"{'='*80}\n")
    
    def forward(self, full_node_features, adj_normalized, batch_node_idx, batch_ethnicity_idx):
        """
        Args:
            full_node_features: (n_nodes, input_dim) - features for ALL nodes
            adj_normalized: normalized adjacency matrix
            batch_node_idx: (batch_size,) - node indices for this batch
            batch_ethnicity_idx: (batch_size,) - ethnicity indices for this batch
        
        Returns:
            predictions: (batch_size,) - predicted populations
        """
        # Apply GCN layers to FULL graph
        x = full_node_features
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj_normalized)
            if i < len(self.gcn_layers) - 1:  # No activation on last GCN layer
                x = F.relu(x)
        
        # Extract batch nodes
        batch_embeddings = x[batch_node_idx]  # (batch_size, hidden_dim)
        
        # Predict
        predictions = self.output(batch_embeddings).squeeze(-1)  # (batch_size,)
        
        return predictions


# ============================================================================
# DATASET
# ============================================================================

class GCNDataset(Dataset):
    """
    Dataset for GCN baseline
    
    Returns full graph features for each batch (GCN needs full graph context)
    """
    def __init__(self, samples_path, graph_path, census_stats=None, is_training=False):
        # Load samples
        if samples_path.endswith('.pkl'):
            self.samples = pd.read_pickle(samples_path)
        else:
            self.samples = pd.read_csv(samples_path)
        
        # Load graph
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        self.adjacency = self.graph['adjacency']
        
        # Identify features
        self.census_cols = [c for c in self.samples.columns if c.startswith('census_')]
        self.other_ethnic_cols = [c for c in self.samples.columns if c.startswith('other_')]
        
        # Compute normalization stats
        if is_training or census_stats is None:
            census_data = self.samples[self.census_cols].values.astype(np.float32)
            self.census_mean = np.nan_to_num(census_data.mean(axis=0), nan=0.0)
            self.census_std = np.nan_to_num(census_data.std(axis=0), nan=1.0)
            self.census_std = np.clip(self.census_std, 1.0, None)
        else:
            self.census_mean = census_stats['mean']
            self.census_std = census_stats['std']
        
        # Build ethnicity mapping
        self.ethnicity_map = {eth: i for i, eth in enumerate(sorted(self.samples['ethnicity'].unique()))}
        self.n_ethnicities = len(self.ethnicity_map)
        self.n_nodes = self.adjacency.shape[0]
        
        # Precompute full graph states
        self._precompute_full_graph_states()
        
        print(f"Loaded {len(self.samples):,} samples")
        print(f"  Ethnicities: {self.n_ethnicities}")
        print(f"  Nodes: {self.n_nodes:,}")
    
    def _precompute_full_graph_states(self):
        """Precompute features for all nodes at each year"""
        self.full_graph_cache = {}
        
        unique_years = self.samples['year_t'].unique()
        n_census = len(self.census_cols)
        input_dim = n_census + self.n_ethnicities
        
        for year in unique_years:
            year_samples = self.samples[self.samples['year_t'] == year]
            
            # Initialize tensors
            full_features = np.zeros((self.n_nodes, input_dim), dtype=np.float32)
            
            # Fill in values
            for _, row in year_samples.iterrows():
                node_idx = int(row['node_idx'])
                ethnicity_idx = self.ethnicity_map[row['ethnicity']]
                
                # Census features (normalized)
                census_values = row[self.census_cols].values.astype(np.float32)
                census_values = np.nan_to_num(census_values, nan=0.0)
                census_normalized = (census_values - self.census_mean) / self.census_std
                census_normalized = np.clip(census_normalized, -5.0, 5.0)
                
                full_features[node_idx, :n_census] = census_normalized
                
                # Population at t (current population)
                full_features[node_idx, n_census + ethnicity_idx] = float(row['pop_t'])
            
            self.full_graph_cache[year] = torch.tensor(full_features, dtype=torch.float32)
    
    def get_census_stats(self):
        return {'mean': self.census_mean, 'std': self.census_std}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        
        return {
            'node_idx': int(row['node_idx']),
            'ethnicity_idx': self.ethnicity_map[row['ethnicity']],
            'target': float(row['pop_t1']),
            'year_t': row['year_t']
        }


class GCNCollator:
    """Picklable collate function for GCN DataLoader"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __call__(self, batch):
        """Custom collate function that includes full graph state"""
        # Get year (all samples in batch should have same year_t)
        year_t = batch[0]['year_t']
        
        return {
            'node_idx': torch.tensor([b['node_idx'] for b in batch], dtype=torch.long),
            'ethnicity_idx': torch.tensor([b['ethnicity_idx'] for b in batch], dtype=torch.long),
            'target': torch.tensor([b['target'] for b in batch], dtype=torch.float32),
            'full_graph_features': self.dataset.full_graph_cache[year_t].clone()
        }


# ============================================================================
# TRAINING
# ============================================================================

def normalize_adjacency(adjacency):
    """Compute normalized adjacency: D^(-1/2) A D^(-1/2)"""
    # Convert to COO for PyTorch sparse
    coo = adjacency.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    
    adj_sparse = torch.sparse_coo_tensor(indices, values, size=adjacency.shape)
    
    # Compute degree
    degree = torch.sparse.sum(adj_sparse, dim=1).to_dense()
    degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
    
    # Normalize edges
    row, col = indices[0], indices[1]
    norm_values = values * degree_inv_sqrt[row] * degree_inv_sqrt[col]
    
    adj_normalized = torch.sparse_coo_tensor(
        indices, norm_values, size=adjacency.shape
    )
    
    return adj_normalized


def train_epoch(model, loader, adj_normalized, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        # Move to device
        node_idx = batch['node_idx'].to(device)
        ethnicity_idx = batch['ethnicity_idx'].to(device)
        target = batch['target'].to(device)
        full_features = batch['full_graph_features'].to(device)
        
        # Forward
        optimizer.zero_grad()
        predictions = model(full_features, adj_normalized, node_idx, ethnicity_idx)
        
        # Loss
        loss = F.mse_loss(predictions, target)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, adj_normalized, device):
    """Validate"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    for batch in loader:
        node_idx = batch['node_idx'].to(device)
        ethnicity_idx = batch['ethnicity_idx'].to(device)
        target = batch['target'].to(device)
        full_features = batch['full_graph_features'].to(device)
        
        predictions = model(full_features, adj_normalized, node_idx, ethnicity_idx)
        
        loss = F.mse_loss(predictions, target)
        total_loss += loss.item()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    metrics = {
        'loss': total_loss / len(loader),
        'mae': mean_absolute_error(all_targets, all_predictions),
        'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
        'r2': r2_score(all_targets, all_predictions)
    }
    
    # MAPE
    mask = all_targets != 0
    if mask.sum() > 0:
        metrics['mape'] = 100 * np.mean(np.abs((all_targets[mask] - all_predictions[mask]) / all_targets[mask]))
    else:
        metrics['mape'] = np.nan
    
    metrics['median_ae'] = np.median(np.abs(all_predictions - all_targets))
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train GCN baseline')
    
    # Data
    parser.add_argument('--train_path', type=str, default='../data/train_rd.pkl')
    parser.add_argument('--val_path', type=str, default='../data/val_rd.pkl')
    parser.add_argument('--test_path', type=str, default='../data/test_rd.pkl')
    parser.add_argument('--graph_path', type=str, default='../data/graph_large_cities_rd.pkl')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    # Windows has issues with multiprocessing, default to 0
    import platform
    default_workers = 0 if platform.system() == 'Windows' else 4
    parser.add_argument('--num_workers', type=int, default=default_workers,
                       help='Number of data loading workers (use 0 on Windows)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints_baseline')
    parser.add_argument('--experiment_name', type=str, default='gcn')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_dataset = GCNDataset(args.train_path, args.graph_path, is_training=True)
    census_stats = train_dataset.get_census_stats()
    
    val_dataset = GCNDataset(args.val_path, args.graph_path, census_stats=census_stats)
    test_dataset = GCNDataset(args.test_path, args.graph_path, census_stats=census_stats)
    
    # Create dataloaders with picklable collate function
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=GCNCollator(train_dataset),
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=GCNCollator(val_dataset),
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=GCNCollator(test_dataset),
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val: {len(val_dataset):,}")
    print(f"  Test: {len(test_dataset):,}")
    
    # Normalize adjacency
    print("\nNormalizing adjacency matrix...")
    adj_normalized = normalize_adjacency(train_dataset.adjacency)
    adj_normalized = adj_normalized.to(device)
    
    # Create model
    model = GCNModel(
        n_census_features=30,
        n_ethnicities=train_dataset.n_ethnicities,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # CSV for train/val metrics
    train_val_csv = checkpoint_dir / 'train_val_metrics.csv'
    with open(train_val_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_rmse', 
                        'val_r2', 'val_mape', 'val_median_ae', 'lr'])
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, adj_normalized, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, adj_normalized, device)
        
        # Scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val MAE: {val_metrics['mae']:.2f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.2f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")
        print(f"  Val MAPE: {val_metrics['mape']:.2f}%")
        
        # Save metrics
        with open(train_val_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_metrics['loss'], val_metrics['mae'],
                val_metrics['rmse'], val_metrics['r2'], val_metrics['mape'],
                val_metrics['median_ae'], optimizer.param_groups[0]['lr']
            ])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_r2': val_metrics['r2']
            }, checkpoint_dir / 'best_model.pt')
            
            print(f"  ✓ Best model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Test evaluation
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, adj_normalized, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  Median AE: {test_metrics['median_ae']:.2f}")
    
    # Save test results
    test_csv = checkpoint_dir / 'test_metrics.csv'
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in test_metrics.items():
            writer.writerow([key, value])
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Train/Val metrics: {train_val_csv}")
    print(f"Test metrics: {test_csv}")
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()