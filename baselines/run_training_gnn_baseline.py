"""
run_training_gnn_baseline.py

Train GNN baseline model for multi-ethnic population prediction.
Enhanced with checkpoint resume capability.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import random
import sys
import csv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from model_gnn_baseline import create_gnn_model
from dataloader_with_city_filter import create_dataloaders_with_city_filter


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train GNN baseline for ethnicity population prediction'
    )
    
    # Data paths
    parser.add_argument('--train_path', type=str, default='../data/train_rd.pkl',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, default='../data/val_rd.pkl',
                        help='Path to validation data')
    parser.add_argument('--test_path', type=str, default='../data/test_rd.pkl',
                        help='Path to test data')
    parser.add_argument('--graph_path', type=str, default='../data/graph_large_cities_rd.pkl',
                        help='Path to graph structure')
    
    # Model hyperparameters
    parser.add_argument('--feature_encoder_dim', type=int, default=128,
                        help='Feature encoder hidden dimension')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help='GNN hidden dimension')
    parser.add_argument('--gnn_num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--gnn_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--ethnicity_embed_dim', type=int, default=32,
                        help='Ethnicity embedding dimension')
    parser.add_argument('--predictor_hidden_dim', type=int, default=128,
                        help='Predictor hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # Scheduler and early stopping
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='LR scheduler patience')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR reduction factor')
    
    # Resume capability
    parser.add_argument('--resume', type=str, default='auto',
                        help='Resume from checkpoint: path to .pt file, "latest", "best", or "auto"')
    parser.add_argument('--resume_training_only', action='store_true',
                        help='Only resume training state (model, optimizer), not metrics history')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='gnn_baseline',
                        help='Name of the experiment')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    return args


def compute_metrics(predictions, targets):
    """Compute comprehensive evaluation metrics"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # MAPE (only for non-zero targets)
    non_zero_mask = targets > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) /
                              targets[non_zero_mask])) * 100
    else:
        mape = 0.0
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    median_ae = np.median(np.abs(predictions - targets))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_ae': median_ae
    }


def train_epoch(model, train_loader, optimizer, device, grad_clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        predictions = model(batch)
        targets = batch['target']
        
        # Loss (MSE)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * len(targets)
        all_predictions.append(predictions.detach())
        all_targets.append(targets.detach())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader.dataset)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = model(batch)
            targets = batch['target']
            
            # Loss
            loss = torch.nn.functional.mse_loss(predictions, targets)
            
            # Track metrics
            total_loss += loss.item() * len(targets)
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader.dataset)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics


def create_scheduler(optimizer, args):
    """Create learning rate scheduler"""
    if args.scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, verbose=True
        )
    elif args.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, verbose=True
        )
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_patience, gamma=args.lr_factor, verbose=True
        )
    else:
        return None


def find_checkpoint(checkpoint_dir, checkpoint_type='latest'):
    """
    Find a specific checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_type: 'latest', 'best', or a specific path
    
    Returns:
        Path to checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if checkpoint_type == 'latest':
        checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
        if checkpoint_path.exists():
            return checkpoint_path
    elif checkpoint_type == 'best':
        checkpoint_path = checkpoint_dir / 'best_model.pt'
        if checkpoint_path.exists():
            return checkpoint_path
    else:
        checkpoint_path = Path(checkpoint_type)
        if checkpoint_path.exists():
            return checkpoint_path
    
    return None


def save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics,
                   best_val_loss, checkpoint_path, args, metrics_history=None):
    """Save model checkpoint with all training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'args': vars(args),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if metrics_history is not None:
        checkpoint['metrics_history'] = metrics_history
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, args):
    """
    Load model checkpoint and restore training state.
    
    Returns:
        Dictionary with resume information
    """
    print(f"\n{'='*80}")
    print("LOADING CHECKPOINT")
    print(f"{'='*80}")
    print(f"Path: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model state loaded")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✓ Optimizer state loaded")
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Scheduler state loaded")
    
    # Extract resume info
    resume_info = {
        'start_epoch': checkpoint['epoch'] + 1,
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'metrics_history': checkpoint.get('metrics_history', None),
        'train_metrics': checkpoint.get('train_metrics', {}),
        'val_metrics': checkpoint.get('val_metrics', {})
    }
    
    print(f"\nCheckpoint info:")
    print(f"  Last completed epoch: {checkpoint['epoch']}")
    print(f"  Resuming from epoch: {resume_info['start_epoch']}")
    print(f"  Best val loss so far: {resume_info['best_val_loss']:.4f}")
    
    if 'train_metrics' in checkpoint and checkpoint['train_metrics']:
        print(f"  Last train loss: {checkpoint['train_metrics'].get('loss', 'N/A'):.4f}")
    if 'val_metrics' in checkpoint and checkpoint['val_metrics']:
        print(f"  Last val loss: {checkpoint['val_metrics'].get('loss', 'N/A'):.4f}")
    
    print(f"{'='*80}\n")
    
    return resume_info


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    GTA_CITIES = [
        'Toronto', 'Mississauga', 'Brampton', 'Markham', 'Vaughan',
        'Richmond Hill', 'Oakville', 'Oshawa', 'Whitby', 'Hamilton'
    ]
    
    train_loader, val_loader, test_loader, graph = create_dataloaders_with_city_filter(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ethnicity_filter=None,
        city_filter=GTA_CITIES
    )
    
    # Get data statistics
    train_dataset = train_loader.dataset
    n_ethnicities = train_dataset.n_ethnicities
    n_census_features = len(train_dataset.census_cols)
    
    print(f"\nData statistics:")
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Validation samples: {len(val_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Ethnicities: {n_ethnicities}")
    print(f"  Census features: {n_census_features}")
    print(f"  Graph nodes: {graph['adjacency'].shape[0]:,}")
    print(f"  Graph edges: {graph['adjacency'].nnz:,}")
    
    # Check for checkpoint to resume from
    resume_checkpoint = None
    if args.resume:
        if args.resume == 'auto':
            # Try to find latest checkpoint
            resume_checkpoint = find_checkpoint(checkpoint_dir, 'latest')
            if not resume_checkpoint:
                resume_checkpoint = find_checkpoint(checkpoint_dir, 'best')
            if resume_checkpoint:
                print(f"\nAuto-detected checkpoint: {resume_checkpoint}")
            else:
                print("\nNo checkpoint found, starting fresh training")
        elif args.resume in ['latest', 'best']:
            resume_checkpoint = find_checkpoint(checkpoint_dir, args.resume)
            if resume_checkpoint:
                print(f"\nFound {args.resume} checkpoint: {resume_checkpoint}")
            else:
                print(f"\n{args.resume.capitalize()} checkpoint not found, starting fresh training")
        else:
            # Specific path provided
            resume_checkpoint = Path(args.resume)
            if not resume_checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")
    
    # Create model
    print("\n" + "="*80)
    print("CREATING GNN MODEL" if not resume_checkpoint else "CREATING GNN MODEL (will load from checkpoint)")
    print("="*80)
    
    model = create_gnn_model(
        n_census_features=n_census_features,
        n_ethnicities=n_ethnicities,
        adjacency=graph['adjacency'],
        feature_encoder_dim=args.feature_encoder_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_heads=args.gnn_heads,
        ethnicity_embed_dim=args.ethnicity_embed_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: GNN Baseline (Graph Attention Network)")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Model size: {n_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print(f"\nArchitecture:")
    print(f"  Feature encoder: {n_census_features + 1} → {args.feature_encoder_dim}")
    print(f"    (census features + aggregated population signal)")
    print(f"  GNN layers: {args.gnn_num_layers} × GAT (dim={args.gnn_hidden_dim}, heads={args.gnn_heads})")
    print(f"  Ethnicity embedding: {n_ethnicities} → {args.ethnicity_embed_dim}")
    print(f"  Predictor: {args.gnn_hidden_dim + args.ethnicity_embed_dim} → 1")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f"\nOptimizer: AdamW")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, args)
    if scheduler is not None:
        print(f"  Scheduler: {args.scheduler}")
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = []
    
    # Load checkpoint if resuming
    if resume_checkpoint:
        resume_info = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, device, args
        )
        
        start_epoch = resume_info['start_epoch']
        best_val_loss = resume_info['best_val_loss']
        
        # Restore metrics history if not training_only mode
        if not args.resume_training_only and resume_info['metrics_history'] is not None:
            metrics_history = resume_info['metrics_history']
            print(f"✓ Restored {len(metrics_history)} epochs of metrics history")
        else:
            print("Starting fresh metrics history")
    
    # Training loop
    print("\n" + "="*80)
    print(f"TRAINING (Epochs {start_epoch} to {args.epochs})")
    print("="*80)
    
    # Create or append to metrics CSV
    metrics_csv = checkpoint_dir / 'metrics.csv'
    if start_epoch == 0:
        # Fresh start - create new CSV
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_rmse', 'train_r2', 
                            'val_loss', 'val_mae', 'val_rmse', 'val_r2'])
    else:
        print(f"Appending metrics to: {metrics_csv}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Save to metrics history
        metrics_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, "
              f"RMSE: {train_metrics['rmse']:.2f}, R²: {train_metrics['r2']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, "
              f"RMSE: {val_metrics['rmse']:.2f}, R²: {val_metrics['r2']:.4f}")
        
        # Save metrics to CSV
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['loss'], train_metrics['mae'], train_metrics['rmse'], train_metrics['r2'],
                val_metrics['loss'], val_metrics['mae'], val_metrics['rmse'], val_metrics['r2']
            ])
        
        # Learning rate scheduling
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Check if best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics, val_metrics,
                best_val_loss, checkpoint_dir / 'best_model.pt', args, metrics_history
            )
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_metrics, val_metrics,
            best_val_loss, checkpoint_dir / 'latest_checkpoint.pt', args, metrics_history
        )
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            periodic_checkpoint = checkpoint_dir / f'checkpoint_epoch_{epoch+1:04d}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics, val_metrics,
                best_val_loss, periodic_checkpoint, args, metrics_history
            )
            print(f"✓ Saved periodic checkpoint: {periodic_checkpoint.name}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    print(f"Loaded best model from epoch {best_epoch+1}")
    
    # Test
    test_metrics = validate(model, test_loader, device)
    
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  Median AE: {test_metrics['median_ae']:.2f}")
    
    # Save test results
    test_results_csv = checkpoint_dir / 'test_results.csv'
    with open(test_results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in test_metrics.items():
            writer.writerow([key, value])
    
    print(f"\nTest results saved to: {test_results_csv}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")
    print(f"Latest checkpoint: {checkpoint_dir / 'latest_checkpoint.pt'}")
    print(f"Training metrics: {checkpoint_dir / 'metrics.csv'}")
    print(f"Test results: {checkpoint_dir / 'test_results.csv'}")


if __name__ == "__main__":
    main()