"""
run_training_dcrnn.py

Training script for DCRNN baseline - matches GraphPDE training script exactly.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import random
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from dataloader_with_city_filter import create_dataloaders_with_city_filter

# Import DCRNN modules
from model_dcrnn import create_dcrnn_model
from trainer_dcrnn import TrainerDCRNN


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
        description='Train DCRNN baseline for ethnicity population prediction'
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
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for DCRNN')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of DCGRU layers')
    parser.add_argument('--K', type=int, default=2,
                        help='Diffusion steps in graph convolution')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping threshold')
    
    # Scheduler and early stopping
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=8,
                        help='LR scheduler patience (for plateau)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR reduction factor')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (always cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='dcrnn_baseline',
                        help='Name of the experiment')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default='latest_checkpoint.pt',
                        help='Checkpoint file to resume from')
    
    args = parser.parse_args()
    return args


def create_optimizer(model, args):
    """Create optimizer"""
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    return optimizer


def create_scheduler(optimizer, args):
    """Create learning rate scheduler"""
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=args.lr_factor
        )
    else:
        scheduler = None
    
    return scheduler


def get_data_statistics(train_loader, graph):
    """Get statistics from the data"""
    # Get number of unique cities directly from graph
    n_cities = len(set(graph['node_features']['city']))

    # Get number of unique ethnicities from dataset
    train_dataset = train_loader.dataset
    n_ethnicities = train_dataset.n_ethnicities

    # Get number of DAUIDs (nodes in graph)
    n_dauids = graph['adjacency'].shape[0]

    return n_ethnicities, n_cities, n_dauids


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device (always CUDA)
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    cities_above_100 = [
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
        city_filter=cities_above_100
    )
    
    # Get data statistics
    n_ethnicities, n_cities, n_dauids = get_data_statistics(train_loader, graph)
    
    print(f"\nData statistics:")
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Validation samples: {len(val_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Ethnicities: {n_ethnicities}")
    print(f"  Cities: {n_cities}")
    print(f"  DAUID nodes: {n_dauids:,}")
    
    # Get feature dimensions
    sample_batch = next(iter(train_loader))
    n_total_features = sample_batch['features'].shape[1]
    n_census_features = 74
    
    print(f"  Census features: {n_census_features}")
    print(f"  Other ethnicity features: {n_total_features - n_census_features}")
    
    # Prepare adjacency and node_to_city mapping
    adjacency = graph['adjacency']
    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])
    
    # Create model
    print("\n" + "="*80)
    print("CREATING DCRNN MODEL")
    print("="*80)
    
    model = create_dcrnn_model(
        n_ethnicities=n_ethnicities,
        n_cities=n_cities,
        n_dauids=n_dauids,
        n_census_features=n_census_features,
        adjacency=adjacency,
        node_to_city=node_to_city,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        K=args.K,
        multi_ethnicity=True
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: DCRNN (Data-Driven Spatiotemporal Baseline)")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Model size: {n_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\nModel Configuration:")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Diffusion steps (K): {args.K}")
    print(f"  Multi-ethnicity heads: Yes ({n_ethnicities} separate heads)")
    
    print("\nKey Differences from GraphPDE:")
    print("  ✗ No physics equations (pure data-driven)")
    print("  ✗ No interpretable parameters (black box)")
    print("  ✓ Learns diffusion from data")
    print("  ✓ Recurrent architecture for temporal dynamics")
    print("  ✓ Graph structure for spatial relationships")
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    print(f"\nOptimizer: Adam")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, args)
    if scheduler is not None:
        print(f"  Scheduler: {args.scheduler}")
    
    # Create trainer
    print("\n" + "="*80)
    print("CREATING TRAINER")
    print("="*80)
    
    trainer = TrainerDCRNN(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        graph=graph,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip=args.grad_clip
    )
    
    # Check if resuming training
    start_epoch = 0
    if args.resume:
        checkpoint_path = checkpoint_dir / args.resume_from
        if checkpoint_path.exists():
            print(f"\n{'='*80}")
            print("RESUMING FROM CHECKPOINT")
            print("="*80)
            start_epoch = trainer.load_checkpoint(args.resume_from, scheduler)
            print(f"Will continue training from epoch {start_epoch}")
        else:
            print(f"\n⚠️ Warning: Checkpoint {checkpoint_path} not found!")
            print("Starting training from scratch...")
            start_epoch = 0
    
    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    trainer.train(
        n_epochs=args.epochs,
        scheduler=scheduler,
        early_stopping_patience=args.patience,
        start_epoch=start_epoch
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Switch to test loader for evaluation
    trainer.val_loader = test_loader
    
    test_metrics = trainer.validate()
    
    print(f"\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  Median AE: {test_metrics['median_ae']:.2f}")
    
    # Save test results to CSV
    test_results_csv = checkpoint_dir / 'test_results.csv'
    with open(test_results_csv, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in test_metrics.items():
            writer.writerow([key, value])
    
    print(f"\nTest results saved to: {test_results_csv}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")
    print(f"Metrics CSV: {checkpoint_dir / 'metrics.csv'}")
    print(f"Test results: {checkpoint_dir / 'test_results.csv'}")
    
    print("\n" + "="*80)
    print("COMPARISON WITH GRAPHPDE")
    print("="*80)
    print("To compare results:")
    print(f"  DCRNN metrics: {checkpoint_dir / 'metrics.csv'}")
    print(f"  GraphPDE metrics: ./checkpoints/graphpde/metrics.csv")
    print("\nKey comparison points:")
    print("  - DCRNN: Pure data-driven, no physics")
    print("  - GraphPDE: Physics-informed with neural augmentation")
    print("  - Both use same data, graph structure, and evaluation metrics")


if __name__ == "__main__":
    main()
