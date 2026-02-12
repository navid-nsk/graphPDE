"""
run_training_graphpde.py

"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import random
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_graphpde import create_graphpde_model
from trainer_graphpde import TrainerGraphPDE
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
        description='Train GraphPDE for ethnicity population prediction'
    )
    
    # Data paths
    parser.add_argument('--train_path', type=str, default='./data/train_rd.pkl',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, default='./data/val_rd.pkl',
                        help='Path to validation data')
    parser.add_argument('--test_path', type=str, default='./data/test_rd.pkl',
                        help='Path to test data')
    parser.add_argument('--graph_path', type=str, default='./data/graph_large_cities_rd.pkl',
                        help='Path to graph structure')
    
    # Model hyperparameters
    parser.add_argument('--n_ode_steps', type=int, default=1,
                        help='Number of ODE integration steps')
    parser.add_argument('--integration_time', type=float, default=5.0,
                        help='Integration time in years (5 years for census)')
    parser.add_argument('--use_cuda_kernels', action='store_true', default=True,
                        help='Use custom CUDA kernels for speed')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer (increased for regularization)')
    parser.add_argument('--grad_clip', type=float, default=500.0,
                        help='Gradient clipping threshold')
    
    # Scheduler and early stopping
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'cosine_warmup', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='LR scheduler patience (for plateau)')
    parser.add_argument('--lr_factor', type=float, default=0.7,
                        help='LR reduction factor')
    parser.add_argument('--label_smoothing', type=float, default=0.1,  
                    help='Label smoothing factor (0.0 = none, 0.1 = 10% smoothing)')
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (always cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='graphpde',
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
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
    elif args.scheduler == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
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
    
    cities_above_100 = ['Toronto', 'Mississauga', 'Brampton', 'Markham', 'Vaughan', 'Abbotsford', 'Winnipeg', 'Surrey', 'Delta', 'Burnaby']  # Customize as needed

    
    train_loader, val_loader, test_loader, graph = create_dataloaders_with_city_filter(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ethnicity_filter=None,
        city_filter=cities_above_100  # Add this line
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
    
    adjacency = graph['adjacency']
    n_nodes = adjacency.shape[0]

    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])
    n_model_cities = len(unique_cities)

    sample_batch = next(iter(train_loader))
    n_census_features = sample_batch['full_graph_census'].shape[1]
    n_ethnicities = sample_batch['full_graph_population_t'].shape[1]

    print(f"\n  Nodes: {n_nodes}")
    print(f"  Cities: {n_model_cities}")
    print(f"  Ethnicities: {n_ethnicities}")
    print(f"  Census features: {n_census_features}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\n" + "="*80)
    print("CREATING GRAPHPDE MODEL")
    print("="*80)
    
    model = create_graphpde_model(
        n_ethnicities=n_ethnicities,
        n_cities=n_model_cities,
        n_dauids=n_nodes,
        n_census_features=n_census_features,
        adjacency=adjacency,
        node_to_city=node_to_city,
        n_ode_steps=args.n_ode_steps,
        integration_time=args.integration_time,
        use_cuda_kernels=args.use_cuda_kernels
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: GraphPDE (Physics-Informed Reaction-Diffusion)")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    # Create trainer
    print("\n" + "="*80)
    print("CREATING TRAINER")
    print("="*80)
    
    trainer = TrainerGraphPDE(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        graph=graph,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing
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
    
    # Save physics parameters
    physics_params = model.get_parameters_dict()

    import pickle
    params_path = checkpoint_dir / 'physics_parameters.pkl'
    with open(params_path, 'wb') as f:
        pickle.dump(physics_params, f)

    print("\nTraining complete!")
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
