"""
run_training_graphpde_multistep.py

Entry point for training GraphPDE with rolling multi-step prediction.

Enhanced with NL Framework features:
- Multi-frequency parameter group updates
- Gradient spike protection
- Warmup + cosine annealing scheduler
- Learnable learning rate controller
- Per-module learning rate scaling

Rolling Windows:
  From 2001: predicts 2006 -> 2011 -> 2016 -> 2021 (4 steps)
  From 2006: predicts 2011 -> 2016 -> 2021 (3 steps)
  From 2011: predicts 2016 -> 2021 (2 steps)
  From 2016: predicts 2021 (1 step)

Usage:
    python run_training_graphpde_multistep.py
    python run_training_graphpde_multistep.py --use_curriculum --curriculum_warmup_epochs 30
    python run_training_graphpde_multistep.py --no_spike_protection --no_warmup_scheduler
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import random
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from model_graphpde import create_graphpde_model
from trainer_graphpde_multistep import (
    TrainerGraphPDEMultistep,
    create_parameter_groups,
    compute_metrics,
    GRAPHPDE_PARAM_GROUPS,
)
from dataloader_with_city_filter import (
    ReactionDiffusionDatasetFiltered,
    create_dataloaders_with_city_filter
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GraphPDE with rolling multi-step prediction (NL Framework)'
    )

    # Data paths (relative to experiments/multistep/)
    parser.add_argument('--train_path', type=str, default='../../data/train_rd.csv',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, default='../../data/val_rd.csv',
                        help='Path to validation data')
    parser.add_argument('--test_path', type=str, default='../../data/test_rd.csv',
                        help='Path to test data')
    parser.add_argument('--graph_path', type=str, default='../../data/graph_large_cities_rd.pkl',
                        help='Path to graph structure')
    parser.add_argument('--da_csv_path', type=str, default='../../data/da_canada.csv',
                        help='Path to DA-to-city mapping CSV')

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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Base learning rate (scaled per module)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Default weight decay for optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0.0 = none)')

    # NL Framework: Gradient Spike Protection
    parser.add_argument('--no_spike_protection', action='store_true',
                        help='Disable gradient spike protection')
    parser.add_argument('--max_grad_norm', type=float, default=2000.0,
                        help='Maximum gradient norm before emergency clipping')
    parser.add_argument('--spike_threshold', type=float, default=3.0,
                        help='Spike threshold (multiple of moving average)')

    # NL Framework: Warmup Scheduler
    parser.add_argument('--no_warmup_scheduler', action='store_true',
                        help='Disable warmup + cosine annealing scheduler')
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help='Number of warmup steps')
    parser.add_argument('--total_steps', type=int, default=10000,
                        help='Total training steps for scheduler')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                        help='Minimum LR as ratio of base (for cosine)')

    # NL Framework: Learnable LR Controller
    parser.add_argument('--no_learnable_lr', action='store_true',
                        help='Disable learnable LR controller')
    parser.add_argument('--lr_update_interval', type=int, default=10,
                        help='Steps between LR updates')
    parser.add_argument('--lr_warmup_steps', type=int, default=100,
                        help='Warmup steps before LR adaptation starts')

    # Multi-step specific
    parser.add_argument('--step_loss_weights', type=float, nargs='+',
                        default=[1.0, 1.0, 1.0, 1.0],
                        help='Loss weights per prediction step (e.g., 0.5 0.75 1.0 1.0)')
    parser.add_argument('--use_curriculum', action='store_true', default=False,
                        help='Use curriculum learning (gradually increase steps)')
    parser.add_argument('--curriculum_warmup_epochs', type=int, default=20,
                        help='Epochs before reaching full multi-step (for curriculum)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')

    # City filter
    parser.add_argument('--city_filter', type=str, nargs='+',
                        default=['Toronto', 'Mississauga', 'Brampton', 'Markham',
                                 'Vaughan', 'Abbotsford', 'Winnipeg', 'Surrey',
                                 'Delta', 'Burnaby'],
                        help='Cities to include in training')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default='graphpde_multistep_nl',
                        help='Name of the experiment')

    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default='latest_checkpoint.pt',
                        help='Checkpoint file to resume from')

    # Verbosity
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbosity')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log step-level metrics')

    args = parser.parse_args()
    return args


def create_optimizer_with_param_groups(model, args, verbose=True):
    """
    Create optimizer with per-module parameter groups.

    Uses the NL Framework parameter group configuration:
    - Different LR scales per module
    - Different weight decay per module
    - Different update frequencies (handled by MultiFrequencyOptimizer)
    """
    # Create parameter groups using NL Framework configuration
    param_groups = create_parameter_groups(
        model=model,
        base_lr=args.lr,
        default_weight_decay=args.weight_decay,
        verbose=verbose
    )

    # Create AdamW optimizer with these groups
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer


def print_nl_framework_config(args):
    """Print NL Framework configuration summary."""
    print("\n" + "=" * 70)
    print("NL FRAMEWORK CONFIGURATION")
    print("=" * 70)

    print("\n[Parameter Groups]")
    for config in GRAPHPDE_PARAM_GROUPS:
        effective_lr = args.lr * config['lr_scale']
        print(f"  {config['name']:20s}: lr_scale={config['lr_scale']:.2f} "
              f"(effective={effective_lr:.6f}), freq={config['update_freq']}, "
              f"wd={config['weight_decay']}")

    print("\n[Gradient Protection]")
    if args.no_spike_protection:
        print("  DISABLED")
    else:
        print(f"  Spike threshold: {args.spike_threshold}x moving average")
        print(f"  Max grad norm: {args.max_grad_norm}")

    print("\n[Warmup Scheduler]")
    if args.no_warmup_scheduler:
        print("  DISABLED")
    else:
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Total steps: {args.total_steps}")
        print(f"  Min LR ratio: {args.min_lr_ratio}")

    print("\n[Learnable LR Controller]")
    if args.no_learnable_lr:
        print("  DISABLED")
    else:
        print(f"  Update interval: {args.lr_update_interval} steps")
        print(f"  Warmup steps: {args.lr_warmup_steps}")

    print("=" * 70 + "\n")


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    verbose = not args.quiet

    if verbose:
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

    # We need the datasets directly (not just dataloaders) for multi-step training
    # because we need access to the full_graph_cache
    if verbose:
        print("\nCreating training dataset...")
    train_dataset = ReactionDiffusionDatasetFiltered(
        samples_path=str(args.train_path),
        graph_path=str(args.graph_path),
        ethnicity_filter=None,
        census_stats=None,
        is_training=True,
        city_filter=args.city_filter,
        da_csv_path=str(args.da_csv_path)
    )

    census_stats = train_dataset.get_census_stats()

    if verbose:
        print("\nCreating validation dataset...")
    val_dataset = ReactionDiffusionDatasetFiltered(
        samples_path=str(args.val_path),
        graph_path=str(args.graph_path),
        ethnicity_filter=None,
        census_stats=census_stats,
        is_training=False,
        city_filter=args.city_filter,
        da_csv_path=str(args.da_csv_path)
    )

    if verbose:
        print("\nCreating test dataset...")
    test_dataset = ReactionDiffusionDatasetFiltered(
        samples_path=str(args.test_path),
        graph_path=str(args.graph_path),
        ethnicity_filter=None,
        census_stats=census_stats,
        is_training=False,
        city_filter=args.city_filter,
        da_csv_path=str(args.da_csv_path)
    )

    # Use the graph from the training dataset (which has been filtered)
    graph = train_dataset.graph

    # =========================================================================
    # GET DATA STATISTICS
    # =========================================================================
    adjacency = graph['adjacency']
    n_nodes = adjacency.shape[0]

    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])
    n_cities = len(unique_cities)

    n_ethnicities = train_dataset.n_ethnicities

    # Get census feature count from the cache
    sample_year = list(train_dataset.full_graph_cache.keys())[0]
    n_census_features = train_dataset.full_graph_cache[sample_year]['census'].shape[1]

    if verbose:
        print(f"\nData statistics:")
        print(f"  Training samples: {len(train_dataset):,}")
        print(f"  Validation samples: {len(val_dataset):,}")
        print(f"  Test samples: {len(test_dataset):,}")
        print(f"  Nodes: {n_nodes:,}")
        print(f"  Cities: {n_cities}")
        print(f"  Ethnicities: {n_ethnicities}")
        print(f"  Census features: {n_census_features}")
        print(f"  Training years cached: {sorted(train_dataset.full_graph_cache.keys())}")
        print(f"  Validation years cached: {sorted(val_dataset.full_graph_cache.keys())}")

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("CREATING GRAPHPDE MODEL")
        print("=" * 80)

    model = create_graphpde_model(
        n_ethnicities=n_ethnicities,
        n_cities=n_cities,
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

    if verbose:
        print(f"\nModel: GraphPDE (Physics-Informed Reaction-Diffusion)")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")
        print(f"  Model size: {n_params * 4 / 1024 / 1024:.2f} MB (float32)")

        print("\nPhysics Configuration:")
        print(f"  ODE integration steps: {args.n_ode_steps}")
        print(f"  Integration time: {args.integration_time} years")
        print(f"  Custom CUDA kernels: {args.use_cuda_kernels}")

        print("\nMulti-step Configuration:")
        print(f"  Step loss weights: {args.step_loss_weights}")
        print(f"  Curriculum learning: {args.use_curriculum}")
        if args.use_curriculum:
            print(f"  Curriculum warmup: {args.curriculum_warmup_epochs} epochs")

    # =========================================================================
    # CREATE OPTIMIZER WITH PARAMETER GROUPS
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("CREATING OPTIMIZER WITH PARAMETER GROUPS")
        print("=" * 80)

    optimizer = create_optimizer_with_param_groups(model, args, verbose=verbose)

    if verbose:
        print(f"\nOptimizer: AdamW with {len(optimizer.param_groups)} parameter groups")
        print(f"  Base learning rate: {args.lr}")
        print(f"  Default weight decay: {args.weight_decay}")

    # Print NL Framework configuration
    if verbose:
        print_nl_framework_config(args)

    # =========================================================================
    # CREATE TRAINER WITH NL FRAMEWORK
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("CREATING MULTI-STEP TRAINER (NL Framework)")
        print("=" * 80)

    trainer = TrainerGraphPDEMultistep(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        graph=graph,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        # Gradient protection
        use_spike_protection=not args.no_spike_protection,
        max_grad_norm=args.max_grad_norm,
        spike_threshold=args.spike_threshold,
        # Warmup scheduler
        use_warmup_scheduler=not args.no_warmup_scheduler,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        min_lr_ratio=args.min_lr_ratio,
        # Learnable LR
        use_learnable_lr=not args.no_learnable_lr,
        lr_update_interval=args.lr_update_interval,
        lr_warmup_steps=args.lr_warmup_steps,
        # Loss
        label_smoothing=args.label_smoothing,
        step_loss_weights=args.step_loss_weights,
        # Curriculum
        use_curriculum=args.use_curriculum,
        curriculum_warmup_epochs=args.curriculum_warmup_epochs,
        # Logging
        verbose=verbose,
        log_interval=args.log_interval,
    )

    # Check if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_path = checkpoint_dir / args.resume_from
        if checkpoint_path.exists():
            if verbose:
                print(f"\n{'=' * 80}")
                print("RESUMING FROM CHECKPOINT")
                print("=" * 80)
            start_epoch = trainer.load_checkpoint(args.resume_from)
            if verbose:
                print(f"Will continue training from epoch {start_epoch}")
        else:
            print(f"\n  Warning: Checkpoint {checkpoint_path} not found!")
            print("Starting training from scratch...")
            start_epoch = 0

    # =========================================================================
    # TRAIN
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING (MULTI-STEP ROLLING + NL FRAMEWORK)")
        print("=" * 80)

    trainer.train(
        n_epochs=args.epochs,
        early_stopping_patience=args.patience,
        start_epoch=start_epoch
    )

    # =========================================================================
    # EVALUATE ON TEST SET
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATING ON TEST SET (Multi-step)")
        print("=" * 80)

    # Load best model
    trainer.load_checkpoint('best_loss_model.pt')

    # Prepare test populations and census
    test_populations = {}
    test_census = {}

    for year in [2001, 2006, 2011, 2016]:
        if year in test_dataset.full_graph_cache:
            test_populations[year] = test_dataset.full_graph_cache[year]['population'].to(device)
            test_census[year] = test_dataset.full_graph_cache[year]['census'].to(device)

    # Build 2021 ground truth for test (pop_t1 from 2016)
    pop_2021_test = test_dataset.get_population_for_year(2016, return_t1=True)
    test_populations[2021] = pop_2021_test.to(device)

    # Run multi-step evaluation
    model.eval()
    with torch.no_grad():
        test_loss, test_step_losses, test_preds, test_tgts = trainer._rolling_multistep_pass(
            test_populations, test_census, max_steps=4
        )

    if test_preds and test_tgts:
        all_test_preds = torch.cat(test_preds)
        all_test_tgts = torch.cat(test_tgts)

        test_metrics = compute_metrics(all_test_preds, all_test_tgts)
        test_metrics['loss'] = test_loss.item()

        print(f"\nTest Set Results (Multi-step Rolling + NL Framework):")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.2f}")
        print(f"  RMSE: {test_metrics['rmse']:.2f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  R2: {test_metrics['r2']:.4f}")
        print(f"  Median AE: {test_metrics['median_ae']:.2f}")

        print(f"\nPer-step test losses:")
        for key, val in sorted(test_step_losses.items()):
            print(f"  {key}: {val:.4f}")

        # Save test results
        import csv
        test_results_csv = checkpoint_dir / 'test_results_multistep.csv'
        with open(test_results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in test_metrics.items():
                writer.writerow([key, value])
        print(f"\nTest results saved to: {test_results_csv}")

    # =========================================================================
    # SAVE PHYSICS PARAMETERS
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("SAVING INTERPRETABLE PHYSICS PARAMETERS")
        print("=" * 80)

    physics_params = model.get_parameters_dict()

    import pickle
    params_path = checkpoint_dir / 'physics_parameters.pkl'
    with open(params_path, 'wb') as f:
        pickle.dump(physics_params, f)

    print(f"Physics parameters saved to: {params_path}")

    # Print spike protector stats if enabled
    if trainer.spike_protector is not None:
        stats = trainer.spike_protector.get_stats()
        print(f"\n[Gradient Spike Protection Stats]")
        print(f"  Total spikes detected: {stats['spike_count']}")
        print(f"  Total times protected: {stats['protected_count']}")
        print(f"  Average grad norm: {stats['avg_grad_norm']:.1f}")
        print(f"  Max grad norm: {stats['max_grad_norm']:.1f}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model (loss): {checkpoint_dir / 'best_loss_model.pt'}")
    print(f"Best model (R2): {checkpoint_dir / 'best_r2_model.pt'}")
    print(f"Metrics CSV: {checkpoint_dir / 'metrics.csv'}")
    print(f"Step metrics CSV: {checkpoint_dir / 'step_metrics.csv'}")
    print(f"Physics parameters: {checkpoint_dir / 'physics_parameters.pkl'}")


if __name__ == "__main__":
    main()
