"""
run_training_v2.py

Training script for GraphPDE Epidemiology model (V2).
Uses warmup scheduler, gradient spike protection, and parameter groups.

Usage:
    python run_training_v2.py --epochs 200
    python run_training_v2.py --epochs 200 --lr 0.002 --warmup-steps 200

Resume training:
    python run_training_v2.py --epochs 300 --resume
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from dataloader_sir import create_dataloaders
from model_epidemiology_clean import create_model
from trainer_epidemiology_v2 import create_trainer_v2


def main():
    parser = argparse.ArgumentParser(description='Train GraphPDE Epidemiology Model (V2)')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=200, help='Warmup steps')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-ode-steps', type=int, default=4, help='ODE integration steps')
    parser.add_argument('--ode-method', type=str, default='euler', choices=['euler', 'rk4'])

    # Paths
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_v2', help='Checkpoint directory')

    # Other
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--early-stopping', type=int, default=30, help='Early stopping patience')

    # Resume training
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 70)
    print("GRAPHPDE EPIDEMIOLOGY - V2 TRAINER")
    print("=" * 70)
    print("Improvements: Warmup scheduler, spike protection, parameter groups")
    print("NO CMS, NO SMM, NO M3")
    print("-" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"ODE: {args.ode_method} ({args.n_ode_steps} steps)")
    print("=" * 70)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[1/3] Loading data...")

    train_loader, val_loader, test_loader, adjacency, metadata = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    n_nodes = metadata['n_nodes']
    input_window = metadata['input_window']
    pred_horizon = metadata['pred_horizon']

    print(f"  Nodes: {n_nodes}")
    print(f"  Input window: {input_window}")
    print(f"  Prediction horizon: {pred_horizon}")

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print("\n[2/3] Creating model...")

    model = create_model(
        n_nodes=n_nodes,
        adjacency=adjacency,
        input_window=input_window,
        pred_horizon=pred_horizon,
        hidden_dim=args.hidden_dim,
        n_ode_steps=args.n_ode_steps,
        ode_method=args.ode_method,
        data_mean=metadata.get('mean', 0.0),
        data_std=metadata.get('std', 1.0),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # =========================================================================
    # TRAIN
    # =========================================================================
    print("\n[3/3] Training...")

    trainer = create_trainer_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume from checkpoint if requested
    resume = False
    if args.resume or args.resume_from:
        checkpoint_path = args.resume_from if args.resume_from else Path(args.checkpoint_dir) / 'last_checkpoint.pt'
        if Path(checkpoint_path).exists():
            trainer.load_checkpoint(str(checkpoint_path), resume_training=True)
            resume = True
            print(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, starting fresh")

    results = trainer.train(
        n_epochs=args.epochs,
        val_interval=1,
        early_stopping_patience=args.early_stopping,
        save_best=True,
        resume=resume,
    )

    # =========================================================================
    # TEST EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    # Load best model
    best_checkpoint = Path(args.checkpoint_dir) / 'best_model.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        print("Loaded best model for testing")

    # Swap val_loader with test_loader
    original_val_loader = trainer.val_loader
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    trainer.val_loader = original_val_loader

    print(f"\nTest Results:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  R²:   {test_metrics['r2']:.4f}")

    # Physics parameters
    print("\n  Learned Physics Parameters:")
    params = model.get_parameters_dict()
    print(f"    Growth rate: {params.get('growth_rate', 0):.4f}")
    print(f"    Decay rate: {params.get('decay_rate', 0):.4f}")
    print(f"    Capacity: {params.get('capacity', 0):.2f}")
    print(f"    Diffusion coef: {params.get('diffusion_coef', 0):.4f}")
    print(f"    Physics weight: {params.get('physics_weight', 0):.2f}")
    print(f"    Neural weight: {params.get('neural_weight', 0):.2f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation MAE: {results['best_val_mae']:.4f} (epoch {results['best_epoch']})")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")

    return results


if __name__ == '__main__':
    main()
