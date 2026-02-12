"""
run_training_v2.py

Training script with improved trainer (v2).
Uses warmup scheduler, gradient spike protection, and parameter groups.

Usage:
    python run_training_v2.py --epochs 200
    python run_training_v2.py --epochs 200 --lr 0.002 --warmup-steps 1000

Resume training:
    python run_training_v2.py --epochs 300 --resume
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from dataloader_traffic import create_dataloaders
from model_traffic_clean import create_model
from trainer_traffic_v2 import create_trainer_v2


def evaluate_checkpoint(trainer, test_loader, checkpoint_path, model, device):
    """
    Evaluate a single checkpoint on test set.

    Returns:
        Dictionary with metrics and checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Swap val_loader with test_loader temporarily
    original_val_loader = trainer.val_loader
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    trainer.val_loader = original_val_loader

    # Get physics parameters
    physics_params = {}
    if hasattr(model, 'get_parameters_dict'):
        physics_params = model.get_parameters_dict()

    return {
        'checkpoint': str(checkpoint_path.name),
        'epoch': checkpoint.get('epoch', -1),
        'metrics': {
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'mape': float(test_metrics['mape']),
            'r2': float(test_metrics['r2']),
            'rmse_h3': float(test_metrics.get('rmse_h3', 0)),
            'mae_h3': float(test_metrics.get('mae_h3', 0)),
            'mape_h3': float(test_metrics.get('mape_h3', 0)),
            'rmse_h6': float(test_metrics.get('rmse_h6', 0)),
            'mae_h6': float(test_metrics.get('mae_h6', 0)),
            'mape_h6': float(test_metrics.get('mape_h6', 0)),
            'rmse_h12': float(test_metrics.get('rmse_h12', 0)),
            'mae_h12': float(test_metrics.get('mae_h12', 0)),
            'mape_h12': float(test_metrics.get('mape_h12', 0)),
        },
        'physics_params': {
            'free_flow_speed': float(physics_params.get('free_flow_speed', 0)),
            'jam_density': float(physics_params.get('jam_density', 0)),
            'relaxation_rate': float(physics_params.get('relaxation_rate', 0)),
            'diffusion_coef': float(physics_params.get('diffusion_coef', 0)),
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train GraphPDE Traffic Model (V2)')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n-ode-steps', type=int, default=4, help='ODE integration steps')
    parser.add_argument('--ode-method', type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument('--no-attention', action='store_true', help='Disable sensor attention')

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
    print("GRAPHPDE TRAFFIC - V2 TRAINER")
    print("=" * 70)
    print("Improvements: Warmup scheduler, spike protection, parameter groups")
    print("-" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"ODE: {args.ode_method} ({args.n_ode_steps} steps)")
    print(f"Sensor attention: {not args.no_attention}")
    print("=" * 70)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[1/3] Loading data...")

    train_loader, val_loader, test_loader, adjacency, metadata = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    print(f"  Nodes: {metadata['n_nodes']}")
    print(f"  Input window: {metadata['input_window']} ({metadata['input_window'] * 5} min)")
    print(f"  Prediction horizon: {metadata['pred_horizon']} ({metadata['pred_horizon'] * 5} min)")
    print(f"  Train samples: {metadata['n_train']}")
    print(f"  Val samples: {metadata['n_val']}")
    print(f"  Test samples: {metadata['n_test']}")

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print("\n[2/3] Creating model...")

    model = create_model(
        n_nodes=metadata['n_nodes'],
        adjacency=adjacency,
        input_window=metadata['input_window'],
        pred_horizon=metadata['pred_horizon'],
        hidden_dim=args.hidden_dim,
        n_ode_steps=args.n_ode_steps,
        ode_method=args.ode_method,
        data_mean=metadata['mean'],
        data_std=metadata['std'],
        use_sensor_attention=not args.no_attention,
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
    # TEST EVALUATION - ALL CHECKPOINTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION - ALL CHECKPOINTS")
    print("=" * 70)

    checkpoint_dir = Path(args.checkpoint_dir)
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'n_ode_steps': args.n_ode_steps,
            'ode_method': args.ode_method,
            'use_sensor_attention': not args.no_attention,
            'n_params': n_params,
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
        },
        'data_info': {
            'n_nodes': metadata['n_nodes'],
            'input_window': metadata['input_window'],
            'pred_horizon': metadata['pred_horizon'],
            'n_train': metadata['n_train'],
            'n_val': metadata['n_val'],
            'n_test': metadata['n_test'],
        },
        'training_results': {
            'best_val_mape': results['best_val_mape'],
            'best_epoch': results['best_epoch'],
            'final_epoch': results['final_epoch'],
        },
        'checkpoints': {},
        'baseline_comparison': {
            'D2STGNN': {
                'mape_h3': 6.48,
                'mape_h6': 8.15,
                'mape_h12': 10.03,
            }
        }
    }

    # Evaluate each checkpoint
    checkpoints_to_evaluate = [
        ('best_model', checkpoint_dir / 'best_model.pt'),
        ('final_model', checkpoint_dir / 'final_model.pt'),
        ('last_checkpoint', checkpoint_dir / 'last_checkpoint.pt'),
    ]

    for name, ckpt_path in checkpoints_to_evaluate:
        if ckpt_path.exists():
            print(f"\nEvaluating {name}...")
            result = evaluate_checkpoint(trainer, test_loader, ckpt_path, model, args.device)
            evaluation_results['checkpoints'][name] = result

            metrics = result['metrics']
            print(f"  {name} (epoch {result['epoch']}):")
            print(f"    Overall: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
            print(f"    H3 (15min): RMSE={metrics['rmse_h3']:.2f}, MAE={metrics['mae_h3']:.2f}, MAPE={metrics['mape_h3']:.2f}%")
            print(f"    H6 (30min): RMSE={metrics['rmse_h6']:.2f}, MAE={metrics['mae_h6']:.2f}, MAPE={metrics['mape_h6']:.2f}%")
            print(f"    H12 (60min): RMSE={metrics['rmse_h12']:.2f}, MAE={metrics['mae_h12']:.2f}, MAPE={metrics['mape_h12']:.2f}%")
        else:
            print(f"\n  {name}: checkpoint not found at {ckpt_path}")

    # Save evaluation results to JSON
    results_json_path = checkpoint_dir / 'evaluation_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_json_path}")

    # Print comparison with baseline
    print("\n" + "-" * 70)
    print("COMPARISON WITH D2STGNN BASELINE")
    print("-" * 70)
    print("Metric        | D2STGNN | Best Model | Difference")
    print("-" * 50)

    if 'best_model' in evaluation_results['checkpoints']:
        best = evaluation_results['checkpoints']['best_model']['metrics']
        d2stgnn = evaluation_results['baseline_comparison']['D2STGNN']

        for horizon, h_name in [('h3', '15min'), ('h6', '30min'), ('h12', '60min')]:
            d2_mape = d2stgnn[f'mape_{horizon}']
            our_mape = best[f'mape_{horizon}']
            diff = our_mape - d2_mape
            status = "✓ better" if diff < 0 else "worse"
            print(f"MAPE {h_name:6s} | {d2_mape:6.2f}% | {our_mape:9.2f}% | {diff:+.2f}% ({status})")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation MAPE: {results['best_val_mape']:.2f}% (epoch {results['best_epoch']})")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"Evaluation JSON saved in: {results_json_path}")

    return results


if __name__ == '__main__':
    main()
