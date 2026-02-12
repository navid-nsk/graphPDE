"""
evaluate.py

Standalone evaluation script for GraphPDE Epidemiology model.
Evaluates best_model, final_model, and last_checkpoint on test set.
Saves results to JSON file.

Metrics focus: MSE ↓ | MAE ↓ | CCC ↑

Usage:
    python evaluate.py
    python evaluate.py --checkpoint-dir ./checkpoints_v2
    python evaluate.py --checkpoint-dir ./checkpoints_v2 --output results.json
"""

import argparse
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from dataloader_sir import create_dataloaders
from model_epidemiology_clean import create_model


def compute_ccc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Concordance Correlation Coefficient (CCC).

    CCC measures agreement between predictions and targets, combining
    precision (Pearson correlation) and accuracy (bias).

    CCC = (2 * ρ * σ_pred * σ_target) / (σ_pred² + σ_target² + (μ_pred - μ_target)²)

    Range: [-1, 1], where 1 is perfect agreement.
    """
    pred_flat = predictions.flatten()
    targ_flat = targets.flatten()

    # Means
    pred_mean = pred_flat.mean()
    targ_mean = targ_flat.mean()

    # Variances
    pred_var = pred_flat.var()
    targ_var = targ_flat.var()

    # Covariance
    covariance = ((pred_flat - pred_mean) * (targ_flat - targ_mean)).mean()

    # CCC
    numerator = 2 * covariance
    denominator = pred_var + targ_var + (pred_mean - targ_mean) ** 2

    ccc = (numerator / (denominator + 1e-8)).item()

    return ccc


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    data_mean: float = 0.0,
    data_std: float = 1.0,
) -> dict:
    """
    Compute epidemiology prediction metrics.

    Primary metrics: MSE ↓ | MAE ↓ | CCC ↑

    Args:
        predictions: Model predictions (normalized)
        targets: Ground truth targets (normalized)
        data_mean: Mean used for normalization (to denormalize for MSE/MAE)
        data_std: Std used for normalization (to denormalize for MSE/MAE)

    Returns:
        Dictionary with mse, mae (in raw scale), and ccc
    """
    with torch.no_grad():
        # Denormalize predictions and targets for MSE/MAE
        # raw = normalized * std + mean
        pred_raw = predictions * data_std + data_mean
        targ_raw = targets * data_std + data_mean

        pred_flat = pred_raw.flatten()
        targ_flat = targ_raw.flatten()

        # MSE (lower is better) - in raw scale
        mse = F.mse_loss(pred_flat, targ_flat).item()

        # MAE (lower is better) - in raw scale
        mae = F.l1_loss(pred_flat, targ_flat).item()

        # CCC (higher is better) - PRIMARY METRIC
        # CCC is scale-invariant, but we use raw for consistency
        ccc = compute_ccc(pred_raw, targ_raw)

        return {
            'mse': mse,
            'mae': mae,
            'ccc': ccc,
        }


def evaluate_model(model, test_loader, device, data_mean: float = 0.0, data_std: float = 1.0):
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if 'input' in batch:
                inputs = batch['input'].to(device)
            else:
                inputs = batch['features'].to(device)
            targets = batch['target'].to(device)

            output = model({'input': inputs})
            predictions = output['predictions']

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    metrics = compute_metrics(
        all_predictions, all_targets,
        data_mean=data_mean, data_std=data_std
    )
    return metrics


def evaluate_checkpoint(model, checkpoint_path, test_loader, device, data_mean: float = 0.0, data_std: float = 1.0):
    """Load checkpoint and evaluate on test set."""
    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get normalization params from model if available
    if hasattr(model, 'data_mean') and hasattr(model, 'data_std'):
        data_mean = model.data_mean.item()
        data_std = model.data_std.item()

    metrics = evaluate_model(model, test_loader, device, data_mean=data_mean, data_std=data_std)

    physics_params = {}
    if hasattr(model, 'get_parameters_dict'):
        physics_params = model.get_parameters_dict()

    return {
        'checkpoint': str(checkpoint_path.name),
        'epoch': checkpoint.get('epoch', -1),
        'metrics': {
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae']),
            'ccc': float(metrics['ccc']),
        },
        'normalization': {
            'data_mean': float(data_mean),
            'data_std': float(data_std),
        },
        'physics_params': {
            'growth_rate': float(physics_params.get('growth_rate', 0)),
            'decay_rate': float(physics_params.get('decay_rate', 0)),
            'capacity': float(physics_params.get('capacity', 0)),
            'diffusion_coef': float(physics_params.get('diffusion_coef', 0)),
            'physics_weight': float(physics_params.get('physics_weight', 0)),
            'neural_weight': float(physics_params.get('neural_weight', 0)),
        }
    }


def print_metrics(name, result):
    """Print metrics in a nice format."""
    metrics = result['metrics']
    physics = result['physics_params']

    print(f"\n{'='*60}")
    print(f"{name.upper()} (Epoch {result['epoch']})")
    print(f"{'='*60}")

    print(f"\nTest Metrics (MSE ↓ | MAE ↓ | CCC ↑):")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  CCC: {metrics['ccc']:.4f}  {'← PRIMARY METRIC' if name == 'best_model' else ''}")

    print(f"\nPhysics Parameters:")
    print(f"  Growth rate:    {physics['growth_rate']:.4f}")
    print(f"  Decay rate:     {physics['decay_rate']:.4f}")
    print(f"  Capacity:       {physics['capacity']:.2f}")
    print(f"  Diffusion coef: {physics['diffusion_coef']:.4f}")
    print(f"  Physics weight: {physics['physics_weight']:.2f}")
    print(f"  Neural weight:  {physics['neural_weight']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphPDE Epidemiology Model Checkpoints')

    # Paths
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_v2', help='Checkpoint directory')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')

    # Model config
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--n-ode-steps', type=int, default=4, help='ODE integration steps')
    parser.add_argument('--ode-method', type=str, default='euler', choices=['euler', 'rk4'])

    # Other
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    checkpoint_dir = Path(args.checkpoint_dir)

    print("=" * 70)
    print("GRAPHPDE EPIDEMIOLOGY - EVALUATION")
    print("Metrics: MSE ↓ | MAE ↓ | CCC ↑")
    print("=" * 70)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Device: {args.device}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[1/3] Loading test data...")

    _, _, test_loader, adjacency, metadata = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

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
        data_mean=metadata.get('mean', 0.0),
        data_std=metadata.get('std', 1.0),
    )
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # =========================================================================
    # EVALUATE CHECKPOINTS
    # =========================================================================
    print("\n[3/3] Evaluating checkpoints...")

    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'n_ode_steps': args.n_ode_steps,
            'ode_method': args.ode_method,
            'n_params': n_params,
        },
        'data_info': {
            'n_nodes': metadata['n_nodes'],
            'input_window': metadata['input_window'],
            'pred_horizon': metadata['pred_horizon'],
            'normalization': {
                'mean': metadata.get('mean', 0.0),
                'std': metadata.get('std', 1.0),
            },
        },
        'metrics_info': {
            'primary_metric': 'CCC (higher is better)',
            'metrics': ['MSE ↓ (raw scale)', 'MAE ↓ (raw scale)', 'CCC ↑'],
            'note': 'MSE and MAE are computed on denormalized (raw) values',
        },
        'checkpoints': {},
    }

    # Checkpoints to evaluate
    checkpoints_to_evaluate = [
        ('best_model', checkpoint_dir / 'best_model.pt'),
        ('final_model', checkpoint_dir / 'final_model.pt'),
        ('last_checkpoint', checkpoint_dir / 'last_checkpoint.pt'),
    ]

    # Get normalization parameters
    data_mean = metadata.get('mean', 0.0)
    data_std = metadata.get('std', 1.0)
    print(f"\nNormalization: mean={data_mean:.4f}, std={data_std:.4f}")

    for name, ckpt_path in checkpoints_to_evaluate:
        if ckpt_path.exists():
            result = evaluate_checkpoint(
                model, ckpt_path, test_loader, args.device,
                data_mean=data_mean, data_std=data_std
            )
            evaluation_results['checkpoints'][name] = result
            print_metrics(name, result)
        else:
            print(f"\n[WARNING] {name}: checkpoint not found at {ckpt_path}")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("Metrics: MSE ↓ | MAE ↓ | CCC ↑ (Primary)")
    print("=" * 70)

    print(f"\n{'Model':<20} {'MSE':>12} {'MAE':>12} {'CCC':>12}")
    print("-" * 58)

    best_ccc = float('-inf')
    best_model_name = None

    for name in ['best_model', 'final_model', 'last_checkpoint']:
        if name in evaluation_results['checkpoints']:
            m = evaluation_results['checkpoints'][name]['metrics']
            marker = ""
            if m['ccc'] > best_ccc:
                best_ccc = m['ccc']
                best_model_name = name
            print(f"{name:<20} {m['mse']:>12.6f} {m['mae']:>12.6f} {m['ccc']:>12.4f}")

    # Mark the best model
    print("-" * 58)
    print(f"Best CCC: {best_ccc:.4f} ({best_model_name})")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_path = Path(args.output) if args.output else checkpoint_dir / 'evaluation_results.json'

    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    return evaluation_results


if __name__ == '__main__':
    main()
