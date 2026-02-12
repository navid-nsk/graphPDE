"""
evaluate.py

Standalone evaluation script for GraphPDE Traffic model.
Evaluates best_model, final_model, and last_checkpoint on test set.
Saves results to JSON file.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint-dir ./checkpoints_v2
    python evaluate.py --checkpoint-dir ./checkpoints_v2 --output results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from dataloader_traffic import create_dataloaders
from model_traffic_clean import create_model
from trainer_traffic_v2 import compute_metrics


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.

    Returns:
        Dictionary with all metrics
    """
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

    metrics = compute_metrics(all_predictions, all_targets)
    return metrics


def evaluate_checkpoint(model, checkpoint_path, test_loader, device):
    """
    Load checkpoint and evaluate on test set.

    Returns:
        Dictionary with metrics and checkpoint info
    """
    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    # Get physics parameters
    physics_params = {}
    if hasattr(model, 'get_parameters_dict'):
        physics_params = model.get_parameters_dict()

    return {
        'checkpoint': str(checkpoint_path.name),
        'epoch': checkpoint.get('epoch', -1),
        'metrics': {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'mape': float(metrics['mape']),
            'r2': float(metrics['r2']),
            'rmse_h3': float(metrics.get('rmse_h3', 0)),
            'mae_h3': float(metrics.get('mae_h3', 0)),
            'mape_h3': float(metrics.get('mape_h3', 0)),
            'rmse_h6': float(metrics.get('rmse_h6', 0)),
            'mae_h6': float(metrics.get('mae_h6', 0)),
            'mape_h6': float(metrics.get('mape_h6', 0)),
            'rmse_h12': float(metrics.get('rmse_h12', 0)),
            'mae_h12': float(metrics.get('mae_h12', 0)),
            'mape_h12': float(metrics.get('mape_h12', 0)),
        },
        'physics_params': {
            'free_flow_speed': float(physics_params.get('free_flow_speed', 0)),
            'jam_density': float(physics_params.get('jam_density', 0)),
            'relaxation_rate': float(physics_params.get('relaxation_rate', 0)),
            'diffusion_coef': float(physics_params.get('diffusion_coef', 0)),
        }
    }


def print_metrics(name, result):
    """Print metrics in a nice format."""
    metrics = result['metrics']
    physics = result['physics_params']

    print(f"\n{'='*60}")
    print(f"{name.upper()} (Epoch {result['epoch']})")
    print(f"{'='*60}")

    print(f"\nOverall Metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f} mph")
    print(f"  MAE:  {metrics['mae']:.2f} mph")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r2']:.4f}")

    print(f"\nPer-Horizon Metrics:")
    print(f"  {'Horizon':<12} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print(f"  {'-'*40}")
    print(f"  {'15min (H3)':<12} {metrics['rmse_h3']:>8.2f} {metrics['mae_h3']:>8.2f} {metrics['mape_h3']:>7.2f}%")
    print(f"  {'30min (H6)':<12} {metrics['rmse_h6']:>8.2f} {metrics['mae_h6']:>8.2f} {metrics['mape_h6']:>7.2f}%")
    print(f"  {'60min (H12)':<12} {metrics['rmse_h12']:>8.2f} {metrics['mae_h12']:>8.2f} {metrics['mape_h12']:>7.2f}%")

    print(f"\nPhysics Parameters:")
    print(f"  Free-flow speed:  {physics['free_flow_speed']:.1f} mph")
    print(f"  Jam density:      {physics['jam_density']:.4f}")
    print(f"  Relaxation rate:  {physics['relaxation_rate']:.4f}")
    print(f"  Diffusion coef:   {physics['diffusion_coef']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphPDE Traffic Model Checkpoints')

    # Paths
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_v2', help='Checkpoint directory')
    parser.add_argument('--output', type=str, default='../../results/generalizability/evaluation_results.json', help='Output JSON file (default: checkpoint_dir/evaluation_results.json)')

    # Model config (must match training)
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n-ode-steps', type=int, default=4, help='ODE integration steps')
    parser.add_argument('--ode-method', type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument('--no-attention', action='store_true', help='Disable sensor attention')

    # Other
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    checkpoint_dir = Path(args.checkpoint_dir)

    print("=" * 70)
    print("GRAPHPDE TRAFFIC - EVALUATION")
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
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # =========================================================================
    # EVALUATE CHECKPOINTS
    # =========================================================================
    print("\n[3/3] Evaluating checkpoints...")

    # Prepare results structure
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'n_ode_steps': args.n_ode_steps,
            'ode_method': args.ode_method,
            'use_sensor_attention': not args.no_attention,
            'n_params': n_params,
        },
        'data_info': {
            'n_nodes': metadata['n_nodes'],
            'input_window': metadata['input_window'],
            'pred_horizon': metadata['pred_horizon'],
            'n_test': metadata['n_test'],
        },
        'checkpoints': {},
        'baseline_comparison': {
            'D2STGNN': {
                'mape_h3': 6.48,
                'mape_h6': 8.15,
                'mape_h12': 10.03,
                'mae_h3': 2.77,
                'mae_h6': 3.28,
                'mae_h12': 4.01,
                'rmse_h3': 5.53,
                'rmse_h6': 6.85,
                'rmse_h12': 8.72,
            }
        }
    }

    # Checkpoints to evaluate
    checkpoints_to_evaluate = [
        ('best_model', checkpoint_dir / 'best_model.pt'),
        ('final_model', checkpoint_dir / 'final_model.pt'),
        ('last_checkpoint', checkpoint_dir / 'last_checkpoint.pt'),
    ]

    for name, ckpt_path in checkpoints_to_evaluate:
        if ckpt_path.exists():
            result = evaluate_checkpoint(model, ckpt_path, test_loader, args.device)
            evaluation_results['checkpoints'][name] = result
            print_metrics(name, result)
        else:
            print(f"\n[WARNING] {name}: checkpoint not found at {ckpt_path}")

    # =========================================================================
    # COMPARISON WITH BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON WITH D2STGNN BASELINE")
    print("=" * 70)

    d2stgnn = evaluation_results['baseline_comparison']['D2STGNN']

    print(f"\n{'Model':<20} {'MAPE H3':>10} {'MAPE H6':>10} {'MAPE H12':>10} {'Avg MAPE':>10}")
    print("-" * 62)
    print(f"{'D2STGNN (baseline)':<20} {d2stgnn['mape_h3']:>9.2f}% {d2stgnn['mape_h6']:>9.2f}% {d2stgnn['mape_h12']:>9.2f}% {(d2stgnn['mape_h3']+d2stgnn['mape_h6']+d2stgnn['mape_h12'])/3:>9.2f}%")

    for name in ['best_model', 'final_model', 'last_checkpoint']:
        if name in evaluation_results['checkpoints']:
            m = evaluation_results['checkpoints'][name]['metrics']
            avg_mape = (m['mape_h3'] + m['mape_h6'] + m['mape_h12']) / 3
            print(f"{name:<20} {m['mape_h3']:>9.2f}% {m['mape_h6']:>9.2f}% {m['mape_h12']:>9.2f}% {avg_mape:>9.2f}%")

    # Detailed comparison for best model
    if 'best_model' in evaluation_results['checkpoints']:
        print(f"\n{'Metric':<15} {'D2STGNN':>10} {'Best Model':>12} {'Difference':>12}")
        print("-" * 52)

        best = evaluation_results['checkpoints']['best_model']['metrics']

        for metric, label in [
            ('mape_h3', 'MAPE 15min'),
            ('mape_h6', 'MAPE 30min'),
            ('mape_h12', 'MAPE 60min'),
            ('mae_h3', 'MAE 15min'),
            ('mae_h6', 'MAE 30min'),
            ('mae_h12', 'MAE 60min'),
        ]:
            d2_val = d2stgnn[metric]
            our_val = best[metric]
            diff = our_val - d2_val
            status = "✓" if diff < 0 else " "
            unit = "%" if "mape" in metric else ""
            print(f"{label:<15} {d2_val:>9.2f}{unit} {our_val:>11.2f}{unit} {diff:>+11.2f}{unit} {status}")

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
