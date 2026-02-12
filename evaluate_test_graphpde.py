"""
evaluate_test_graphpde.py

Evaluate GraphPDE model on the test dataset.
Reports overall metrics: MAE, RMSE, MAPE, R², Median AE.
"""

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import sys
import pickle
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model_graphpde import create_graphpde_model
from dataloader_with_city_filter import create_dataloaders_with_city_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate GraphPDE model on test dataset'
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
    parser.add_argument('--da_csv_path', type=str, default='./data/da_canada.csv',
                        help='Path to DA-to-city mapping CSV')

    # Model checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/graphpde',
                        help='Directory containing trained model')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pt',
                        help='Checkpoint file name')

    # Model hyperparameters (must match training)
    parser.add_argument('--n_ode_steps', type=int, default=1,
                        help='Number of ODE integration steps')
    parser.add_argument('--integration_time', type=float, default=5.0,
                        help='Integration time in years')
    parser.add_argument('--use_cuda_kernels', action='store_true', default=True,
                        help='Use custom CUDA kernels')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/evaluation_test_graphpde',
                        help='Directory to save evaluation results')

    # City filter (must match training)
    parser.add_argument('--city_filter', type=str, nargs='+',
                        default=['Toronto', 'Mississauga', 'Brampton', 'Markham', 'Vaughan',
                                'Richmond Hill', 'Oakville', 'Oshawa', 'Whitby', 'Hamilton'],
                        help='Cities to include')

    args = parser.parse_args()
    return args


def compute_metrics(predictions, targets):
    """Compute comprehensive evaluation metrics."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    non_zero_mask = targets > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) /
                              targets[non_zero_mask])) * 100
    else:
        mape = 0.0

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    median_ae = np.median(np.abs(predictions - targets))

    # Additional metrics
    relative_errors = np.abs(predictions - targets) / (targets + 1.0)
    mean_relative_error = np.mean(relative_errors)

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_ae': median_ae,
        'mean_relative_error': mean_relative_error,
        'n_samples': len(predictions),
        'mean_target': np.mean(targets),
        'mean_prediction': np.mean(predictions),
        'std_target': np.std(targets),
        'std_prediction': np.std(predictions)
    }


@torch.no_grad()
def evaluate_test(model, test_loader, graph, device):
    """
    Evaluate model on test dataset.
    """
    model.eval()

    # Prepare city mapping
    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])

    all_predictions = []
    all_targets = []
    total_loss = 0.0

    print("\nEvaluating on test set...")
    for batch in tqdm(test_loader, desc="Testing"):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Get city indices for batch
        node_indices = batch['node_idx'].cpu().numpy()
        city_indices = node_to_city[node_indices]
        city_indices = torch.tensor(city_indices, dtype=torch.long, device=device)

        # Forward pass
        output = model(batch, city_indices)

        predictions = output['pop_pred']
        targets = batch['target']

        # Compute loss
        loss = F.mse_loss(predictions, targets)
        total_loss += loss.item() * len(targets)

        # Store predictions and targets
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(test_loader.dataset)

    return metrics, all_predictions, all_targets


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    train_loader, val_loader, test_loader, graph = create_dataloaders_with_city_filter(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        graph_path=args.graph_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ethnicity_filter=None,
        city_filter=None
    )

    # Get data statistics
    n_nodes = graph['adjacency'].shape[0]
    n_ethnicities = test_loader.dataset.n_ethnicities
    census_cols = test_loader.dataset.census_cols
    n_census_features = len(census_cols)

    # Get city mapping
    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])
    n_cities = len(unique_cities)

    print(f"\nData statistics:")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Cities: {n_cities}")
    print(f"  Ethnicities: {n_ethnicities}")
    print(f"  Census features: {n_census_features}")

    # Create model
    print("\n" + "="*80)
    print("LOADING GRAPHPDE MODEL")
    print("="*80)

    model = create_graphpde_model(
        n_ethnicities=n_ethnicities,
        n_cities=n_cities,
        n_dauids=n_nodes,
        n_census_features=n_census_features,
        adjacency=graph['adjacency'],
        node_to_city=node_to_city,
        n_ode_steps=args.n_ode_steps,
        integration_time=args.integration_time,
        use_cuda_kernels=args.use_cuda_kernels
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # Run evaluation
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)

    metrics, predictions, targets = evaluate_test(model, test_loader, graph, device)

    # Print results
    print("\n" + "="*80)
    print("TEST SET RESULTS - GraphPDE")
    print("="*80)
    print(f"\n  Samples:            {metrics['n_samples']:,}")
    print(f"  Loss (MSE):         {metrics['loss']:.4f}")
    print(f"  MAE:                {metrics['mae']:.2f}")
    print(f"  RMSE:               {metrics['rmse']:.2f}")
    print(f"  MAPE:               {metrics['mape']:.2f}%")
    print(f"  R²:                 {metrics['r2']:.4f}")
    print(f"  Median AE:          {metrics['median_ae']:.2f}")
    print(f"  Mean Relative Err:  {metrics['mean_relative_error']:.4f}")
    print(f"\n  Mean Target:        {metrics['mean_target']:.2f}")
    print(f"  Mean Prediction:    {metrics['mean_prediction']:.2f}")
    print(f"  Std Target:         {metrics['std_target']:.2f}")
    print(f"  Std Prediction:     {metrics['std_prediction']:.2f}")

    # Save metrics to CSV
    metrics_path = output_dir / 'test_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"\nMetrics saved to: {metrics_path}")

    # Save predictions
    predictions_path = output_dir / 'test_predictions.csv'
    pred_df = pd.DataFrame({
        'prediction': predictions.numpy(),
        'target': targets.numpy(),
        'error': (predictions - targets).numpy(),
        'abs_error': np.abs((predictions - targets).numpy())
    })
    pred_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("GraphPDE Test Set Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n\n")
        f.write(f"Test Samples: {metrics['n_samples']:,}\n\n")
        f.write("Metrics:\n")
        f.write(f"  MAE:      {metrics['mae']:.2f}\n")
        f.write(f"  RMSE:     {metrics['rmse']:.2f}\n")
        f.write(f"  MAPE:     {metrics['mape']:.2f}%\n")
        f.write(f"  R²:       {metrics['r2']:.4f}\n")
        f.write(f"  Median AE:{metrics['median_ae']:.2f}\n")
    print(f"Summary saved to: {summary_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
