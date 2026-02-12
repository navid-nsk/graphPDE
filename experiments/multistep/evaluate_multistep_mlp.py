"""
evaluate_multistep_mlp.py

Multi-step prediction evaluation for MLP baseline starting from 2001:
- 2001 → 2006 (1 step forward)
- 2001 → 2011 (2 steps forward: 2001→2006→2011)
- 2001 → 2016 (3 steps forward: 2001→2006→2011→2016)
- 2001 → 2021 (4 steps forward: 2001→2006→2011→2016→2021)

The model is applied iteratively, using predictions as input for the next step.
Note: MLP has no spatial structure, processes each sample independently.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import sys
import pickle
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'baselines'))

from model_mlp_baseline import create_mlp_model
from dataloader_with_city_filter import ReactionDiffusionDatasetFiltered


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-step prediction evaluation for MLP from 2001'
    )

    # Data paths
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='Directory containing data files')
    parser.add_argument('--graph_path', type=str, default='../../data/graph_large_cities_rd.pkl',
                        help='Path to graph structure')
    parser.add_argument('--da_csv_path', type=str, default='../../data/da_canada.csv',
                        help='Path to DA-to-city mapping CSV')

    # Model checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoints/mlp_baseline',
                        help='Directory containing trained model')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pt',
                        help='Checkpoint file name')

    # Model hyperparameters (must match training)
    parser.add_argument('--feature_encoder_dim', type=int, default=256,
                        help='Feature encoder hidden dimension')
    parser.add_argument('--ethnicity_embed_dim', type=int, default=32,
                        help='Ethnicity embedding dimension')
    parser.add_argument('--predictor_hidden_dim', type=int, default=128,
                        help='Predictor hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for evaluation')

    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_multistep_mlp',
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

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'median_ae': median_ae,
        'n_samples': len(predictions),
        'mean_target': np.mean(targets),
        'mean_prediction': np.mean(predictions)
    }


def load_ground_truth_populations(data_dir, graph, city_filter=None, da_csv_path='./data/da_canada.csv'):
    """Load ground truth population data for all years."""
    train_df = pd.read_pickle(Path(data_dir) / 'train_rd.pkl')
    val_df = pd.read_pickle(Path(data_dir) / 'val_rd.pkl')
    test_df = pd.read_pickle(Path(data_dir) / 'test_rd.pkl')

    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if city_filter is not None:
        da_df = pd.read_csv(da_csv_path)
        da_df['DAUID'] = da_df['DAUID'].astype(str).str.strip()
        dauid_to_city = dict(zip(da_df['DAUID'], da_df['Name']))
        all_data['city_name'] = all_data['dauid'].map(dauid_to_city)
        all_data = all_data[all_data['city_name'].isin(city_filter)]

    populations = {}
    years = [2001, 2006, 2011, 2016, 2021]

    for year in years:
        if year == 2021:
            year_data = all_data[all_data['year_t'] == 2016][['dauid', 'ethnicity', 'pop_t1', 'node_idx']].copy()
            year_data = year_data.rename(columns={'pop_t1': 'population'})
        else:
            year_data = all_data[all_data['year_t'] == year][['dauid', 'ethnicity', 'pop_t', 'node_idx']].copy()
            year_data = year_data.rename(columns={'pop_t': 'population'})

        populations[year] = year_data
        print(f"  Year {year}: {len(year_data):,} samples")

    return populations, all_data


def build_full_graph_population(population_df, n_nodes, ethnicity_map, dauid_to_idx):
    """Build full graph population tensor from DataFrame."""
    n_ethnicities = len(ethnicity_map)
    population_tensor = torch.zeros(n_nodes, n_ethnicities, dtype=torch.float32)

    for _, row in population_df.iterrows():
        node_idx = int(row['node_idx'])
        if node_idx < n_nodes:
            ethnicity_idx = ethnicity_map.get(row['ethnicity'])
            if ethnicity_idx is not None:
                population_tensor[node_idx, ethnicity_idx] = float(row['population'])

    return population_tensor


def build_census_features(all_data, n_nodes, year, dauid_to_idx):
    """Build census features tensor for a specific year."""
    census_cols = [c for c in all_data.columns if c.startswith('census_')]
    n_census = len(census_cols)

    year_data = all_data[all_data['year_t'] == year]

    census_values = year_data[census_cols].values.astype(np.float32)
    census_mean = np.nan_to_num(census_values.mean(axis=0), nan=0.0)
    census_std = np.nan_to_num(census_values.std(axis=0), nan=1.0)
    census_std = np.clip(census_std, 1.0, None)

    census_tensor = torch.zeros(n_nodes, n_census, dtype=torch.float32)

    for _, row in year_data.drop_duplicates(subset=['dauid']).iterrows():
        dauid = row['dauid']
        if dauid in dauid_to_idx:
            node_idx = dauid_to_idx[dauid]
            if node_idx < n_nodes:
                census_values = row[census_cols].values.astype(np.float32)
                census_values = np.nan_to_num(census_values, nan=0.0)
                census_normalized = (census_values - census_mean) / census_std
                census_normalized = np.clip(census_normalized, -5.0, 5.0)
                census_tensor[node_idx] = torch.tensor(census_normalized)

    return census_tensor


@torch.no_grad()
def predict_one_step_mlp(model, population_t, census_features, ethnicity_map, device):
    """
    Run one step of MLP prediction (5 years forward).

    MLP processes each sample independently - no spatial structure.
    Input features: census + all ethnicities populations
    """
    model.eval()
    n_nodes, n_ethnicities = population_t.shape

    # Move to device
    population_t = population_t.to(device)
    census_features = census_features.to(device)

    # Create predictions tensor
    predicted_population = torch.zeros(n_nodes, n_ethnicities, device=device)

    batch_size = 2048

    for eth_idx in range(n_ethnicities):
        node_indices = torch.arange(n_nodes, device=device)
        ethnicity_indices = torch.full((n_nodes,), eth_idx, dtype=torch.long, device=device)

        for start_idx in range(0, n_nodes, batch_size):
            end_idx = min(start_idx + batch_size, n_nodes)
            batch_node_idx = node_indices[start_idx:end_idx]
            batch_eth_idx = ethnicity_indices[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx

            # Build features: census + all ethnicities populations
            # Shape: (batch_size, n_census + n_ethnicities)
            batch_census = census_features[batch_node_idx]  # (batch, n_census)
            batch_populations = population_t[batch_node_idx]  # (batch, n_ethnicities)

            # Combine features
            batch_features = torch.cat([batch_census, batch_populations], dim=1)

            # Create batch dictionary matching MLP's expected format
            batch = {
                'features': batch_features,
                'ethnicity': batch_eth_idx,
            }

            # Forward pass
            predictions = model(batch)

            # Store predictions
            predicted_population[batch_node_idx, eth_idx] = predictions

    return predicted_population


def evaluate_multistep(model, populations, all_data, graph, device, output_dir):
    """
    Perform multi-step prediction starting from 2001.
    """
    model.eval()

    n_nodes = graph['adjacency'].shape[0]
    dauid_to_idx = graph['dauid_to_idx']

    ethnicities = sorted(all_data['ethnicity'].unique())
    ethnicity_map = {eth: i for i, eth in enumerate(ethnicities)}
    n_ethnicities = len(ethnicity_map)

    print(f"\nGraph info:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Ethnicities: {n_ethnicities}")

    # Build initial population (2001)
    print("\nBuilding 2001 initial population...")
    pop_2001 = build_full_graph_population(populations[2001], n_nodes, ethnicity_map, dauid_to_idx)

    # Build census features for each period
    print("Building census features...")
    census_2001 = build_census_features(all_data, n_nodes, 2001, dauid_to_idx)
    census_2006 = build_census_features(all_data, n_nodes, 2006, dauid_to_idx)
    census_2011 = build_census_features(all_data, n_nodes, 2011, dauid_to_idx)
    census_2016 = build_census_features(all_data, n_nodes, 2016, dauid_to_idx)

    # Store predictions
    predictions = {2001: pop_2001}

    period_configs = [
        {'from_year': 2001, 'to_year': 2006, 'census': census_2001},
        {'from_year': 2006, 'to_year': 2011, 'census': census_2006},
        {'from_year': 2011, 'to_year': 2016, 'census': census_2011},
        {'from_year': 2016, 'to_year': 2021, 'census': census_2016},
    ]

    print("\n" + "="*60)
    print("RUNNING MULTI-STEP PREDICTIONS")
    print("="*60)

    current_pop = pop_2001.clone()

    for config in period_configs:
        from_year = config['from_year']
        to_year = config['to_year']
        census = config['census']

        print(f"\nPredicting {from_year} → {to_year}...")

        predicted_pop = predict_one_step_mlp(
            model, current_pop, census, ethnicity_map, device
        )

        predictions[to_year] = predicted_pop.cpu()
        current_pop = predicted_pop

        print(f"  Predicted population: mean={predicted_pop.mean():.2f}, max={predicted_pop.max():.2f}")

    # Evaluate predictions against ground truth
    print("\n" + "="*60)
    print("EVALUATING PREDICTIONS")
    print("="*60)

    results = []
    target_years = [2006, 2011, 2016, 2021]

    for target_year in target_years:
        print(f"\n--- Evaluating 2001 → {target_year} ---")

        gt_df = populations[target_year]
        gt_tensor = build_full_graph_population(gt_df, n_nodes, ethnicity_map, dauid_to_idx)

        pred_tensor = predictions[target_year]

        pred_values = []
        gt_values = []

        for _, row in gt_df.iterrows():
            node_idx = int(row['node_idx'])
            if node_idx < n_nodes:
                ethnicity_idx = ethnicity_map.get(row['ethnicity'])
                if ethnicity_idx is not None:
                    pred_values.append(pred_tensor[node_idx, ethnicity_idx].item())
                    gt_values.append(row['population'])

        pred_values = np.array(pred_values)
        gt_values = np.array(gt_values)

        metrics = compute_metrics(pred_values, gt_values)
        metrics['source_year'] = 2001
        metrics['target_year'] = target_year
        metrics['n_steps'] = (target_year - 2001) // 5

        results.append(metrics)

        print(f"  Samples: {metrics['n_samples']:,}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²: {metrics['r2']:.4f}")

        # Save detailed predictions
        detailed_df = gt_df.copy()
        detailed_df['prediction'] = pred_values
        detailed_df['target'] = gt_values
        detailed_path = output_dir / f'predictions_2001_to_{target_year}.csv'
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  Saved to: {detailed_path}")

        # Save metrics
        metrics_path = output_dir / f'metrics_2001_to_{target_year}.csv'
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"  Metrics saved to: {metrics_path}")

    return results


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load graph
    print("\n" + "="*80)
    print("LOADING GRAPH")
    print("="*80)

    with open(args.graph_path, 'rb') as f:
        graph = pickle.load(f)

    # Load ground truth populations
    print("\n" + "="*80)
    print("LOADING GROUND TRUTH POPULATIONS")
    print("="*80)

    populations, all_data = load_ground_truth_populations(
        args.data_dir, graph, args.city_filter, args.da_csv_path
    )

    # If city filter applied, reload with filtered dataset
    if args.city_filter is not None:
        print("\nReloading with city filter...")
        test_dataset = ReactionDiffusionDatasetFiltered(
            samples_path=str(Path(args.data_dir) / 'test_rd.pkl'),
            graph_path=str(args.graph_path),
            ethnicity_filter=None,
            census_stats=None,
            is_training=False,
            city_filter=None,
            da_csv_path=str(args.da_csv_path)
        )
        graph = test_dataset.graph

        populations, all_data = load_ground_truth_populations(
            args.data_dir, graph, args.city_filter, args.da_csv_path
        )

    # Get model configuration
    n_nodes = graph['adjacency'].shape[0]
    n_ethnicities = len(all_data['ethnicity'].unique())
    census_cols = [c for c in all_data.columns if c.startswith('census_')]
    n_census_features = len(census_cols)

    print(f"\nConfiguration:")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Ethnicities: {n_ethnicities}")
    print(f"  Census features: {n_census_features}")

    # Create model
    print("\n" + "="*80)
    print("LOADING MLP MODEL")
    print("="*80)

    model = create_mlp_model(
        n_census_features=n_census_features,
        n_ethnicities=n_ethnicities,
        feature_encoder_dim=args.feature_encoder_dim,
        ethnicity_embed_dim=args.ethnicity_embed_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        dropout=args.dropout
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

    # Run multi-step evaluation
    results = evaluate_multistep(
        model, populations, all_data, graph, device, output_dir
    )

    # Save summary
    print("\n" + "="*80)
    print("SAVING SUMMARY")
    print("="*80)

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / 'summary_multistep.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print("MULTI-STEP PREDICTION SUMMARY - MLP (Starting from 2001)")
    print("="*80)
    print(f"\n{'Target Year':<15} {'Steps':>8} {'Samples':>10} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'MAPE':>10}")
    print("-" * 80)
    for result in results:
        print(f"{result['target_year']:<15} {result['n_steps']:>8} {result['n_samples']:>10,} "
              f"{result['mae']:>10.2f} {result['rmse']:>10.2f} "
              f"{result['r2']:>10.4f} {result['mape']:>9.2f}%")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
