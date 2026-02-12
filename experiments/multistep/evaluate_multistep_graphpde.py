"""
evaluate_multistep_graphpde.py

Evaluate trained GraphPDE multistep models at different forecast horizons:
  - 5-year  (1-step prediction)
  - 10-year (2-step prediction)
  - 15-year (3-step prediction)
  - 20-year (4-step prediction)

Evaluates multiple checkpoints (best_model.pt, best_r2_model.pt, best_loss_model.pt)
and computes averaged metrics across all models for each horizon.

Metrics (MAE, RMSE, MAPE, R2, Median AE, Mean Relative Error) are computed
per horizon and saved to CSV.

Usage:
    python evaluate_multistep_graphpde.py
    python evaluate_multistep_graphpde.py --checkpoint_dir ./checkpoints/graphpde_multistep
    python evaluate_multistep_graphpde.py --split test
"""

import argparse
import torch
import numpy as np
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from model_graphpde import create_graphpde_model
from trainer_graphpde_multistep import compute_metrics
from dataloader_with_city_filter import ReactionDiffusionDatasetFiltered


# =========================================================================
# Rolling windows for each horizon (internal use only)
# =========================================================================
# Each entry: (start_year, list of (period_idx, target_year) per step)
# Note: 2001 excluded from 5-year, 10-year, 15-year to focus on stable periods
_HORIZON_WINDOWS = {
    '5-year': [
        (2006, [(1, 2011)]),
        (2011, [(2, 2016)]),
        (2016, [(3, 2021)]),
    ],
    '10-year': [
        (2006, [(1, 2011), (2, 2016)]),
        (2011, [(2, 2016), (3, 2021)]),
    ],
    '15-year': [
        (2006, [(1, 2011), (2, 2016), (3, 2021)]),
    ],
    '20-year': [
        (2001, [(0, 2006), (1, 2011), (2, 2016), (3, 2021)]),
    ],
}

# Checkpoint files to evaluate and average
DEFAULT_CHECKPOINT_FILES = [
    'best_model.pt',
    'best_r2_model.pt',
    'best_loss_model.pt',
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate GraphPDE multistep model at different horizons'
    )

    # Data paths
    parser.add_argument('--train_path', type=str, default='../../data/train_rd.csv')
    parser.add_argument('--val_path', type=str, default='../../data/val_rd.csv')
    parser.add_argument('--test_path', type=str, default='../../data/test_rd.csv')
    parser.add_argument('--graph_path', type=str, default='../../data/graph_large_cities_rd.pkl')
    parser.add_argument('--da_csv_path', type=str, default='../../data/da_canada.csv')

    # Model
    parser.add_argument('--n_ode_steps', type=int, default=1)
    parser.add_argument('--integration_time', type=float, default=5.0)
    parser.add_argument('--use_cuda_kernels', action='store_true', default=True)

    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str,
                        default='../../checkpoints/graphpde_multistep')
    parser.add_argument('--checkpoint_files', type=str, nargs='+',
                        default=None,
                        help='Checkpoint files to evaluate. Defaults to best_model.pt, '
                             'best_r2_model.pt, best_loss_model.pt')

    # Evaluation split
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which data split to evaluate')

    # City filter (must match training)
    parser.add_argument('--city_filter', type=str, nargs='+',
                        default=['Toronto', 'Mississauga', 'Brampton', 'Markham',
                                 'Vaughan', 'Abbotsford', 'Winnipeg', 'Surrey',
                                 'Delta', 'Burnaby'])

    # Output
    parser.add_argument('--output_csv', type=str, default='../../results/experiment/evaluation_horizons_test.csv',
                        help='Output CSV path. Defaults to <checkpoint_dir>/evaluation_horizons_<split>.csv')

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def predict_one_step(model, population_t, census_features, period_idx,
                     node_to_city_tensor, device):
    """
    Run one step of prediction on the full graph (5 years forward).
    """
    n_nodes = population_t.shape[0]

    diffusion_coef = model.diffusion_coef[period_idx]

    full_period_idx = torch.full(
        (n_nodes,), period_idx, dtype=torch.long, device=device
    )
    full_node_idx = torch.arange(n_nodes, dtype=torch.long, device=device)

    def diffusion_fn(state):
        return model.diffusion_weight * model.graph_laplacian(
            state, diffusion_coef
        )

    def reaction_fn(state):
        return model.reaction_module(
            state,
            node_to_city_tensor,
            census_features,
            full_node_idx,
            torch.zeros(n_nodes, dtype=torch.long, device=device),
            full_period_idx,
            debug_batch=False
        )

    predicted_population = model.ode_solver(
        population_t,
        diffusion_fn,
        reaction_fn,
        T=model.integration_time,
        debug_batch=False
    )

    return predicted_population


def evaluate_chain(model, populations, census_features, start_year, steps,
                   node_to_city_tensor, device):
    """
    Run an autoregressive chain from start_year through the given steps.
    """
    current_pop = populations[start_year]

    for period_idx, target_year in steps:
        census_year = [2001, 2006, 2011, 2016][period_idx]
        census = census_features[census_year]

        predicted_pop = predict_one_step(
            model, current_pop, census, period_idx,
            node_to_city_tensor, device
        )
        predicted_pop = torch.clamp(predicted_pop, min=0.0, max=1e6)

        # Feed prediction forward (autoregressive)
        current_pop = predicted_pop

    final_target_year = steps[-1][1]
    return predicted_pop, final_target_year


def evaluate_model_all_horizons(
    model, populations, census_features, node_to_city_tensor, device
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single model across all horizons.

    Returns:
        Dict mapping horizon name to metrics dict
    """
    results = {}

    with torch.no_grad():
        for horizon_name, windows in _HORIZON_WINDOWS.items():
            horizon_preds = []
            horizon_tgts = []

            for start_year, steps in windows:
                if start_year not in populations:
                    continue

                final_target_year = steps[-1][1]
                if final_target_year not in populations:
                    continue

                # Run chain
                pred, target_year = evaluate_chain(
                    model, populations, census_features,
                    start_year, steps,
                    node_to_city_tensor, device
                )
                gt = populations[target_year]

                # Mask: only evaluate where ground truth > 0
                non_zero_mask = gt > 0
                if non_zero_mask.sum() == 0:
                    continue

                pred_masked = pred[non_zero_mask]
                gt_masked = gt[non_zero_mask]

                # Accumulate for horizon-level aggregate
                horizon_preds.append(pred_masked)
                horizon_tgts.append(gt_masked)

            # Compute horizon-level aggregate metrics
            if horizon_preds:
                all_pred = torch.cat(horizon_preds)
                all_tgt = torch.cat(horizon_tgts)
                metrics = compute_metrics(all_pred, all_tgt)
                metrics['n_samples'] = all_pred.shape[0]
                results[horizon_name] = metrics

    return results


def average_metrics(all_model_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Any]]:
    """
    Average metrics across multiple model evaluations.

    Args:
        all_model_results: List of results from each model (output of evaluate_model_all_horizons)

    Returns:
        Dict mapping horizon name to averaged metrics with std
    """
    averaged = {}

    # Get all horizon names from first model
    if not all_model_results:
        return averaged

    horizon_names = list(all_model_results[0].keys())
    metric_names = ['mae', 'rmse', 'mape', 'r2', 'median_ae', 'mean_relative_error']

    for horizon in horizon_names:
        horizon_metrics = {}

        for metric in metric_names:
            values = []
            for model_results in all_model_results:
                if horizon in model_results and metric in model_results[horizon]:
                    values.append(model_results[horizon][metric])

            if values:
                horizon_metrics[metric] = np.mean(values)
                horizon_metrics[f'{metric}_std'] = np.std(values)
                horizon_metrics[f'{metric}_min'] = np.min(values)
                horizon_metrics[f'{metric}_max'] = np.max(values)

        # n_samples should be the same across models
        if horizon in all_model_results[0]:
            horizon_metrics['n_samples'] = all_model_results[0][horizon].get('n_samples', 0)

        averaged[horizon] = horizon_metrics

    return averaged


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Determine which checkpoint files to use
    if args.checkpoint_files is None:
        checkpoint_files = DEFAULT_CHECKPOINT_FILES
    else:
        checkpoint_files = args.checkpoint_files

    # Filter to only existing checkpoints
    existing_checkpoints = []
    for cf in checkpoint_files:
        cp_path = checkpoint_dir / cf
        if cp_path.exists():
            existing_checkpoints.append(cf)
        else:
            print(f"Warning: Checkpoint not found: {cp_path}")

    if not existing_checkpoints:
        print("ERROR: No checkpoint files found!")
        return

    print(f"\nWill evaluate {len(existing_checkpoints)} checkpoint(s):")
    for cf in existing_checkpoints:
        print(f"  - {cf}")

    # Output path
    if args.output_csv is None:
        output_csv = checkpoint_dir / f'evaluation_horizons_{args.split}.csv'
    else:
        output_csv = Path(args.output_csv)

    # =====================================================================
    # LOAD DATA
    # =====================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Always need training dataset for census_stats normalization
    print("Loading training dataset (for normalization stats)...")
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

    # Load evaluation dataset
    split_path_map = {
        'train': args.train_path,
        'val': args.val_path,
        'test': args.test_path,
    }
    eval_path = split_path_map[args.split]

    if args.split == 'train':
        eval_dataset = train_dataset
    else:
        print(f"Loading {args.split} dataset...")
        eval_dataset = ReactionDiffusionDatasetFiltered(
            samples_path=str(eval_path),
            graph_path=str(args.graph_path),
            ethnicity_filter=None,
            census_stats=census_stats,
            is_training=False,
            city_filter=args.city_filter,
            da_csv_path=str(args.da_csv_path)
        )

    graph = eval_dataset.graph

    # =====================================================================
    # PREPARE POPULATIONS AND CENSUS
    # =====================================================================
    print("\nPreparing ground truth populations...")

    populations = {}
    census_features = {}

    for year in [2001, 2006, 2011, 2016]:
        if year in eval_dataset.full_graph_cache:
            populations[year] = eval_dataset.full_graph_cache[year]['population'].to(device)
            census_features[year] = eval_dataset.full_graph_cache[year]['census'].to(device)

    # Build 2021 ground truth (pop_t1 from 2016)
    pop_2021 = eval_dataset.get_population_for_year(2016, return_t1=True)
    populations[2021] = pop_2021.to(device)

    print(f"  Data prepared for evaluation")

    # =====================================================================
    # PREPARE MODEL STRUCTURE
    # =====================================================================
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)

    adjacency = graph['adjacency']
    n_nodes = adjacency.shape[0]

    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])
    n_cities = len(unique_cities)

    node_to_city_tensor = torch.tensor(
        node_to_city, dtype=torch.long, device=device
    )

    n_ethnicities = eval_dataset.n_ethnicities
    sample_year = list(eval_dataset.full_graph_cache.keys())[0]
    n_census_features = eval_dataset.full_graph_cache[sample_year]['census'].shape[1]

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
    model = model.to(device)

    print(f"  Model created successfully")

    # =====================================================================
    # EVALUATE EACH CHECKPOINT
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"EVALUATING ON {args.split.upper()} SET")
    print("=" * 80)

    all_model_results = []

    for checkpoint_file in existing_checkpoints:
        checkpoint_path = checkpoint_dir / checkpoint_file
        print(f"  Evaluating: {checkpoint_file}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Evaluate
        results = evaluate_model_all_horizons(
            model, populations, census_features,
            node_to_city_tensor, device
        )
        all_model_results.append(results)

    # =====================================================================
    # COMPUTE AVERAGED METRICS
    # =====================================================================
    print("\n" + "=" * 80)
    print("COMPUTING AVERAGED METRICS")
    print("=" * 80)

    averaged_results = average_metrics(all_model_results)

    # Prepare averaged results for CSV
    averaged_csv_rows = []
    for horizon_name in ['5-year', '10-year', '15-year', '20-year']:
        if horizon_name in averaged_results:
            m = averaged_results[horizon_name]
            n_steps = int(horizon_name.replace('-year', '')) // 5

            averaged_csv_rows.append({
                'horizon': horizon_name,
                'n_steps': n_steps,
                'mae': m['mae'],
                'mae_std': m['mae_std'],
                'rmse': m['rmse'],
                'rmse_std': m['rmse_std'],
                'mape': m['mape'],
                'mape_std': m['mape_std'],
                'r2': m['r2'],
                'r2_std': m['r2_std'],
                'median_ae': m['median_ae'],
                'median_ae_std': m['median_ae_std'],
                'mean_relative_error': m['mean_relative_error'],
                'mean_relative_error_std': m['mean_relative_error_std'],
                'n_samples': m['n_samples'],
                'n_models': len(existing_checkpoints),
            })

    # =====================================================================
    # SAVE TO CSV (same format as displayed table)
    # =====================================================================
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save results with +/- format
    fieldnames = ['Horizon', 'Steps', 'MAE', 'RMSE', 'MAPE(%)', 'R2']

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in averaged_csv_rows:
            mae_str = f"{row['mae']:.2f} +/- {row['mae_std']:.2f}"
            rmse_str = f"{row['rmse']:.2f} +/- {row['rmse_std']:.2f}"
            mape_str = f"{row['mape']:.2f} +/- {row['mape_std']:.2f}"
            r2_str = f"{row['r2']:.4f} +/- {row['r2_std']:.4f}"
            writer.writerow([
                row['horizon'],
                row['n_steps'],
                mae_str,
                rmse_str,
                mape_str,
                r2_str
            ])

    # =====================================================================
    # PRINT SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"MULTI-STEP PREDICTION RESULTS")
    print("=" * 80)
    print(f"{'Horizon':<14}{'Steps':>6}{'MAE':>18}{'RMSE':>18}{'MAPE(%)':>18}{'R2':>22}")
    print("-" * 98)

    for row in averaged_csv_rows:
        mae_str = f"{row['mae']:.2f} +/- {row['mae_std']:.2f}"
        rmse_str = f"{row['rmse']:.2f} +/- {row['rmse_std']:.2f}"
        mape_str = f"{row['mape']:.2f} +/- {row['mape_std']:.2f}"
        r2_str = f"{row['r2']:.4f} +/- {row['r2_std']:.4f}"

        print(f"{row['horizon']:<14}"
              f"{row['n_steps']:>6}"
              f"{mae_str:>18}"
              f"{rmse_str:>18}"
              f"{mape_str:>18}"
              f"{r2_str:>22}")

    print("-" * 98)
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()
