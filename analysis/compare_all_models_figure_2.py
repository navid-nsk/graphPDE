"""
Figure 2: Model Performance Comparison

Generates panels for Figure 2 comparing GraphPDE against baseline models.
Outputs saved to: ./figures/figure2/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from matplotlib.gridspec import GridSpec

# Publication-quality styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['grid.alpha'] = 0.25

# Consistent color scheme - GraphPDE stands out
MODEL_COLORS = {
    'XGBoost + SHAP': '#7570b3',      # Purple
    'MLP': '#d95f02',                  # Orange
    'GCN': '#1b9e77',                  # Teal
    'GAT': '#e7298a',                  # Magenta
    'GraphSage': '#66a61e',            # Olive green
    'DeepONet': '#e6ab02',             # Gold/Yellow
    'TGCN': '#a6761d',                 # Brown
    'DCRNN': '#666666',                # Gray
    'GraphPDE No Residual': '#ff9896', # Light red/pink
    'GraphPDE': '#d62728'              # Bright red for emphasis
}


def load_model_results(model_dir, model_name):
    """Load results for a specific model"""
    model_dir = Path(model_dir)
    
    test_csv = model_dir / 'test_results.csv'
    if not test_csv.exists():
        test_json = model_dir / 'test_results.json'
        if test_json.exists():
            with open(test_json, 'r') as f:
                metrics = json.load(f)
            if 'overall' in metrics:
                metrics = metrics['overall']
        else:
            raise FileNotFoundError(f"No test results found in {model_dir}")
    else:
        df = pd.read_csv(test_csv)
        metrics = dict(zip(df['metric'], df['value']))
    
    training_history = None
    metrics_csv = model_dir / 'metrics.csv'
    if metrics_csv.exists():
        training_history = pd.read_csv(metrics_csv)
    
    return {
        'name': model_name,
        'metrics': metrics,
        'training_history': training_history
    }


def smooth_curve(y, window=20):
    """Apply moving average smoothing"""
    if len(y) < window:
        return y
    return uniform_filter1d(y, size=window, mode='nearest')


def create_performance_table(models_data, output_dir):
    """Create comprehensive performance comparison table"""
    print("\nCreating performance comparison table...")
    
    comparison_data = []
    for model_data in models_data:
        metrics = model_data['metrics']
        comparison_data.append({
            'Model': model_data['name'],
            'MAE': metrics.get('mae', np.nan),
            'RMSE': metrics.get('rmse', np.nan),
            'R²': metrics.get('r2', np.nan),
            'MAPE (%)': metrics.get('mape', np.nan),
            'Median AE': metrics.get('median_ae', np.nan)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save outputs
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    df.to_excel(output_dir / 'model_comparison.xlsx', index=False)
    
    # LaTeX table
    latex_str = df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Performance comparison of all models on test set",
        label="tab:model_comparison",
        column_format='l' + 'r' * (len(df.columns) - 1)
    )
    with open(output_dir / 'model_comparison.tex', 'w') as f:
        f.write(latex_str)
    
    print("\n" + "="*90)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    
    return df


def create_clean_performance_comparison(comparison_df, output_dir):
    """Create clean, focused performance comparison figure"""
    print("\nCreating clean performance comparison...")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Panel 1: R² (most important metric) - SORTED HIGH TO LOW
    ax1 = fig.add_subplot(gs[0])
    
    # Sort by R² descending
    r2_sorted = comparison_df.sort_values('R²', ascending=True)  # ascending=True for horizontal bars
    models_sorted_r2 = r2_sorted['Model'].values
    r2_values = r2_sorted['R²'].values
    colors_r2 = [MODEL_COLORS.get(m, 'gray') for m in models_sorted_r2]
    
    y_pos = np.arange(len(models_sorted_r2))
    
    bars = ax1.barh(y_pos, r2_values, color=colors_r2, alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_sorted_r2)
    ax1.set_xlabel('R² Score', fontweight='bold')
    ax1.set_title('(a) Coefficient of Determination', fontweight='bold', pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, max(r2_values) * 1.15)
    
    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Panel 2: Error metrics (keep original order)
    ax2 = fig.add_subplot(gs[1])
    models = comparison_df['Model'].values
    colors = [MODEL_COLORS.get(m, 'gray') for m in models]
    mae_values = comparison_df['MAE'].values
    rmse_values = comparison_df['RMSE'].values
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, mae_values, width, label='MAE',
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x_pos + width/2, rmse_values, width, label='RMSE',
                    color='coral', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Error Value', fontweight='bold')
    ax2.set_title('(b) Absolute Error Metrics', fontweight='bold', pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=25, ha='right')
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel 3: MAPE - SORTED LOW TO HIGH (lower is better, so best at top)
    ax3 = fig.add_subplot(gs[2])
    
    # Sort by MAPE ascending (so lowest/best is at top)
    mape_sorted = comparison_df.sort_values('MAPE (%)', ascending=True)
    models_sorted_mape = mape_sorted['Model'].values
    mape_values = mape_sorted['MAPE (%)'].values
    colors_mape = [MODEL_COLORS.get(m, 'gray') for m in models_sorted_mape]
    
    y_pos = np.arange(len(models_sorted_mape))
    
    bars = ax3.barh(y_pos, mape_values, color=colors_mape, alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models_sorted_mape)
    ax3.set_xlabel('MAPE (%)', fontweight='bold')
    ax3.set_title('(c) Mean Absolute Percentage Error', fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim(0, max(mape_values) * 1.15)
    
    for i, (bar, val) in enumerate(zip(bars, mape_values)):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison on Test Set', 
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.savefig(output_dir / 'performance_comparison_clean.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_comparison_clean.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_improved_training_curves(models_data, output_dir):
    """Create clean, smoothed training curves with GraphPDE highlighted"""
    print("\nCreating improved training curves...")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics_to_plot = ['loss', 'mae', 'rmse', 'r2']
    metric_labels = ['Training Loss', 'Mean Absolute Error (MAE)', 
                     'Root Mean Squared Error (RMSE)', 'R² Score']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        has_data = False
        
        # Plot baselines first (lighter) - validation only
        for model_data in models_data:
            if model_data['name'] != 'GraphPDE':
                history = model_data['training_history']
                if history is not None:
                    val_col = f'val_{metric}'
                    if val_col in history.columns:
                        has_data = True
                        epochs = history['epoch'].values
                        val_values = history[val_col].values
                        
                        # Smooth the curve
                        val_smooth = smooth_curve(val_values, window=10)
                        
                        color = MODEL_COLORS.get(model_data['name'], 'gray')
                        ax.plot(epochs, val_smooth, linewidth=1.8, 
                               label=f'{model_data["name"]} (val)', color=color, alpha=0.5)
        
        # Plot GraphPDE last (prominent)
        for model_data in models_data:
            if model_data['name'] == 'GraphPDE':
                history = model_data['training_history']
                if history is not None:
                    train_col = f'train_{metric}'
                    val_col = f'val_{metric}'
                    
                    if train_col in history.columns and val_col in history.columns:
                        has_data = True
                        epochs = history['epoch'].values
                        train_values = history[train_col].values
                        val_values = history[val_col].values
                        
                        # Smooth curves
                        train_smooth = smooth_curve(train_values, window=10)
                        val_smooth = smooth_curve(val_values, window=10)
                        
                        color = MODEL_COLORS['GraphPDE']
                        
                        # Plot training (dashed, thinner)
                        ax.plot(epochs, train_smooth, linewidth=2.5,
                               linestyle='--', label='GraphPDE (train)', 
                               color=color, alpha=0.6, zorder=10)
                        
                        # Plot validation (solid, thicker)
                        ax.plot(epochs, val_smooth, linewidth=3.5,
                               label='GraphPDE (val)', color=color, 
                               alpha=0.95, zorder=11)
        
        if has_data:
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(label, fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {label} over Training', 
                        fontweight='bold', pad=10)
            ax.legend(loc='best', frameon=True, fancybox=True, fontsize=9)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
    plt.suptitle('Model Training Dynamics (Smoothed Curves)', 
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'training_curves_improved.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves_improved.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    


def create_convergence_plot(models_data, output_dir):
    """Create focused convergence comparison for R²"""
    print("\nCreating convergence comparison...")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(12, 7))
    
    has_data = False
    best_r2 = {}
    
    # Plot baseline models
    for model_data in models_data:
        if model_data['name'] != 'GraphPDE':
            history = model_data['training_history']
            if history is not None and 'val_r2' in history.columns:
                has_data = True
                epochs = history['epoch'].values
                val_r2 = smooth_curve(history['val_r2'].values, window=10)
                
                color = MODEL_COLORS.get(model_data['name'], 'gray')
                ax.plot(epochs, val_r2, linewidth=2.0,
                       label=model_data['name'], color=color, alpha=0.6)
                best_r2[model_data['name']] = val_r2.max()
    
    # Plot GraphPDE prominently
    for model_data in models_data:
        if model_data['name'] == 'GraphPDE':
            history = model_data['training_history']
            if history is not None and 'val_r2' in history.columns:
                has_data = True
                epochs = history['epoch'].values
                val_r2 = smooth_curve(history['val_r2'].values, window=10)
                
                color = MODEL_COLORS['GraphPDE']
                ax.plot(epochs, val_r2, linewidth=4.0,
                       label='GraphPDE', color=color, alpha=0.95, zorder=10)
                best_r2['GraphPDE'] = val_r2.max()
                
                # Add best performance line
                ax.axhline(y=val_r2.max(), color=color, linestyle=':', 
                          alpha=0.5, linewidth=2, zorder=5)
                ax.text(epochs[-1] * 0.7, val_r2.max() + 0.02,
                       f'GraphPDE best: {val_r2.max():.4f}',
                       color=color, fontweight='bold', fontsize=11)
    
    if has_data:
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Validation R² Score', fontweight='bold', fontsize=12)
        ax.set_title('Model Convergence Comparison (Validation R²)',
                    fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, 
                 shadow=True, fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set reasonable y-limits
        if best_r2:
            min_r2 = min(best_r2.values())
            max_r2 = max(best_r2.values())
            ax.set_ylim(min_r2 - 0.1, max_r2 + 0.05)
    
    plt.tight_layout()
    
    plt.savefig(output_dir / 'convergence_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_comparison.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    


def create_performance_heatmap(comparison_df, output_dir):
    """Create clean heatmap (keep this - it was good!)"""
    print("\nCreating performance heatmap...")
    
    metrics = ['MAE', 'RMSE', 'R²', 'MAPE (%)', 'Median AE']
    heatmap_data = comparison_df[metrics].values.T
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Normalize
    normalized_data = np.zeros_like(heatmap_data)
    for i in range(heatmap_data.shape[0]):
        row = heatmap_data[i]
        if i == 2:  # R²
            normalized_data[i] = (row - row.min()) / (row.max() - row.min() + 1e-8)
        else:
            normalized_data[i] = 1 - (row - row.min()) / (row.max() - row.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(comparison_df)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(comparison_df['Model'].values, rotation=30, ha='right')
    ax.set_yticklabels(metrics)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Performance\n(0=worst, 1=best)', 
                   rotation=270, labelpad=25, fontweight='bold')
    
    # Add annotations
    for i in range(len(metrics)):
        for j in range(len(comparison_df)):
            actual_value = heatmap_data[i, j]
            text_str = f'{actual_value:.3f}' if metrics[i] == 'R²' else f'{actual_value:.1f}'
            ax.text(j, i, text_str, ha="center", va="center", 
                   color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Model Performance Heatmap\n(Normalized: Green=Better, Red=Worse)',
                fontweight='bold', fontsize=13, pad=15)
    
    plt.tight_layout()
    
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    


def main():
    parser = argparse.ArgumentParser(
        description='Final publication-quality model comparison')
    parser.add_argument('--mlp_dir', type=str, default='../checkpoints/mlp_baseline')
    parser.add_argument('--gcn_dir', type=str, default='../checkpoints/gcn_baseline')
    parser.add_argument('--gnn_dir', type=str, default='../checkpoints/gnn_baseline')
    parser.add_argument('--graphsage_dir', type=str, default='../checkpoints/graphsage_baseline',
                       help='Directory with TGCN results')
    parser.add_argument('--deeponet_dir', type=str, default='../checkpoints/deeponet_baseline',
                       help='Directory with DeepONet results')
    parser.add_argument('--tgcn_dir', type=str, default='../checkpoints/tgcn_baseline',
                       help='Directory with TGCN results')
    parser.add_argument('--dcrnn_dir', type=str, default='../checkpoints/dcrnn_baseline',
                       help='Directory with DCRNN results')
    parser.add_argument('--graphpde_no_residual_dir', type=str, default='../checkpoints/graphpde_no_residual')
    parser.add_argument('--graphpde_dir', type=str, default='../checkpoints/graphpde')
    parser.add_argument('--output_dir', type=str, default='../figures/figure2')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*90)
    print("FINAL PUBLICATION-QUALITY MODEL COMPARISON")
    print("="*90)
    
    # Load all model results
    models_data = []
    
    model_paths = [
        (args.mlp_dir, 'MLP'),
        (args.gcn_dir, 'GCN'),
        (args.gnn_dir, 'GAT'),
        (args.graphsage_dir, 'GraphSage'),
        (args.deeponet_dir, 'DeepONet'),
        (args.tgcn_dir, 'TGCN'),
        (args.dcrnn_dir, 'DCRNN'),
        (args.graphpde_dir, 'GraphPDE'),
        (args.graphpde_no_residual_dir, 'GraphPDE No Residual')
    ]
    
    for model_dir, model_name in model_paths:
        if Path(model_dir).exists():
            print(f"Loading {model_name} results...")
            models_data.append(load_model_results(model_dir, model_name))
    
    if len(models_data) == 0:
        print("\nError: No model results found!")
        return
    
    print(f"\nComparing {len(models_data)} models...")
    
    # Create all outputs
    print("\n" + "-"*90)
    print("GENERATING PUBLICATION-QUALITY OUTPUTS")
    print("-"*90)
    
    comparison_df = create_performance_table(models_data, output_dir)
    create_clean_performance_comparison(comparison_df, output_dir)
    create_performance_heatmap(comparison_df, output_dir)
    create_improved_training_curves(models_data, output_dir)
    create_convergence_plot(models_data, output_dir)
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()