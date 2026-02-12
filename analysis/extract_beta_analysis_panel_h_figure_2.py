"""
Figure 2 - Panel H: Beta Analysis

Generates Panel H for Figure 2 analyzing physics vs neural contributions.
Outputs saved to: ./figures/figure2/
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Set PDF font configuration
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def load_model_and_extract_beta(checkpoint_path, model_creation_params):
    """
    Load trained model and extract β (residual_weight)

    Args:
        checkpoint_path: Path to saved checkpoint (e.g., 'best_model.pt')
        model_creation_params: Dict with model architecture parameters

    Returns:
        beta_value: The learned β weight
        full_checkpoint: Full checkpoint dict for further analysis
        model: Loaded model
    """
    from ..model_graphpde import create_graphpde_model

    # Create model with same architecture
    model = create_graphpde_model(**model_creation_params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Extract β (residual_weight)
    beta = model.reaction_module.residual_weight.item()

    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"✓ Neural residual weight (β): {beta:.6f}")

    return beta, checkpoint, model


def analyze_beta_contribution(model, test_loader, device='cuda', max_batches=100):
    """
    Decompose predictions into physics vs neural components

    Args:
        model: Trained GraphPDE model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        max_batches: Maximum number of batches to analyze

    Returns:
        analysis_dict: Dictionary with decomposition statistics
    """
    model.eval()
    model.to(device)

    physics_contributions = []
    neural_contributions = []
    total_reactions = []

    print(f"\nAnalyzing contributions from {max_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            if batch_idx % 20 == 0:
                print(f"  Processing batch {batch_idx}/{max_batches}...")

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Get full graph state
            full_population = batch['full_graph_population_t'].to(device)
            full_census = batch['full_graph_census'].to(device)
            node_indices = batch['node_idx']
            ethnicity_indices = batch['ethnicity']

            n_nodes = full_population.shape[0]
            full_node_idx = torch.arange(n_nodes, dtype=torch.long, device=device)

            # Get period indices
            year_t = batch['year_t']
            if isinstance(year_t, list):
                period_indices = torch.tensor(
                    [model.period_map[int(y)] for y in year_t],
                    dtype=torch.long, device=device
                )
            else:
                period_idx = model.period_map[int(year_t[0])]
                period_indices = torch.full(
                    (len(node_indices),), period_idx,
                    dtype=torch.long, device=device
                )

            full_period_idx = torch.zeros(n_nodes, dtype=torch.long, device=device)
            full_period_idx[node_indices] = period_indices

            # Save original beta
            original_beta = model.reaction_module.residual_weight.item()

            # Get physics-only reaction (β=0)
            model.reaction_module.residual_weight.data = torch.tensor(0.0).to(device)

            physics_reaction = model.reaction_module(
                full_population,
                model.graph_laplacian.node_to_city,
                full_census,
                full_node_idx,
                ethnicity_indices,
                full_period_idx,
                debug_batch=False
            )

            # Restore β and get full reaction
            model.reaction_module.residual_weight.data = torch.tensor(original_beta).to(device)

            full_reaction = model.reaction_module(
                full_population,
                model.graph_laplacian.node_to_city,
                full_census,
                full_node_idx,
                ethnicity_indices,
                full_period_idx,
                debug_batch=False
            )

            # Neural residual = full - physics
            neural_residual = full_reaction - physics_reaction

            # Extract for batch nodes
            physics_contrib = physics_reaction[node_indices, ethnicity_indices]
            neural_contrib = neural_residual[node_indices, ethnicity_indices]
            total_reaction = full_reaction[node_indices, ethnicity_indices]

            physics_contributions.append(physics_contrib.cpu().numpy())
            neural_contributions.append(neural_contrib.cpu().numpy())
            total_reactions.append(total_reaction.cpu().numpy())

    # Concatenate results
    physics_contrib = np.concatenate(physics_contributions)
    neural_contrib = np.concatenate(neural_contributions)
    total_reaction = np.concatenate(total_reactions)

    # Compute statistics - CORRECTED PERCENTAGES
    physics_mean_abs = np.mean(np.abs(physics_contrib))
    neural_mean_abs = np.mean(np.abs(neural_contrib))
    total_magnitude = physics_mean_abs + neural_mean_abs

    # Correct percentage calculation (sum to 100%)
    physics_percentage = (physics_mean_abs / total_magnitude) * 100
    neural_percentage = (neural_mean_abs / total_magnitude) * 100

    analysis = {
        'beta': original_beta,
        'n_samples': len(physics_contrib),
        # Physics statistics
        'physics_mean_abs': physics_mean_abs,
        'physics_std': np.std(physics_contrib),
        'physics_median': np.median(physics_contrib),
        'physics_percentage': physics_percentage,
        # Neural statistics
        'neural_mean_abs': neural_mean_abs,
        'neural_std': np.std(neural_contrib),
        'neural_median': np.median(neural_contrib),
        'neural_percentage': neural_percentage,
        # Total statistics
        'total_mean_abs': np.mean(np.abs(total_reaction)),
        'total_std': np.std(total_reaction),
        # Raw data for plotting
        'physics_contrib_raw': physics_contrib,
        'neural_contrib_raw': neural_contrib,
        'total_reaction_raw': total_reaction,
    }

    # Verification
    assert abs(physics_percentage + neural_percentage - 100.0) < 0.01, \
        "Percentages should sum to 100%"

    return analysis


def visualize_beta_analysis(analysis, save_path='beta_analysis.pdf'):
    """
    Create visualization of physics vs neural contributions with corrected percentages
    (Panel B - pie chart removed)
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(14, 10))

    # Create custom grid (3 rows, 2 columns - no panel B)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Color scheme
    physics_color = '#3498db'  # Blue
    neural_color = '#e74c3c'   # Red

    # ========================================================================
    # 1. Distribution comparison (top, spanning full width)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate histogram bins
    all_data = np.concatenate([
        analysis['physics_contrib_raw'],
        analysis['neural_contrib_raw']
    ])
    bins = np.linspace(np.percentile(all_data, 1),
                      np.percentile(all_data, 99), 60)

    ax1.hist(analysis['neural_contrib_raw'], bins=bins,
         alpha=0.65, label=f'Neural (β={analysis["beta"]:.3f})',
         color=neural_color, edgecolor='black')

    # PLOT PHYSICS SECOND (so it appears on top)
    ax1.hist(analysis['physics_contrib_raw'], bins=bins,
            alpha=0.65, label='Physics', color=physics_color, edgecolor='black')

    ax1.set_xlabel('Reaction Term Magnitude', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frequency (log scale)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Distribution of Physics vs Neural Contributions',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, framealpha=0.95)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)

    # Add vertical lines for means
    ax1.axvline(np.mean(analysis['neural_contrib_raw']),
                color=neural_color, linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(np.mean(analysis['physics_contrib_raw']),
                color=physics_color, linestyle='--', linewidth=2, alpha=0.8)

    # ========================================================================
    # 2. Box plot comparison (middle-left)
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    bp = ax2.boxplot(
        [analysis['physics_contrib_raw'], analysis['neural_contrib_raw']],
        labels=['Physics', 'Neural'],
        patch_artist=True,
        widths=0.6,
        showfliers=False  # Hide outliers for cleaner plot
    )

    bp['boxes'][0].set_facecolor(physics_color)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(neural_color)
    bp['boxes'][1].set_alpha(0.7)

    # Style whiskers and medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, linestyle='--')
    for median in bp['medians']:
        median.set(linewidth=2.5, color='black')

    ax2.set_ylabel('Contribution Magnitude', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Statistical Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.tick_params(labelsize=11)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # ========================================================================
    # 3. Violin plot (middle-right)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    data_for_violin = [analysis['physics_contrib_raw'], analysis['neural_contrib_raw']]
    parts = ax3.violinplot(
        data_for_violin,
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )

    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([physics_color, neural_color][i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Style the mean/median lines
    parts['cmeans'].set_color('darkgreen')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('darkred')
    parts['cmedians'].set_linewidth(2)

    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Physics', 'Neural'])
    ax3.set_ylabel('Contribution Magnitude', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Density Distribution', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.tick_params(labelsize=11)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # ========================================================================
    # 4. Scatter plot: Physics vs Neural contributions (bottom-left)
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    # Subsample for visualization (too many points otherwise)
    n_plot = min(5000, len(analysis['physics_contrib_raw']))
    indices = np.random.choice(len(analysis['physics_contrib_raw']), n_plot, replace=False)

    scatter = ax4.scatter(
        analysis['physics_contrib_raw'][indices],
        analysis['neural_contrib_raw'][indices],
        alpha=0.3,
        s=10,
        c=np.abs(analysis['physics_contrib_raw'][indices]),
        cmap='viridis'
    )

    # Add diagonal line
    lims = [
        min(ax4.get_xlim()[0], ax4.get_ylim()[0]),
        max(ax4.get_xlim()[1], ax4.get_ylim()[1]),
    ]
    ax4.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Equal contribution')

    ax4.set_xlabel('Physics Contribution', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Neural Contribution', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Physics vs Neural (per sample)', fontsize=13, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10)
    ax4.tick_params(labelsize=11)
    plt.colorbar(scatter, ax=ax4, label='|Physics|')

    # ========================================================================
    # 5. Cumulative distribution (bottom-right)
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    # Sort and compute cumulative distributions
    physics_sorted = np.sort(np.abs(analysis['physics_contrib_raw']))
    neural_sorted = np.sort(np.abs(analysis['neural_contrib_raw']))

    physics_cum = np.arange(1, len(physics_sorted) + 1) / len(physics_sorted)
    neural_cum = np.arange(1, len(neural_sorted) + 1) / len(neural_sorted)

    ax5.plot(physics_sorted, physics_cum, color=physics_color,
             linewidth=2.5, label='Physics', alpha=0.8)
    ax5.plot(neural_sorted, neural_cum, color=neural_color,
             linewidth=2.5, label='Neural', alpha=0.8)

    ax5.set_xlabel('|Contribution| Magnitude', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax5.set_title('(e) Cumulative Distribution Functions', fontsize=13, fontweight='bold', pad=10)
    ax5.legend(fontsize=11, framealpha=0.95)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.tick_params(labelsize=11)
    ax5.set_xscale('log')

    # Add main title
    fig.suptitle(
        'GraphPDE: Physics vs Neural Component Decomposition Analysis',
        fontsize=16, fontweight='bold', y=0.98
    )

    # Save as PDF with Type 42 fonts
    print(f"\nSaving visualization to: {save_path}")
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ PDF saved with Type 42 fonts")

    # Also save as PNG for quick viewing
    png_path = str(save_path).replace('.pdf', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ PNG saved to: {png_path}")

    plt.close()

    return fig


def print_analysis_summary(analysis):
    """Print detailed analysis summary to console"""
    print("\n" + "="*80)
    print("BETA ANALYSIS SUMMARY")
    print("="*80)

    print(f"\n{'LEARNED PARAMETER:':<30} β = {analysis['beta']:.6f}")
    print(f"{'Samples analyzed:':<30} {analysis['n_samples']:,}")

    print(f"\n{'PHYSICS COMPONENT:':<30}")
    print(f"{'  Mean |contribution|:':<30} {analysis['physics_mean_abs']:.4f}")
    print(f"{'  Standard deviation:':<30} {analysis['physics_std']:.4f}")
    print(f"{'  Median:':<30} {analysis['physics_median']:.4f}")
    print(f"{'  Contribution percentage:':<30} {analysis['physics_percentage']:.2f}%")

    print(f"\n{'NEURAL COMPONENT (β-weighted):':<30}")
    print(f"{'  Mean |contribution|:':<30} {analysis['neural_mean_abs']:.4f}")
    print(f"{'  Standard deviation:':<30} {analysis['neural_std']:.4f}")
    print(f"{'  Median:':<30} {analysis['neural_median']:.4f}")
    print(f"{'  Contribution percentage:':<30} {analysis['neural_percentage']:.2f}%")

    print(f"\n{'VERIFICATION:':<30}")
    print(f"{'  Percentages sum to:':<30} {analysis['physics_percentage'] + analysis['neural_percentage']:.2f}%")
    print(f"{'  Physics dominates:':<30} {'YES ✓' if analysis['physics_percentage'] > 50 else 'NO ✗'}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if analysis['physics_percentage'] > 70:
        print("\n✓✓ Physics STRONGLY dominates predictions (>70%)")
        print("   The model is highly interpretable - reaction-diffusion dynamics")
        print("   provide the primary predictive mechanism.")
    elif analysis['physics_percentage'] > 55:
        print("\n✓ Physics dominates predictions (>55%)")
        print("  The model maintains good interpretability - most predictions")
        print("  derive from parametric demographic processes.")
    else:
        print("\n⚖ Balanced contribution between physics and neural components")
        print("  The model uses both parametric and non-parametric mechanisms.")

    print(f"\nβ = {analysis['beta']:.3f} indicates that the neural residual was weighted")
    print(f"at {analysis['beta']*100:.0f}% of its raw magnitude during training.")

    if analysis['beta'] > 1.5:
        print("This relatively high β suggests substantial systematic deviations")
        print("from classical reaction-diffusion dynamics.")
    elif analysis['beta'] > 0.7:
        print("This moderate β suggests the model needed flexibility beyond")
        print("pure physics, but physics still provides the primary structure.")
    else:
        print("This low β suggests classical reaction-diffusion dynamics")
        print("capture most demographic patterns well.")

    print("\n" + "="*80 + "\n")


def main():
    """
    Main analysis script
    """
    # Paths
    checkpoint_dir = Path('../checkpoints/graphpde')
    checkpoint_path = checkpoint_dir / 'best_model.pt'

    # Output directory
    output_dir = Path('../figures/figure2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (no city filtering - trained on all cities)
    print("="*80)
    print("LOADING DATA AND MODEL")
    print("="*80)
    print("\nLoading data...")

    from ..dataloader_with_city_filter import create_dataloaders_with_city_filter

    train_loader, _, test_loader, graph = create_dataloaders_with_city_filter(
        train_path='./data/train_rd.pkl',
        val_path='./data/val_rd.pkl',
        test_path='./data/test_rd.pkl',
        graph_path='./data/graph_large_cities_rd.pkl',
        batch_size=2048,
        num_workers=4,
        city_filter=None  # No filtering - use all cities
    )

    # Get data statistics from graph
    adjacency = graph['adjacency']
    cities = graph['node_features']['city']
    unique_cities = sorted(set(cities))
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    node_to_city = np.array([city_to_idx[city] for city in cities])

    n_dauids = adjacency.shape[0]
    n_cities = len(unique_cities)
    n_ethnicities = 9
    n_census_features = 74

    print(f"\nGraph statistics:")
    print(f"  DAUIDs: {n_dauids:,}")
    print(f"  Cities: {n_cities}")
    print(f"  Ethnicities: {n_ethnicities}")

    # Model creation parameters
    model_params = {
        'n_ethnicities': n_ethnicities,
        'n_cities': n_cities,
        'n_dauids': n_dauids,
        'n_census_features': n_census_features,
        'adjacency': adjacency,
        'node_to_city': node_to_city,
        'n_ode_steps': 1,
        'integration_time': 5.0,
        'use_cuda_kernels': True
    }

    # Extract β
    print("\n" + "="*80)
    print("EXTRACTING β (NEURAL RESIDUAL WEIGHT)")
    print("="*80)

    beta, checkpoint, model = load_model_and_extract_beta(checkpoint_path, model_params)

    print(f"\nTest samples: {len(test_loader.dataset):,}")

    # Analyze contributions
    print("\n" + "="*80)
    print("DECOMPOSING PHYSICS vs NEURAL CONTRIBUTIONS")
    print("="*80)

    analysis = analyze_beta_contribution(
        model, test_loader, device='cuda', max_batches=100
    )

    # Print summary
    print_analysis_summary(analysis)

    # Visualize and save
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    pdf_path = output_dir / 'panel_h.pdf'
    fig = visualize_beta_analysis(analysis, save_path=pdf_path)

    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {str(pdf_path).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
