"""
Figure 5 - Panel E: Multi-scale Pattern Analysis

Generates Panel E for Figure 5.
Outputs saved to: ./figures/figure5/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy import ndimage
from scipy.signal import find_peaks
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def create_panel_e_pattern_characteristics_separated(analyzer, focal_ethnicity, all_ethnicities,
                                                     output_base, dpi=600, figsize=(10, 8)):
    """
    Create publication-quality multi-scale pattern analysis.
    Saves components separately for Photoshop assembly.
    """
    
    print("\n" + "="*80)
    print("PANEL E: MULTI-SCALE PATTERN ANALYSIS (SEPARATED COMPONENTS)")
    print("="*80)
    
    # Extract patterns
    turing_data = analyzer.extract_turing_patterns(
        focal_ethnicity=focal_ethnicity,
        all_ethnicities=all_ethnicities,
        use_model=True,
        simulation_years=20,
        city_filter='Toronto'
    )
    
    coords = turing_data['coords']
    conc = turing_data['concentration']
    
    print(f"\nData: {len(coords)} DAUIDs")
    print(f"Concentration range: [{conc.min():.1f}, {conc.max():.1f}]")
    
    if conc.sum() == 0:
        print("ERROR: No population data!")
        return None
    
    # === MULTI-THRESHOLD CLUSTER ANALYSIS ===
    print("\nPerforming multi-scale cluster detection...")
    
    # Create high-resolution grid for proper analysis
    grid_size = 200
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid_size)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate to grid
    Zi = griddata(coords, conc, (Xi, Yi), method='linear', fill_value=0)
    
    # Analyze at multiple thresholds
    thresholds_percentiles = [25, 40, 55, 70, 85]
    
    all_clusters = []
    
    for pct in thresholds_percentiles:
        threshold = np.percentile(conc[conc > 0], pct) if conc.sum() > 0 else 0
        
        if threshold == 0:
            continue
        
        print(f"  Threshold {pct}th percentile ({threshold:.1f})...")
        
        binary_grid = Zi > threshold
        
        # Morphological operations to clean up
        struct = ndimage.generate_binary_structure(2, 2)
        binary_grid = ndimage.binary_closing(binary_grid, structure=struct, iterations=2)
        binary_grid = ndimage.binary_opening(binary_grid, structure=struct, iterations=1)
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_grid)
        
        print(f"    Found {num_features} clusters")
        
        for i in range(1, num_features + 1):
            cluster_mask = labeled_array == i
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size < 5:  # Skip tiny clusters
                continue
            
            # Compute cluster metrics
            cluster_values = Zi[cluster_mask]
            
            cluster_mean_density = np.mean(cluster_values)
            cluster_max_density = np.max(cluster_values)
            cluster_total_pop = cluster_values.sum()
            cluster_area_pct = cluster_size / (grid_size * grid_size) * 100
            
            # Cluster shape metrics
            y_coords, x_coords = np.where(cluster_mask)
            cluster_centroid = (np.mean(x_coords), np.mean(y_coords))
            
            # Eccentricity (how elongated)
            if len(x_coords) > 2:
                cov = np.cov(x_coords, y_coords)
                eigenvals, eigenvecs = np.linalg.eig(cov)
                eccentricity = np.sqrt(1 - np.min(eigenvals) / (np.max(eigenvals) + 1e-10))
            else:
                eccentricity = 0
            
            # Compactness (circularity)
            perimeter = np.sum(ndimage.binary_dilation(cluster_mask) & ~cluster_mask)
            compactness = 4 * np.pi * cluster_size / (perimeter**2 + 1e-10)
            
            all_clusters.append({
                'size': cluster_size,
                'mean_density': cluster_mean_density,
                'max_density': cluster_max_density,
                'total_pop': cluster_total_pop,
                'area_pct': cluster_area_pct,
                'threshold_pct': pct,
                'eccentricity': eccentricity,
                'compactness': compactness,
                'centroid': cluster_centroid
            })
    
    if len(all_clusters) == 0:
        print("ERROR: No clusters detected!")
        return None
    
    print(f"\nTotal clusters found: {len(all_clusters)}")
    
    # Convert to arrays for analysis
    cluster_sizes = np.array([c['size'] for c in all_clusters])
    cluster_densities = np.array([c['mean_density'] for c in all_clusters])
    cluster_areas = np.array([c['area_pct'] for c in all_clusters])
    cluster_eccentricities = np.array([c['eccentricity'] for c in all_clusters])
    cluster_compactnesses = np.array([c['compactness'] for c in all_clusters])
    cluster_thresholds = np.array([c['threshold_pct'] for c in all_clusters])
    
    # Normalize densities to max
    max_density = np.max(cluster_densities)
    cluster_densities_norm = cluster_densities / max_density
    
    # === PATTERN CLASSIFICATION ===
    print("\nClassifying pattern type...")
    
    n_clusters = len(all_clusters)
    avg_size = np.mean(cluster_sizes)
    std_size = np.std(cluster_sizes)
    cv_size = std_size / (avg_size + 1e-10)  # Coefficient of variation
    
    avg_eccentricity = np.mean(cluster_eccentricities)
    avg_compactness = np.mean(cluster_compactnesses)
    
    # Classification logic
    pattern_scores = {
        'Spots': 0,
        'Stripes': 0,
        'Labyrinthine': 0,
        'Large Patches': 0,
        'Scattered': 0
    }
    
    # Spots: many small, compact, circular clusters
    if n_clusters > 20 and avg_compactness > 0.4 and avg_eccentricity < 0.6:
        pattern_scores['Spots'] += 3
    elif n_clusters > 10:
        pattern_scores['Spots'] += 1
    
    # Stripes: elongated clusters
    if avg_eccentricity > 0.7:
        pattern_scores['Stripes'] += 3
    elif avg_eccentricity > 0.6:
        pattern_scores['Stripes'] += 1
    
    # Labyrinthine: intermediate complexity
    if 10 <= n_clusters <= 30 and 0.4 < avg_eccentricity < 0.7:
        pattern_scores['Labyrinthine'] += 3
    
    # Large patches: few large clusters
    if n_clusters < 10 and avg_size > 100:
        pattern_scores['Large Patches'] += 3
    elif n_clusters < 15:
        pattern_scores['Large Patches'] += 1
    
    # Scattered: many small clusters with high variation
    if n_clusters > 30 and cv_size > 0.8:
        pattern_scores['Scattered'] += 2
    
    # Determine dominant pattern
    dominant_pattern = max(pattern_scores, key=pattern_scores.get)
    confidence = pattern_scores[dominant_pattern] / sum(pattern_scores.values()) if sum(pattern_scores.values()) > 0 else 0
    
    print(f"  Pattern type: {dominant_pattern} (confidence: {confidence:.2%})")
    print(f"  Scores: {pattern_scores}")
    print(f"  Statistics:")
    print(f"    Clusters: {n_clusters}")
    print(f"    Avg size: {avg_size:.1f} ± {std_size:.1f} (CV={cv_size:.2f})")
    print(f"    Avg eccentricity: {avg_eccentricity:.3f}")
    print(f"    Avg compactness: {avg_compactness:.3f}")
    
    # Fit power-law relationship (for later use)
    slope, intercept = None, None
    sizes_fit, densities_fit = None, None
    if len(cluster_sizes) > 5:
        log_sizes = np.log10(cluster_sizes + 1)
        log_densities = np.log10(cluster_densities_norm + 0.01)
        coeffs = np.polyfit(log_sizes, log_densities, 1)
        slope, intercept = coeffs
        sizes_fit = np.logspace(np.log10(cluster_sizes.min()), 
                               np.log10(cluster_sizes.max()), 100)
        densities_fit = 10**(slope * np.log10(sizes_fit) + intercept)
    
    # =============================================================================
    # COMPONENT 1: MAIN PLOT (scatter, marginals, hexbin, fit line)
    # =============================================================================
    
    print("\n[1/4] Creating main plot...")
    
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    gs = GridSpec(4, 4, figure=fig, left=0.12, right=0.96, top=0.92, bottom=0.10,
                  hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    
    # Main hexbin plot
    hexbin = ax_main.hexbin(
        cluster_sizes, cluster_densities_norm,
        C=cluster_areas, gridsize=30,
        cmap='YlOrRd', alpha=0.8,
        mincnt=1, reduce_C_function=np.mean,
        edgecolors='white', linewidths=0.5
    )
    
    # Overlay scatter
    scatter = ax_main.scatter(
        cluster_sizes, cluster_densities_norm,
        c=cluster_thresholds, s=40,
        cmap='viridis', alpha=0.6,
        edgecolors='black', linewidths=0.5,
        zorder=5
    )
    
    # Fit line
    if slope is not None:
        ax_main.plot(sizes_fit, densities_fit, 'r--', linewidth=2.5,
                    alpha=0.8, label=f'Power law: ρ ∝ S^{slope:.2f}',
                    zorder=10)
    
    # Styling
    ax_main.set_xlabel('Cluster Size (grid points)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Normalized Mean Density', fontsize=13, fontweight='bold')
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    ax_main.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax_main.tick_params(labelsize=11)
    
    # Top marginal
    ax_top.hist(cluster_sizes, bins=30, color='#3498DB', alpha=0.7,
               edgecolor='black', linewidth=0.8)
    ax_top.axvline(avg_size, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {avg_size:.0f}')
    ax_top.set_ylabel('Count', fontsize=10)
    ax_top.legend(fontsize=9, loc='upper right')
    ax_top.tick_params(labelbottom=False, labelsize=10)
    ax_top.grid(True, alpha=0.3, axis='y')
    
    # Right marginal
    ax_right.hist(cluster_densities_norm, bins=30, orientation='horizontal',
                 color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_right.axhline(np.mean(cluster_densities_norm), color='red',
                    linestyle='--', linewidth=2)
    ax_right.set_xlabel('Count', fontsize=10)
    ax_right.tick_params(labelleft=False, labelsize=10)
    ax_right.grid(True, alpha=0.3, axis='x')
    
    # Title
    fig.suptitle(f'Multi-Scale Pattern Analysis: {focal_ethnicity} Settlement',
                fontsize=15, fontweight='bold', y=0.98, color='#2C3E50')
    
    output_1 = f"{output_base}_e1_main.png"
    plt.savefig(output_1, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_1).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel e1'})
    plt.close()
    print(f"  ✓ Saved: {output_1}")
    
    # =============================================================================
    # COMPONENT 2: COLORBARS (transparent background)
    # =============================================================================
    
    print("\n[2/4] Creating colorbars...")
    
    # Recreate just for colorbars
    fig = plt.figure(figsize=(8, 2), dpi=dpi, facecolor='none')
    
    # Create temporary axes for colorbars
    ax_temp = fig.add_axes([0.1, 0.5, 0.35, 0.3])
    
    # Recreate hexbin temporarily
    hexbin_temp = ax_temp.hexbin(
        cluster_sizes, cluster_densities_norm,
        C=cluster_areas, gridsize=30,
        cmap='YlOrRd', alpha=0.8,
        mincnt=1, reduce_C_function=np.mean
    )
    
    # Colorbar 1 (hexbin - area)
    cbar1_ax = fig.add_axes([0.15, 0.25, 0.25, 0.05])
    cbar1 = fig.colorbar(hexbin_temp, cax=cbar1_ax, orientation='horizontal')
    cbar1.set_label('Cluster Area (%)', fontsize=11, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    # Recreate scatter temporarily
    scatter_temp = ax_temp.scatter(
        cluster_sizes, cluster_densities_norm,
        c=cluster_thresholds, s=40,
        cmap='viridis', alpha=0.6
    )
    
    # Colorbar 2 (scatter - threshold)
    cbar2_ax = fig.add_axes([0.60, 0.3, 0.25, 0.04])
    cbar2 = fig.colorbar(scatter_temp, cax=cbar2_ax, orientation='horizontal')
    cbar2.set_label('Threshold (%ile)', fontsize=10, fontweight='bold')
    cbar2.ax.tick_params(labelsize=9)
    
    # Remove temporary axes
    ax_temp.remove()
    
    output_2 = f"{output_base}_e2_colorbars.png"
    plt.savefig(output_2, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_2).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel e2'})
    plt.close()
    print(f"  ✓ Saved: {output_2}")
    
    # =============================================================================
    # COMPONENT 3: ANNOTATIONS (transparent background)
    # =============================================================================
    
    print("\n[3/4] Creating annotations...")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='none')
    gs = GridSpec(4, 4, figure=fig, left=0.12, right=0.96, top=0.92, bottom=0.10,
                  hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_main.set_xlim(cluster_sizes.min() * 0.8, cluster_sizes.max() * 1.2)
    ax_main.set_ylim(cluster_densities_norm.min() * 0.8, cluster_densities_norm.max() * 1.2)
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.axis('off')
    
    # Pattern classification box (top right)
    pattern_text = (f'Pattern Type: {dominant_pattern}\n'
                   f'Confidence: {confidence:.0%}\n'
                   f'Clusters: {n_clusters}\n'
                   f'Avg Size: {avg_size:.0f}\n'
                   f'Regularity: {1-cv_size:.2f}')
    
    ax_main.text(0.98, 0.98, pattern_text,
                transform=ax_main.transAxes, ha='right', va='top',
                fontsize=11, family='monospace', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                         edgecolor='#2C3E50', linewidth=2, alpha=0.95),
                zorder=15)
    
    # Shape metrics box (bottom left)
    shape_text = (f'Shape Metrics:\n'
                 f'Eccentricity: {avg_eccentricity:.3f}\n'
                 f'Compactness: {avg_compactness:.3f}')
    
    ax_main.text(0.02, 0.02, shape_text,
                transform=ax_main.transAxes, ha='left', va='bottom',
                fontsize=10, style='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='gray', linewidth=1.5, alpha=0.95),
                zorder=15)
    
    output_3 = f"{output_base}_e3_annotations.png"
    plt.savefig(output_3, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_3).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel e3'})
    plt.close()
    print(f"  ✓ Saved: {output_3}")
    
    # =============================================================================
    # COMPONENT 4: PANEL LABEL (transparent background)
    # =============================================================================
    
    print("\n[4/4] Creating panel label...")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='none')
    gs = GridSpec(4, 4, figure=fig, left=0.12, right=0.96, top=0.92, bottom=0.10,
                  hspace=0.05, wspace=0.05)
    
    ax_top = fig.add_subplot(gs[0, :-1])
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')
    
    # Panel label
    ax_top.text(0.02, 0.85, 'E', transform=ax_top.transAxes,
               fontsize=36, fontweight='bold', va='top', ha='left',
               color='white',
               bbox=dict(boxstyle='round,pad=0.7', facecolor='#2C3E50',
                        edgecolor='white', linewidth=3, alpha=0.98),
               zorder=15)
    
    output_4 = f"{output_base}_e4_panel_label.png"
    plt.savefig(output_4, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_4).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel e4'})
    plt.close()
    print(f"  ✓ Saved: {output_4}")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("\n" + "="*80)
    print("COMPONENTS SAVED")
    print("="*80)
    print(f"\nBase path: {output_base}")
    print("\nFiles created:")
    print(f"  1. {output_base}_e1_main.png        - Main plot with marginals")
    print(f"  2. {output_base}_e2_colorbars.png   - Colorbars (transparent)")
    print(f"  3. {output_base}_e3_annotations.png - Text boxes (transparent)")
    print(f"  4. {output_base}_e4_panel_label.png - Panel 'E' label (transparent)")
    print("\nPhotoshop Assembly Instructions:")
    print("  1. Import e1_main.png as base layer")
    print("  2. Add e2_colorbars.png - position below/beside main plot")
    print("  3. Add e3_annotations.png - align with main plot")
    print("  4. Add e4_panel_label.png - align with top marginal")
    print("  5. Adjust positions to avoid any overlaps")
    
    return {
        'n_clusters': n_clusters,
        'pattern_type': dominant_pattern,
        'confidence': confidence,
        'avg_size': avg_size,
        'cv_size': cv_size,
        'output_files': [output_1, output_2, output_3, output_4]
    }


def main():
    parser = argparse.ArgumentParser(description='Create Panel E - Pattern Characteristics (Separated)')
    parser.add_argument('--checkpoint', type=str,
                       default='../checkpoints/graphpde/best_model.pt')
    parser.add_argument('--graph', type=str,
                       default='../data/graph_large_cities_rd.pkl')
    parser.add_argument('--da_info', type=str,
                       default='../data/da_canada.csv')
    parser.add_argument('--ethnicity', type=str, default='China')
    parser.add_argument('--ethnicities', type=str, nargs='+',
                       default=['China', 'Philippines', 'India', 'Pakistan',
                               'Iran', 'Sri Lanka', 'Portugal', 'Italy',
                               'United Kingdom'])
    parser.add_argument('--output', type=str,
                       default='../figures/figure5/panel_e',
                       help='Base path for output files (without extension)')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 8])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PANEL E: PATTERN CHARACTERISTICS (SEPARATED COMPONENTS)")
    print("="*80)
    
    analyzer = GraphPDEAnalyzer(
        checkpoint_path=args.checkpoint,
        graph_path=args.graph,
        da_info_path=args.da_info,
        device=args.device
    )
    
    analyzer.set_ethnicities(args.ethnicities)
    
    # Create output directory
    output_base = Path(args.output)
    output_base.parent.mkdir(exist_ok=True, parents=True)
    
    results = create_panel_e_pattern_characteristics_separated(
        analyzer,
        focal_ethnicity=args.ethnicity,
        all_ethnicities=args.ethnicities,
        output_base=str(output_base),
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )
    
    if results is None:
        print("\nFAILED")
        return
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)


if __name__ == "__main__":
    main()
