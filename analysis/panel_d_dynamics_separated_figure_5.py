"""
Figure 5 - Panel D: Pattern Formation Dynamics

Generates Panel D for Figure 5.
Outputs saved to: ./figures/figure5/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def create_panel_d_dynamics_separated(analyzer, focal_ethnicity, all_ethnicities, 
                                      output_base, dpi=600, figsize=(10, 7)):
    """
    Create publication-quality pattern formation dynamics visualization.
    Saves components separately for Photoshop assembly.
    """
    
    print("\n" + "="*80)
    print("PANEL D: PATTERN FORMATION DYNAMICS (SEPARATED COMPONENTS)")
    print("="*80)
    
    # Extract patterns WITH trajectory
    print("\nRunning 20-year simulation...")
    turing_data = analyzer.extract_turing_patterns(
        focal_ethnicity=focal_ethnicity,
        all_ethnicities=all_ethnicities,
        use_model=True,
        simulation_years=20,
        city_filter='Toronto'
    )
    
    trajectory = turing_data.get('trajectory')
    coords = turing_data['coords']
    params = turing_data['parameters']
    
    if trajectory is None or len(trajectory) < 2:
        print("ERROR: No trajectory data available!")
        print("Model simulation may not have saved intermediate states.")
        return None
    
    print(f"\nTrajectory: {len(trajectory)} timesteps")
    print(f"  Shape: {trajectory.shape}")
    
    # Analyze temporal evolution
    focal_idx = all_ethnicities.index(focal_ethnicity)
    n_steps = len(trajectory)
    
    # Compute metrics over time
    spatial_variance = []
    total_population = []
    max_concentration = []
    spatial_entropy = []
    clustering_coefficient = []
    
    print("\nComputing spatial metrics...")
    for t in range(n_steps):
        # Extract concentration at time t
        if len(trajectory.shape) == 3:
            conc_t = trajectory[t, :, focal_idx]
        else:
            conc_t = trajectory[t][:, focal_idx] if len(trajectory[t].shape) > 1 else trajectory[t]
        
        # Spatial variance (pattern strength)
        variance = np.var(conc_t)
        spatial_variance.append(variance)
        
        # Total population
        total_pop = np.sum(conc_t)
        total_population.append(total_pop)
        
        # Max concentration (enclave strength)
        max_conc = np.max(conc_t)
        max_concentration.append(max_conc)
        
        # Spatial entropy (dispersion measure)
        prob = conc_t / (np.sum(conc_t) + 1e-10)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        spatial_entropy.append(entropy)
        
        # Clustering coefficient (how clustered vs. dispersed)
        if len(conc_t) > 10:
            # Simple clustering: fraction of population in top 10% of locations
            threshold = np.percentile(conc_t, 90)
            clustered_pop = np.sum(conc_t[conc_t > threshold])
            clustering_coef = clustered_pop / (total_pop + 1e-10)
            clustering_coefficient.append(clustering_coef)
        else:
            clustering_coefficient.append(0)
    
    # Convert to arrays
    spatial_variance = np.array(spatial_variance)
    total_population = np.array(total_population)
    max_concentration = np.array(max_concentration)
    spatial_entropy = np.array(spatial_entropy)
    clustering_coefficient = np.array(clustering_coefficient)
    
    # Time array (assuming 20 years total)
    time_years = np.linspace(0, 20, n_steps)
    
    print(f"\nMetrics computed:")
    print(f"  Variance range: [{spatial_variance.min():.1f}, {spatial_variance.max():.1f}]")
    print(f"  Population range: [{total_population.min():.0f}, {total_population.max():.0f}]")
    print(f"  Clustering range: [{clustering_coefficient.min():.3f}, {clustering_coefficient.max():.3f}]")
    
    # === IDENTIFY PATTERN FORMATION PHASES ===
    
    # Phase 1: Initial equilibration (first 20% of time)
    equilibration_end = int(n_steps * 0.2)
    
    # Phase 2: Pattern emergence (variance increases significantly)
    variance_baseline = np.mean(spatial_variance[:equilibration_end])
    variance_std = np.std(spatial_variance[:equilibration_end])
    emergence_threshold = variance_baseline + 2 * variance_std
    
    # Find when variance crosses threshold
    emergence_mask = spatial_variance > emergence_threshold
    if np.any(emergence_mask):
        emergence_start = np.argmax(emergence_mask)
    else:
        emergence_start = equilibration_end
    
    # Phase 3: Pattern maturation (variance stabilizes)
    variance_derivative = np.gradient(spatial_variance)
    variance_accel = np.gradient(variance_derivative)
    
    maturation_start = equilibration_end
    if emergence_start < n_steps - 10:
        search_range = range(emergence_start + 5, n_steps - 5)
        for i in search_range:
            window = variance_accel[i:i+5]
            if np.abs(np.mean(window)) < 0.1 * np.abs(variance_accel).max():
                maturation_start = i
                break
    
    print(f"\nPattern formation phases:")
    print(f"  Phase 1 (Equilibration): 0 - {time_years[equilibration_end]:.1f} years")
    print(f"  Phase 2 (Emergence): {time_years[emergence_start]:.1f} - {time_years[maturation_start]:.1f} years")
    print(f"  Phase 3 (Maturation): {time_years[maturation_start]:.1f} - 20.0 years")
    
    # =============================================================================
    # COMPONENT 1: MAIN PLOT (with phase backgrounds and data lines)
    # =============================================================================
    
    print("\n[1/5] Creating main plot...")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    ax1 = fig.add_axes([0.12, 0.15, 0.75, 0.70])
    ax2 = ax1.twinx()
    
    # Phase backgrounds
    ax1.axvspan(time_years[0], time_years[equilibration_end],
               alpha=0.15, color='#3498DB', zorder=0)
    ax1.axvspan(time_years[emergence_start], time_years[maturation_start],
               alpha=0.15, color='#E74C3C', zorder=0)
    ax1.axvspan(time_years[maturation_start], time_years[-1],
               alpha=0.15, color='#27AE60', zorder=0)
    
    # Spatial variance (gradient line)
    from matplotlib.collections import LineCollection
    points = np.array([time_years, spatial_variance]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(time_years.min(), time_years.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=3.5, zorder=5)
    lc.set_array(time_years)
    ax1.add_collection(lc)
    
    # Key point markers
    ax1.scatter(time_years[equilibration_end], spatial_variance[equilibration_end],
               s=120, c='#3498DB', edgecolors='white', linewidths=2, zorder=10, marker='o')
    ax1.scatter(time_years[emergence_start], spatial_variance[emergence_start],
               s=120, c='#E74C3C', edgecolors='white', linewidths=2, zorder=10, marker='s')
    ax1.scatter(time_years[maturation_start], spatial_variance[maturation_start],
               s=120, c='#27AE60', edgecolors='white', linewidths=2, zorder=10, marker='^')
    
    # Threshold line
    ax1.axhline(emergence_threshold, color='#E74C3C', linestyle='--',
               linewidth=2, alpha=0.6, zorder=3)
    
    # Total population
    ax2.plot(time_years, total_population / 1000,
            color='#34495E', linewidth=2.5, linestyle=':', alpha=0.7, zorder=4)
    
    # Styling
    ax1.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Spatial Variance (Pattern Strength)', fontsize=14, fontweight='bold', color='black')
    ax2.set_ylabel('Total Population (thousands)', fontsize=14, fontweight='bold', color='#34495E')
    ax1.set_title('Pattern Formation Dynamics During 20-Year Simulation',
                 fontsize=15, fontweight='bold', pad=15, color='#2C3E50')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.tick_params(axis='y', labelcolor='#34495E', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='gray', zorder=1)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, spatial_variance.max() * 1.15)
    ax2.set_ylim(total_population.min() / 1000 * 0.9, total_population.max() / 1000 * 1.1)
    
    output_1 = f"{output_base}_d1_main.png"
    plt.savefig(output_1, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_1).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel d1'})
    plt.close()
    print(f"  ✓ Saved: {output_1}")
    
    # =============================================================================
    # COMPONENT 2: CLUSTERING INSET (transparent background)
    # =============================================================================
    
    print("\n[2/5] Creating clustering inset...")
    
    fig = plt.figure(figsize=(4, 3), dpi=dpi, facecolor='none')
    ax_inset = fig.add_axes([0.15, 0.20, 0.75, 0.68])
    
    ax_inset.plot(time_years, clustering_coefficient, 
                 color='#9B59B6', linewidth=2.5, marker='o', markersize=3)
    ax_inset.set_xlabel('Time (yr)', fontsize=11, fontweight='bold')
    ax_inset.set_ylabel('Clustering\nCoefficient', fontsize=11, fontweight='bold')
    ax_inset.set_title('Spatial Clustering', fontsize=12, fontweight='bold', pad=8)
    ax_inset.grid(True, alpha=0.3, linewidth=0.6)
    ax_inset.tick_params(labelsize=10)
    ax_inset.set_xlim(0, 20)
    ax_inset.set_ylim(0, 1)
    ax_inset.set_facecolor('white')
    
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('#9B59B6')
        spine.set_linewidth(2.5)
    
    output_2 = f"{output_base}_d2_inset.png"
    plt.savefig(output_2, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_2).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel d2'})
    plt.close()
    print(f"  ✓ Saved: {output_2}")
    
    # =============================================================================
    # COMPONENT 3: PHASE ANNOTATIONS (transparent background)
    # =============================================================================
    
    print("\n[3/5] Creating phase annotations...")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='none')
    ax = fig.add_axes([0.12, 0.15, 0.75, 0.70])
    
    # Make axes invisible but preserve coordinate system
    ax.set_xlim(0, 20)
    ax.set_ylim(0, spatial_variance.max() * 1.15)
    ax.axis('off')
    
    # Phase labels
    ax.text(time_years[equilibration_end//2], spatial_variance.max() * 1.08,
            'Phase 1:\nEquilibration', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#3498DB',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#3498DB', linewidth=2, alpha=0.95))
    
    mid_emergence = (emergence_start + maturation_start) // 2
    ax.text(time_years[mid_emergence], spatial_variance.max() * 1.08,
            'Phase 2:\nEmergence', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#E74C3C',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#E74C3C', linewidth=2, alpha=0.95))
    
    mid_maturation = (maturation_start + n_steps - 1) // 2
    ax.text(time_years[mid_maturation], spatial_variance.max() * 1.08,
            'Phase 3:\nMaturation', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#27AE60',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#27AE60', linewidth=2, alpha=0.95))
    
    output_3 = f"{output_base}_d3_annotations.png"
    plt.savefig(output_3, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_3).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel d3'})
    plt.close()
    print(f"  ✓ Saved: {output_3}")
    
    # =============================================================================
    # COMPONENT 4: STATISTICS BOXES (transparent background)
    # =============================================================================
    
    print("\n[4/5] Creating statistics boxes...")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='none')
    ax = fig.add_axes([0.12, 0.15, 0.75, 0.70])
    
    # Make axes invisible but preserve coordinate system
    ax.set_xlim(0, 20)
    ax.set_ylim(0, spatial_variance.max() * 1.15)
    ax.axis('off')
    
    # Statistics box (bottom right)
    initial_var = spatial_variance[0]
    final_var = spatial_variance[-1]
    variance_increase = (final_var - initial_var) / initial_var * 100
    
    stats_text = (f'Initial Variance: {initial_var:.1f}\n'
                  f'Final Variance: {final_var:.1f}\n'
                  f'Change: +{variance_increase:.0f}%\n'
                  f'Population: {total_population[-1]:.0f}')
    
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                     edgecolor='#2C3E50', linewidth=2, alpha=0.95))
    
    # Model info (bottom left)
    D = params['diffusion_coefficients']  # Already averaged over cities
    D_focal = np.abs(D[focal_idx])
    
    info_text = f'GraphPDE Model\nD = {D_focal:.4f} km²/yr'
    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='gray', linewidth=1.5, alpha=0.95))
    
    output_4 = f"{output_base}_d4_stats.png"
    plt.savefig(output_4, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_4).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel d4'})
    plt.close()
    print(f"  ✓ Saved: {output_4}")
    
    # =============================================================================
    # COMPONENT 5: PANEL LABEL (transparent background)
    # =============================================================================
    
    print("\n[5/5] Creating panel label...")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='none')
    ax = fig.add_axes([0.12, 0.15, 0.75, 0.70])
    
    # Make axes invisible but preserve coordinate system
    ax.set_xlim(0, 20)
    ax.set_ylim(0, spatial_variance.max() * 1.15)
    ax.axis('off')
    
    # Panel label
    ax.text(0.02, 0.98, 'D', transform=ax.transAxes,
            fontsize=36, fontweight='bold', va='top', ha='left',
            color='white',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#2C3E50',
                     edgecolor='white', linewidth=3, alpha=0.98))
    
    output_5 = f"{output_base}_d5_panel_label.png"
    plt.savefig(output_5, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_5).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel d5'})
    plt.close()
    print(f"  ✓ Saved: {output_5}")
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("\n" + "="*80)
    print("COMPONENTS SAVED")
    print("="*80)
    print(f"\nBase path: {output_base}")
    print("\nFiles created:")
    print(f"  1. {output_base}_d1_main.png        - Main plot with data")
    print(f"  2. {output_base}_d2_inset.png       - Clustering inset (transparent)")
    print(f"  3. {output_base}_d3_annotations.png - Phase labels (transparent)")
    print(f"  4. {output_base}_d4_stats.png       - Statistics boxes (transparent)")
    print(f"  5. {output_base}_d5_panel_label.png - Panel 'D' label (transparent)")
    print("\nPhotoshop Assembly Instructions:")
    print("  1. Import d1_main.png as base layer")
    print("  2. Add d2_inset.png - position in upper right")
    print("  3. Add d3_annotations.png - align with main plot")
    print("  4. Add d4_stats.png - align with main plot")
    print("  5. Add d5_panel_label.png - align with main plot")
    print("  6. Adjust opacity/position as needed to avoid overlaps")
    
    return {
        'n_timesteps': n_steps,
        'variance_change': variance_increase,
        'emergence_time': time_years[emergence_start],
        'maturation_time': time_years[maturation_start],
        'output_files': [output_1, output_2, output_3, output_4, output_5]
    }


def main():
    parser = argparse.ArgumentParser(description='Create Panel D - Pattern Dynamics (Separated)')
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
                       default='../figures/figure5/panel_d',
                       help='Base path for output files (without extension)')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 7])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PANEL D: PATTERN FORMATION DYNAMICS (SEPARATED COMPONENTS)")
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
    
    results = create_panel_d_dynamics_separated(
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