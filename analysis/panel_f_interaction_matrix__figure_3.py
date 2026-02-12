"""
Figure 3 - Panel F: Learned Ethnic Interaction Matrix

Generates Panel F for Figure 3.
Outputs saved to: ./figures/figure3/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def create_panel_f_interaction_matrix_separated(analyzer, focal_ethnicity, all_ethnicities,
                                                output_base, dpi=600, figsize=(11, 9)):
    """
    Create interaction matrix with two-tier color scaling.
    """
    
    print("\n" + "="*80)
    print("PANEL F: INTERACTION MATRIX (TWO-TIER SCALING)")
    print("="*80)
    
    # Extract patterns
    turing_data = analyzer.extract_turing_patterns(
        focal_ethnicity=focal_ethnicity,
        all_ethnicities=all_ethnicities,
        use_model=True,
        simulation_years=0,
        city_filter='Toronto'
    )
    
    params = turing_data['parameters']
    W = params['interaction_matrix']
    
    print(f"\nRaw interaction matrix:")
    print(f"  Diagonal: [{np.diag(W).min():.6f}, {np.diag(W).max():.6f}]")
    
    W_offdiag = W.copy()
    np.fill_diagonal(W_offdiag, 0)
    print(f"  Off-diagonal: [{W_offdiag[W_offdiag != 0].min():.6f}, {W_offdiag[W_offdiag != 0].max():.6f}]")
    
    # Scale to percentages
    W_display = W * 100
    
    focal_idx = all_ethnicities.index(focal_ethnicity)
    
    # === HIERARCHICAL CLUSTERING ===
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    distance_matrix = 1 - np.abs(np.corrcoef(W))
    distance_matrix = np.clip(distance_matrix, 0, 2)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    cluster_order = dendro['leaves']
    
    W_display_ordered = W_display[cluster_order, :][:, cluster_order]
    ethnicities_ordered = [all_ethnicities[i] for i in cluster_order]
    focal_idx_ordered = ethnicities_ordered.index(focal_ethnicity)
    
    print(f"\nClustered order: {ethnicities_ordered}")
    
    # === TWO-TIER COLOR MAPPING ===
    
    # Separate diagonal and off-diagonal for independent scaling
    W_diag_ordered = np.diag(W_display_ordered)
    W_offdiag_ordered = W_display_ordered.copy()
    np.fill_diagonal(W_offdiag_ordered, 0)
    
    print(f"\nScaled for display (×100):")
    print(f"  Diagonal: [{W_diag_ordered.min():.2f}, {W_diag_ordered.max():.2f}]")
    print(f"  Off-diagonal: [{W_offdiag_ordered[W_offdiag_ordered != 0].min():.2f}, {W_offdiag_ordered[W_offdiag_ordered != 0].max():.2f}]")
    
    # Color scales
    # Diagonal: all negative, use single blue scale
    diag_vmin = W_diag_ordered.min()
    diag_vmax = W_diag_ordered.max()
    diag_cmap = LinearSegmentedColormap.from_list('diag',
        ['#08519C', '#3182BD', '#6BAED6', '#9ECAE1', '#C6DBEF'])
    
    # Off-diagonal: diverging around zero
    offdiag_max = max(abs(W_offdiag_ordered.min()), abs(W_offdiag_ordered.max()))
    offdiag_vmin = -offdiag_max
    offdiag_vmax = offdiag_max
    offdiag_cmap = LinearSegmentedColormap.from_list('offdiag',
        ['#2166AC', '#67A9CF', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#EF8A62', '#B2182B'])
    
    print(f"\nColor scales:")
    print(f"  Diagonal: [{diag_vmin:.2f}, {diag_vmax:.2f}]")
    print(f"  Off-diagonal: [{offdiag_vmin:.2f}, {offdiag_vmax:.2f}]")
    
    # === COMPONENT 1: MAIN HEATMAP WITH TWO-TIER COLORING ===
    
    print("\n[1/5] Creating main heatmap with two-tier coloring...")
    
    fig, ax_main = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
    
    n_eth = len(ethnicities_ordered)
    
    # Create custom colored squares for each cell
    for i in range(n_eth):
        for j in range(n_eth):
            value = W_display_ordered[i, j]
            
            if i == j:
                # Diagonal: use diagonal colormap
                norm_val = (value - diag_vmin) / (diag_vmax - diag_vmin + 1e-10)
                color = diag_cmap(norm_val)
            else:
                # Off-diagonal: use diverging colormap
                norm_val = (value - offdiag_vmin) / (offdiag_vmax - offdiag_vmin + 1e-10)
                color = offdiag_cmap(norm_val)
            
            # Draw rectangle
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                           facecolor=color, edgecolor='white', linewidth=2)
            ax_main.add_patch(rect)
            
            # Add text annotation
            if i == j:
                # Diagonal: bold, white text on dark blue
                ax_main.text(j, i, f'{value:.1f}',
                           ha='center', va='center',
                           fontsize=11, fontweight='bold',
                           color='white')
            else:
                # Off-diagonal: smaller text, color based on value
                if abs(value) > offdiag_max * 0.3:
                    text_color = 'white' if abs(value) > offdiag_max * 0.6 else 'black'
                else:
                    text_color = 'black'
                
                if abs(value) > 0.01:  # Only show if non-negligible
                    ax_main.text(j, i, f'{value:.2f}',
                               ha='center', va='center',
                               fontsize=8,
                               color=text_color)
    
    # Highlight focal ethnicity
    # Row highlight
    rect_row = Rectangle((-0.5, focal_idx_ordered - 0.5), n_eth, 1,
                        linewidth=5, edgecolor='gold', facecolor='none', 
                        linestyle='-', zorder=10)
    ax_main.add_patch(rect_row)
    
    # Column highlight
    rect_col = Rectangle((focal_idx_ordered - 0.5, -0.5), 1, n_eth,
                        linewidth=5, edgecolor='gold', facecolor='none',
                        linestyle='-', zorder=10)
    ax_main.add_patch(rect_col)
    
    # Labels
    eth_labels = [eth[:8] + '..' if len(eth) > 10 else eth
                  for eth in ethnicities_ordered]
    
    ax_main.set_xlim(-0.5, n_eth - 0.5)
    ax_main.set_ylim(n_eth - 0.5, -0.5)
    
    ax_main.set_xticks(range(n_eth))
    ax_main.set_yticks(range(n_eth))
    ax_main.set_xticklabels(eth_labels, rotation=45, ha='right', fontsize=12)
    ax_main.set_yticklabels(eth_labels, fontsize=12)
    
    # Style focal ethnicity labels
    for i, label in enumerate(ax_main.get_yticklabels()):
        if i == focal_idx_ordered:
            label.set_fontweight('bold')
            label.set_color('gold')
            label.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.9))
    
    for i, label in enumerate(ax_main.get_xticklabels()):
        if i == focal_idx_ordered:
            label.set_fontweight('bold')
            label.set_color('gold')
            label.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.9))
    
    ax_main.set_title('Learned Ethnic Interaction Matrix',
                     fontsize=17, fontweight='bold', pad=20)
    
    ax_main.set_xlabel('To (Target Ethnicity) →', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('From (Source Ethnicity) →', fontsize=13, fontweight='bold')
    
    output_1 = f"{output_base}_f1_main.png"
    plt.savefig(output_1, dpi=dpi, bbox_inches='tight', facecolor='white')
    pdf_path = Path(output_1).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel f1'})
    plt.close()
    print(f"  ✓ Saved: {output_1}")
    
    # === COMPONENT 2: TWO COLORBARS ===
    
    print("\n[2/5] Creating dual colorbars...")
    
    fig = plt.figure(figsize=(3, 8), dpi=dpi, facecolor='none')
    
    # Diagonal colorbar (top)
    from matplotlib.cm import ScalarMappable
    
    ax_cbar1 = fig.add_axes([0.25, 0.55, 0.15, 0.35])
    norm_diag = Normalize(vmin=diag_vmin, vmax=diag_vmax)
    sm_diag = ScalarMappable(cmap=diag_cmap, norm=norm_diag)
    cbar1 = plt.colorbar(sm_diag, cax=ax_cbar1)
    cbar1.set_label('Self-Competition\n(diagonal)\n×10⁻²',
                   fontsize=11, fontweight='bold', labelpad=12)
    cbar1.ax.tick_params(labelsize=10)
    
    # Off-diagonal colorbar (bottom)
    ax_cbar2 = fig.add_axes([0.25, 0.10, 0.15, 0.35])
    norm_offdiag = Normalize(vmin=offdiag_vmin, vmax=offdiag_vmax)
    sm_offdiag = ScalarMappable(cmap=offdiag_cmap, norm=norm_offdiag)
    cbar2 = plt.colorbar(sm_offdiag, cax=ax_cbar2)
    cbar2.set_label('Cross-Interaction\n(off-diagonal)\n×10⁻²',
                   fontsize=11, fontweight='bold', labelpad=12)
    cbar2.ax.tick_params(labelsize=10)
    
    # Labels
    ax_cbar2.text(1.6, 0.95, 'Attract', transform=ax_cbar2.transAxes,
                 ha='left', va='top', fontsize=9, style='italic',
                 fontweight='bold', color='#B2182B')
    ax_cbar2.text(1.6, 0.05, 'Repel', transform=ax_cbar2.transAxes,
                 ha='left', va='bottom', fontsize=9, style='italic',
                 fontweight='bold', color='#2166AC')
    
    output_2 = f"{output_base}_f2_colorbar.png"
    plt.savefig(output_2, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_2).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel f2'})
    plt.close()
    print(f"  ✓ Saved: {output_2}")
    
    # === COMPONENT 3: NETWORK ===
    
    print("\n[3/5] Creating network diagram...")
    
    fig = plt.figure(figsize=(5, 4), dpi=dpi, facecolor='none')
    ax_network = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_network.set_xlim(-1.2, 1.2)
    ax_network.set_ylim(-1.2, 1.2)
    ax_network.set_aspect('equal')
    ax_network.axis('off')
    
    # Circular layout
    angles = np.linspace(0, 2*np.pi, n_eth, endpoint=False)
    radius = 1.0
    positions = {i: (radius * np.cos(a), radius * np.sin(a))
                for i, a in enumerate(angles)}
    
    # Draw edges - use off-diagonal max as reference
    offdiag_abs_max = abs(W_offdiag_ordered[W_offdiag_ordered != 0]).max()
    threshold_edge = offdiag_abs_max * 0.15  # 15% threshold
    
    edges_drawn = 0
    for i in range(n_eth):
        for j in range(i+1, n_eth):
            value = W_display_ordered[i, j]
            
            if abs(value) > threshold_edge:
                x_coords = [positions[i][0], positions[j][0]]
                y_coords = [positions[i][1], positions[j][1]]
                
                if value > 0:
                    color = '#D6604D'
                    linestyle = '-'
                else:
                    color = '#4393C3'
                    linestyle = '--'
                
                width = 0.8 + 3.5 * abs(value) / (offdiag_abs_max + 1e-10)
                
                ax_network.plot(x_coords, y_coords, color=color,
                              linewidth=width, alpha=0.75, linestyle=linestyle,
                              zorder=1)
                edges_drawn += 1
    
    print(f"  Drew {edges_drawn} edges (threshold: {threshold_edge:.3f})")
    
    # Draw nodes
    for i in range(n_eth):
        x, y = positions[i]
        
        self_val = abs(W_display_ordered[i, i])
        diag_abs_max = abs(W_diag_ordered).max()
        node_size = 200 + 400 * self_val / diag_abs_max
        
        if i == focal_idx_ordered:
            color = 'gold'
            edgecolor = '#2C3E50'
            linewidth = 3
        else:
            color = '#ECF0F1'
            edgecolor = '#2C3E50'
            linewidth = 1.5
        
        ax_network.scatter([x], [y], s=node_size, c=color,
                         edgecolors=edgecolor, linewidths=linewidth,
                         zorder=10, alpha=0.9)
        
        label_r = radius * 1.18
        label_x = label_r * np.cos(angles[i])
        label_y = label_r * np.sin(angles[i])
        
        ax_network.text(label_x, label_y, eth_labels[i],
                      ha='center', va='center', fontsize=8,
                      fontweight='bold' if i == focal_idx_ordered else 'normal',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.9, edgecolor='none'))
    
    ax_network.text(0, 1.25, 'Interaction Network', ha='center', fontsize=12,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
                   facecolor='white', edgecolor='#2C3E50', linewidth=2))
    
    if edges_drawn > 0:
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#D6604D', linewidth=2, label='Attraction'),
            Line2D([0], [0], color='#4393C3', linewidth=2, linestyle='--', label='Repulsion')
        ]
        ax_network.legend(handles=legend_elements, loc='lower center',
                        bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9,
                        frameon=True, fancybox=True,
                        title='(>15% of max cross-interaction)', title_fontsize=7)
    
    output_3 = f"{output_base}_f3_network.png"
    plt.savefig(output_3, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_3).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel f3'})
    plt.close()
    print(f"  ✓ Saved: {output_3}")
    
    # === COMPONENT 4: STATISTICS ===
    
    print("\n[4/5] Creating statistics...")
    
    fig = plt.figure(figsize=(4, 3), dpi=dpi, facecolor='none')
    ax_stats = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_stats.axis('off')
    
    positive_count = (W_offdiag_ordered > 0).sum()
    negative_count = (W_offdiag_ordered < 0).sum()
    
    stats_text = f"""LEARNED INTERACTIONS

Self-Competition (diagonal):
  Mean: {W_diag_ordered.mean():.2f}
  Range: [{W_diag_ordered.min():.2f}, {W_diag_ordered.max():.2f}]

Cross-Interactions (off-diagonal):
  Mean: {W_offdiag_ordered[W_offdiag_ordered != 0].mean():.3f}
  Range: [{W_offdiag_ordered[W_offdiag_ordered != 0].min():.3f}, 
         {W_offdiag_ordered[W_offdiag_ordered != 0].max():.3f}]

Interaction Types:
  Positive (attraction): {positive_count}
  Negative (repulsion): {negative_count}

Units: ×10⁻²
"""
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 ha='center', va='center', fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                          edgecolor='#2C3E50', linewidth=2, alpha=0.95))
    
    output_4 = f"{output_base}_f4_statistics.png"
    plt.savefig(output_4, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_4).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel f4'})
    plt.close()
    print(f"  ✓ Saved: {output_4}")
    
    # === COMPONENT 5: ANNOTATIONS ===
    
    print("\n[5/5] Creating annotations...")
    
    fig = plt.figure(figsize=(3, 2), dpi=dpi, facecolor='none')
    ax_annot = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_annot.set_xlim(0, 1)
    ax_annot.set_ylim(0, 1)
    ax_annot.axis('off')
    
    ax_annot.text(0.1, 0.7, 'F', fontsize=32, fontweight='bold',
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#2C3E50',
                          edgecolor='white', linewidth=3, alpha=0.95))
    
    guide_text = """TWO-TIER COLORING:
• Diagonal = self (blue scale)
• Off-diagonal = cross (red/blue)
• Negative = competition
• Positive = facilitation"""
    
    ax_annot.text(0.5, 0.3, guide_text, fontsize=9, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#555555', linewidth=1.5, alpha=0.9))
    
    output_5 = f"{output_base}_f5_annotations.png"
    plt.savefig(output_5, dpi=dpi, facecolor='none', edgecolor='none', bbox_inches='tight')
    pdf_path = Path(output_5).with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel f5'})
    plt.close()
    print(f"  ✓ Saved: {output_5}")
    
    print("\n" + "="*80)
    print("SUCCESS - ALL COMPONENTS SAVED")
    print("="*80)
    
    return {
        'W': W,
        'W_display': W_display_ordered,
        'ethnicities': ethnicities_ordered
    }


def main():
    parser = argparse.ArgumentParser(description='Create Panel F - Interaction Matrix')
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
                       default='../figures/figure3/panel_f')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--figsize', type=float, nargs=2, default=[11, 9])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PANEL F: INTERACTION MATRIX (TWO-TIER COLORING)")
    print("="*80)
    
    analyzer = GraphPDEAnalyzer(
        checkpoint_path=args.checkpoint,
        graph_path=args.graph,
        da_info_path=args.da_info,
        device=args.device
    )
    
    analyzer.set_ethnicities(args.ethnicities)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    results = create_panel_f_interaction_matrix_separated(
        analyzer,
        focal_ethnicity=args.ethnicity,
        all_ethnicities=args.ethnicities,
        output_base=output_path,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )
    
    print("\n✓ Panel F complete with two-tier coloring")
    print("  Diagonal and off-diagonal are now both visible!")


if __name__ == "__main__":
    main()