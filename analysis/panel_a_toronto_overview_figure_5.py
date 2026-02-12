"""
Figure 5 - Panel A: Toronto Overview Map

Generates Panel A for Figure 5.
Outputs saved to: ./figures/figure5/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.cluster import DBSCAN
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def create_panel_a_toronto_overview(analyzer, focal_ethnicity, all_ethnicities,
                                    output_path, dpi=600, figsize=(12, 12)):
    """
    Create publication-quality Toronto overview with shapefile boundaries.
    
    Key improvements:
    1. Uses actual DAUID shapefiles from da_canada2.shp
    2. High-resolution rendering (600 DPI)
    3. Elegant colormap with perceptual uniformity
    4. Zoom box indicator for Panel B - NOW MATCHES PANEL B EXACTLY
    5. Professional cartographic styling
    6. Scale bar and north arrow
    """
    
    print("\n" + "="*80)
    print("PANEL A: TORONTO OVERVIEW MAP")
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
    
    # CRITICAL FIX: Use final simulation state, not initial census data
    trajectory = turing_data.get('trajectory')
    focal_idx = all_ethnicities.index(focal_ethnicity)
    
    if trajectory is not None:
        print(f"  Using final state from 20-year simulation (trajectory shape: {trajectory.shape})")
        # Extract final timestep for focal ethnicity
        if len(trajectory.shape) == 3:
            # Shape: (timesteps, nodes, ethnicities)
            conc = trajectory[-1, :, focal_idx]
        elif len(trajectory.shape) == 2:
            # Shape: (timesteps, nodes) - single ethnicity
            conc = trajectory[-1, :]
        else:
            print(f"  WARNING: Unexpected trajectory shape {trajectory.shape}")
            conc = turing_data['concentration']
    else:
        print(f"  WARNING: No trajectory found, using initial census data")
        conc = turing_data['concentration']
    
    print(f"\nData: {len(coords)} Toronto DAUIDs")
    print(f"Total {focal_ethnicity} population (20-year endpoint): {conc.sum():.0f}")
    print(f"Initial population (census): {turing_data['concentration'].sum():.0f}")
    print(f"Growth over simulation: {(conc.sum() / turing_data['concentration'].sum() - 1)*100:.1f}%")
    
    # === LOAD SHAPEFILE ===
    print("\nLoading shapefile...")
    
    try:
        import geopandas as gpd
        
        # Load Canada shapefile
        gdf = gpd.read_file('../data/da_canada2.shp')
        gdf['DAUID'] = gdf['DAUID'].astype(str).str.strip()
        print(f"  Loaded {len(gdf):,} DAUIDs from shapefile")
        
        # Get Toronto DAUIDs
        graph_dauids = analyzer.graph['node_features']['dauid']
        city_names = analyzer.graph['node_features']['city']
        toronto_mask = np.array([city == 'Toronto' for city in city_names])
        toronto_dauids = graph_dauids[toronto_mask]
        
        # Filter shapefile to Toronto
        toronto_gdf = gdf[gdf['DAUID'].isin([str(d) for d in toronto_dauids])].copy()
        print(f"  Filtered to {len(toronto_gdf)} Toronto DAUIDs")
        
        # Reproject to suitable CRS for visualization
        toronto_gdf = toronto_gdf.to_crs(epsg=3857)  # Web Mercator
        
        # Merge with concentration data
        dauid_to_conc = {str(d): c for d, c in zip(toronto_dauids, conc)}
        toronto_gdf['concentration'] = toronto_gdf['DAUID'].map(dauid_to_conc).fillna(0)
        toronto_gdf['log_conc'] = np.log1p(toronto_gdf['concentration'])
        
        print(f"  Concentration range: {toronto_gdf['concentration'].min():.0f} - "
              f"{toronto_gdf['concentration'].max():.0f}")
        
        use_shapefile = True
        
    except Exception as e:
        print(f"  ERROR loading shapefile: {e}")
        print(f"  Falling back to scatter plot")
        use_shapefile = False
    
    # === CREATE FIGURE ===
    print("\nCreating figure...")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
    
    # Enhanced colormap (perceptually uniform)
    SETTLEMENT_CMAP = LinearSegmentedColormap.from_list('settlement',
        [(0.0, '#f7f7f7'), (0.15, '#fee5d9'), (0.3, '#fcbba1'),
         (0.5, '#fc9272'), (0.7, '#fb6a4a'), (0.85, '#de2d26'),
         (1.0, '#a50f15')],
        N=256)
    
    if use_shapefile:
        # Plot shapefile with elegant styling
        vmin = 0
        vmax = np.percentile(toronto_gdf['log_conc'], 98)  # Clip extreme outliers
        
        toronto_gdf.plot(
            column='log_conc',
            ax=ax,
            cmap=SETTLEMENT_CMAP,
            edgecolor='#555555',
            linewidth=0.12,
            alpha=0.92,
            vmin=vmin,
            vmax=vmax,
            legend=False
        )
        
        # Elegant colorbar
        sm = ScalarMappable(cmap=SETTLEMENT_CMAP, norm=Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04, shrink=0.75)
        cbar.set_label('Settlement Intensity\nlog(Population + 1)', 
                      fontsize=13, labelpad=12)
        cbar.ax.tick_params(labelsize=11)
        
        # Set aspect and limits from shapefile bounds
        ax.set_aspect('equal')
        bounds = toronto_gdf.total_bounds
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        margin = 0.05
        ax.set_xlim(bounds[0] - margin*x_range, bounds[2] + margin*x_range)
        ax.set_ylim(bounds[1] - margin*y_range, bounds[3] + margin*y_range)
        
    else:
        # Fallback: scatter plot
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=np.log1p(conc),
            s=8,
            cmap=SETTLEMENT_CMAP,
            alpha=0.85,
            edgecolors='black',
            linewidths=0.1,
            rasterized=True
        )
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.04, shrink=0.75)
        cbar.set_label('Settlement Intensity\nlog(Population + 1)',
                      fontsize=13, labelpad=12)
        cbar.ax.tick_params(labelsize=11)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
    
    # === FIND ZOOM REGION FOR PANEL B (GEOGRAPHIC COORDINATES) ===
    print("\nIdentifying zoom region (matching Panel B logic)...")
    
    # Get geographic coordinates if using shapefile
    if use_shapefile:
        # Map DAUIDs to geographic coords
        dauid_to_geo = {}
        for idx, row in toronto_gdf.iterrows():
            dauid_str = row['DAUID']
            centroid = row.geometry.centroid
            dauid_to_geo[dauid_str] = (centroid.x, centroid.y)
        
        # Create geographic coordinate array matching conc order
        geo_coords = np.zeros_like(coords)
        for i, dauid in enumerate(toronto_dauids):
            dauid_str = str(dauid)
            if dauid_str in dauid_to_geo:
                geo_coords[i] = dauid_to_geo[dauid_str]
    else:
        geo_coords = coords
    
    # Find high-density clusters (SAME LOGIC AS PANEL B)
    high_threshold = np.percentile(conc[conc > 0], 85) if conc.sum() > 0 else 0
    high_mask = conc > high_threshold
    high_coords = geo_coords[high_mask]
    high_conc = conc[high_mask]
    
    print(f"  High-density points (>{high_threshold:.1f}): {len(high_coords)}")
    
    # Retry with lower threshold if needed (SAME AS PANEL B)
    if len(high_coords) < 50:
        print("  WARNING: Insufficient high-density points, lowering threshold")
        high_threshold = np.percentile(conc[conc > 0], 70) if conc.sum() > 0 else 0
        high_mask = conc > high_threshold
        high_coords = geo_coords[high_mask]
        high_conc = conc[high_mask]
        print(f"  Retry with threshold {high_threshold:.1f}: {len(high_coords)} points")
    
    zoom_region_geo = None
    
    if len(high_coords) < 20:
        print("  Using center of mass of all data")
        center_geo = np.average(geo_coords, weights=conc + 1e-10, axis=0)
        zoom_size_geo = 7000  # 7 km in meters (SAME AS PANEL B)
        
        zoom_region_geo = {
            'center': center_geo,
            'size': zoom_size_geo
        }
    elif len(high_coords) >= 20:
        # DBSCAN clustering (SAME AS PANEL B: eps=2000m, min_samples=5)
        clustering = DBSCAN(eps=2000, min_samples=5).fit(high_coords)
        labels = set(clustering.labels_)
        labels.discard(-1)
        
        if not labels:
            print("  WARNING: No clusters found, using center of mass")
            center_geo = np.average(high_coords, weights=high_conc, axis=0)
            zoom_size_geo = 7000
            
            zoom_region_geo = {
                'center': center_geo,
                'size': zoom_size_geo
            }
        else:
            # Find largest cluster by population (SAME AS PANEL B)
            cluster_pops = []
            for label in labels:
                cluster_mask = clustering.labels_ == label
                cluster_pop = high_conc[cluster_mask].sum()
                cluster_pops.append((label, cluster_pop, cluster_mask.sum()))
            
            largest_cluster = max(cluster_pops, key=lambda x: x[1])[0]
            cluster_mask = clustering.labels_ == largest_cluster
            
            print(f"  Found {len(labels)} clusters, using largest with {cluster_pops[0][2]} points")
            
            # Weighted center (SAME AS PANEL B)
            cluster_coords = high_coords[cluster_mask]
            cluster_weights = high_conc[cluster_mask]
            center_geo = np.average(cluster_coords, weights=cluster_weights, axis=0)
            
            # Adaptive zoom: 3-10 km based on cluster size (SAME AS PANEL B)
            cluster_std = np.std(cluster_coords, axis=0).mean()
            zoom_size_geo = np.clip(cluster_std * 3, 3000, 10000)  # 3-10 km in meters
            
            zoom_region_geo = {
                'center': center_geo,
                'size': zoom_size_geo
            }
            
            print(f"  Cluster center: ({center_geo[0]:.0f}, {center_geo[1]:.0f}) meters")
            print(f"  Zoom size: {zoom_size_geo:.0f} m = {zoom_size_geo/1000:.1f} km")
    
    # Draw zoom box if found
    if zoom_region_geo:
        center_geo = zoom_region_geo['center']
        size_geo = zoom_region_geo['size']
        
        # Draw rectangle in geographic coordinates
        rect = Rectangle(
            (center_geo[0] - size_geo/2, center_geo[1] - size_geo/2),
            size_geo, size_geo,
            fill=False, edgecolor='#E74C3C', linewidth=3.5, zorder=10,
            linestyle='-'
        )
        
        ax.add_patch(rect)
        
        # Label pointing to zoom region
        label_x = center_geo[0]
        label_y = center_geo[1] + size_geo/2 + 0.02 * y_range if use_shapefile else center_geo[1] + size_geo/2 + 0.03
        
        ax.text(label_x, label_y, 'See Panel B',
               fontsize=14, fontweight='bold', ha='center',
               color='white',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E74C3C',
                        edgecolor='white', linewidth=2.5, alpha=0.95),
               zorder=11)
    
    # === ANNOTATIONS ===
    
    # Panel label
    # ax.text(0.02, 0.98, 'A', transform=ax.transAxes,
    #        fontsize=32, fontweight='bold', va='top', color='white',
    #        bbox=dict(boxstyle='round,pad=0.6', facecolor='#2C3E50',
    #                 edgecolor='white', linewidth=2.5, alpha=0.95),
    #        zorder=10)
    
    # Title
    ax.set_title(f'{focal_ethnicity} Settlement Distribution in Toronto\n'
                 f'GraphPDE Model Prediction (20-Year Simulation)',
                fontsize=15, fontweight='bold', pad=18, color='#2C3E50')
    
    # Scale bar (professional cartographic style)
    if use_shapefile:
        # 20 km scale bar
        scale_length_km = 20
        scale_length_m = scale_length_km * 1000
        
        # Position in lower right
        scale_x_start = bounds[2] - scale_length_m - 0.1 * x_range
        scale_x_end = bounds[2] - 0.1 * x_range
        scale_y = bounds[1] + 0.05 * y_range
        
        # Draw scale bar
        ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
               'k-', linewidth=5, solid_capstyle='butt', zorder=9)
        ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
               'w-', linewidth=3, solid_capstyle='butt', zorder=10)
        
        # Tick marks
        for x in [scale_x_start, scale_x_end]:
            ax.plot([x, x], [scale_y, scale_y - 0.01*y_range],
                   'k-', linewidth=3, zorder=10)
        
        # Label
        ax.text((scale_x_start + scale_x_end)/2, scale_y - 0.025*y_range,
               f'{scale_length_km} km', ha='center', va='top',
               fontsize=13, fontweight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='black', linewidth=1.5, alpha=0.9),
               zorder=10)
    else:
        # Normalized coordinates scale
        ax.plot([0.75, 0.95], [0.05, 0.05], 'k-', linewidth=5, zorder=9)
        ax.plot([0.75, 0.95], [0.05, 0.05], 'w-', linewidth=3, zorder=10)
        ax.text(0.85, 0.02, '~20 km', ha='center', fontsize=13,
               fontweight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='black', linewidth=1.5, alpha=0.9),
               zorder=10)
    
    # North arrow (elegant)
    if use_shapefile:
        arrow_x = bounds[0] + 0.08 * x_range
        arrow_y = bounds[3] - 0.08 * y_range
        arrow_length = 0.04 * y_range
    else:
        arrow_x, arrow_y = 0.08, 0.92
        arrow_length = 0.04
    
    ax.annotate('N', xy=(arrow_x, arrow_y + arrow_length),
               xytext=(arrow_x, arrow_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
               fontsize=16, fontweight='bold', ha='center', va='bottom',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                        edgecolor='black', linewidth=2, alpha=0.95),
               zorder=10)
    
    # Statistics box
    stats_text = (f'Total Population: {conc.sum():.0f}\n'
                  f'DAUIDs: {len(coords)}\n'
                  f'Max Concentration: {conc.max():.0f}')
    ax.text(0.98, 0.02, stats_text,
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=10, family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='#2C3E50', linewidth=1.5, alpha=0.92),
           zorder=10)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#999999')
        spine.set_linewidth(1.5)
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none',
               metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel A - Toronto Overview {focal_ethnicity}'})
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel A - Toronto Overview {focal_ethnicity}'})

    plt.close()
    
    print(f"\n✓ Panel A saved: {output_path}")
    print(f"  Resolution: {figsize[0]*dpi} x {figsize[1]*dpi} pixels")
    print(f"  Used shapefile: {use_shapefile}")
    
    return {
        'used_shapefile': use_shapefile,
        'n_dauids': len(coords),
        'population': conc.sum(),
        'zoom_region': zoom_region_geo
    }


def main():
    parser = argparse.ArgumentParser(description='Create Panel A - Toronto Overview')
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
                       default='../figures/figure5/panel_a.png')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 12])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PANEL A: TORONTO OVERVIEW (HIGH-QUALITY STANDALONE)")
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
    
    results = create_panel_a_toronto_overview(
        analyzer,
        focal_ethnicity=args.ethnicity,
        all_ethnicities=args.ethnicities,
        output_path=output_path,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\nPanel A ready for Photoshop")
    print(f"File: {output_path}")


if __name__ == "__main__":
    main()