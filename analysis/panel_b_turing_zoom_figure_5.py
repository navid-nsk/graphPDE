"""
Figure 5 - Panel B: Turing Pattern Zoom

Generates Panel B for Figure 5.
Outputs saved to: ./figures/figure5/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def get_geographic_coordinates(analyzer, city_filter='Toronto'):
    """
    Get ACTUAL geographic coordinates for DAUIDs, not normalized.
    
    Returns:
        coords: (n_nodes, 2) array with REAL [x, y] in meters (Web Mercator)
        dauids: Array of DAUID strings
        city_mask: Boolean mask for filtered city
    """
    import geopandas as gpd
    from pyproj import Transformer
    
    print("Loading geographic coordinates from shapefile...")
    
    # Load shapefile
    gdf = gpd.read_file('../data/da_canada2.shp')
    gdf['DAUID'] = gdf['DAUID'].astype(str).str.strip()
    
    # Get DAUIDs from graph
    graph_dauids = analyzer.graph['node_features']['dauid']
    city_names = analyzer.graph['node_features']['city']
    
    if city_filter:
        city_mask = np.array([city == city_filter for city in city_names])
        filtered_dauids = graph_dauids[city_mask]
    else:
        city_mask = np.ones(len(graph_dauids), dtype=bool)
        filtered_dauids = graph_dauids
    
    print(f"  Total DAUIDs in graph: {len(graph_dauids)}")
    print(f"  Filtered to {city_filter}: {len(filtered_dauids)}")
    
    # Filter shapefile to city
    city_gdf = gdf[gdf['DAUID'].isin([str(d) for d in filtered_dauids])].copy()
    
    # Reproject to Web Mercator (meters) for better distance calculations
    city_gdf = city_gdf.to_crs(epsg=3857)
    
    print(f"  Matched {len(city_gdf)} DAUIDs in shapefile")
    
    # Get centroids in projected coordinates (meters)
    centroids = city_gdf.geometry.centroid
    
    # Create coordinate array matching order of filtered_dauids
    coords = np.zeros((len(filtered_dauids), 2))
    dauid_to_idx = {str(d): i for i, d in enumerate(filtered_dauids)}
    
    for idx, row in city_gdf.iterrows():
        dauid_str = row['DAUID']
        if dauid_str in dauid_to_idx:
            graph_idx = dauid_to_idx[dauid_str]
            coords[graph_idx, 0] = centroids.loc[idx].x
            coords[graph_idx, 1] = centroids.loc[idx].y
    
    # Get bounds for context
    bounds = city_gdf.total_bounds
    x_span = bounds[2] - bounds[0]
    y_span = bounds[3] - bounds[1]
    
    print(f"  Geographic extent:")
    print(f"    X: {bounds[0]:.0f} to {bounds[2]:.0f} (span: {x_span:.0f} m = {x_span/1000:.1f} km)")
    print(f"    Y: {bounds[1]:.0f} to {bounds[3]:.0f} (span: {y_span:.0f} m = {y_span/1000:.1f} km)")
    
    return coords, filtered_dauids, city_mask, bounds


def create_panel_b_turing_zoom(analyzer, focal_ethnicity, all_ethnicities, output_path,
                                dpi=300, figsize=(10, 10), simulation_years=50,
                                city_filter='Toronto'):
    """
    Create Turing pattern zoom using REAL geographic coordinates.
    """
    
    print("\n" + "="*80)
    print("PANEL B: TURING PATTERN ZOOM (GEOGRAPHIC COORDINATES)")
    print("="*80)
    
    # Get REAL coordinates from shapefile
    try:
        coords, dauids, city_mask, bounds = get_geographic_coordinates(analyzer, city_filter)
    except Exception as e:
        print(f"\n❌ ERROR: Could not load geographic coordinates: {e}")
        print("Falling back to normalized coordinates (not recommended)")
        
        # Fallback to normalized
        turing_data = analyzer.extract_turing_patterns(
            focal_ethnicity=focal_ethnicity,
            all_ethnicities=all_ethnicities,
            use_model=True,
            simulation_years=simulation_years,
            city_filter=city_filter
        )
        coords = turing_data['coords']
        conc = turing_data['concentration']
        params = turing_data['parameters']
    else:
        # Extract patterns WITH GEOGRAPHIC COORDS
        print("\n" + "="*80)
        print(f"Extracting Turing patterns for {focal_ethnicity}...")
        print(f"  Using REAL geographic coordinates (Web Mercator meters)")
        print("="*80)
        
        # Get focal ethnicity index
        focal_idx = all_ethnicities.index(focal_ethnicity)
        
        # Get parameters
        params = analyzer.get_parameters_for_period(period_idx=3)
        
        # Get initial population data
        temporal_immigration = analyzer.graph['temporal_data']['immigration']
        most_recent_year = sorted(temporal_immigration.keys())[-1]
        df_recent = temporal_immigration[most_recent_year].copy()
        df_recent['DAUID'] = df_recent['DAUID'].astype(str).str.strip()
        
        # Map ethnicity to column
        ethnic_cols_map = {
            'China': 'dim_405', 'Philippines': 'dim_410', 'India': 'dim_407',
            'Pakistan': 'dim_419', 'Iran': 'dim_421', 'Sri Lanka': 'dim_417',
            'Portugal': 'dim_413', 'Italy': 'dim_406', 'United Kingdom': 'dim_404'
        }
        
        # Get initial concentrations
        initial_conc = np.zeros((len(dauids), len(all_ethnicities)))
        
        for i, dauid in enumerate(dauids):
            dauid_str = str(dauid)
            if dauid_str in df_recent['DAUID'].values:
                row = df_recent[df_recent['DAUID'] == dauid_str].iloc[0]
                for j, eth in enumerate(all_ethnicities):
                    if eth in ethnic_cols_map:
                        col = ethnic_cols_map[eth]
                        if col in row:
                            initial_conc[i, j] = float(row[col])
        
        print(f"  Extracting trajectory from model predictions...")

        # Call extract_turing_patterns to get trajectory
        turing_data = analyzer.extract_turing_patterns(
            focal_ethnicity=focal_ethnicity,
            all_ethnicities=all_ethnicities,
            predictions_csv='../model/all_predictions.csv',  # Path to your CSV
            simulation_years=20,
            city_filter=city_filter
        )

        trajectory = turing_data.get('trajectory')

        if trajectory is not None:
            print(f"  ✓ Using model trajectory: {trajectory.shape}")
            # trajectory shape: (5, n_dauids) for years [2001, 2006, 2011, 2016, 2021]
            
            # Use FINAL time point (2021) for visualization
            conc_all_time = trajectory[-1, :]  # Final state
            
            # Match DAUIDs between trajectory and geographic coords
            traj_dauids = turing_data['dauids']
            
            # Create mapping
            traj_dauid_to_idx = {str(d): i for i, d in enumerate(traj_dauids)}
            
            conc = np.zeros(len(dauids))
            for i, dauid in enumerate(dauids):
                dauid_str = str(dauid)
                if dauid_str in traj_dauid_to_idx:
                    traj_idx = traj_dauid_to_idx[dauid_str]
                    conc[i] = conc_all_time[traj_idx]
            
            print(f"  Final {focal_ethnicity} population (model 2021): {conc.sum():.0f}")
            print(f"  Initial population (census 2001): {trajectory[0, :].sum():.0f}")
            print(f"  Growth: {(conc.sum() / trajectory[0, :].sum() - 1)*100:.1f}%")
            
        else:
            print(f"  WARNING: No trajectory found, using initial census data")
            # Fallback to census data
            for i, dauid in enumerate(dauids):
                dauid_str = str(dauid)
                if dauid_str in df_recent['DAUID'].values:
                    row = df_recent[df_recent['DAUID'] == dauid_str].iloc[0]
                    if col in row:
                        initial_conc[i, focal_idx] = float(row[col])
            
            conc = initial_conc[:, focal_idx]
    
    print(f"\nData: {len(coords)} DAUIDs, {conc.sum():.0f} total population")
    
    # === DIAGNOSTIC: Check coordinate distribution ===
    print(f"\nCoordinate distribution (GEOGRAPHIC):")
    print(f"  X range: [{coords[:, 0].min():.0f}, {coords[:, 0].max():.0f}] meters")
    print(f"  Y range: [{coords[:, 1].min():.0f}, {coords[:, 1].max():.0f}] meters")
    x_span_m = coords[:, 0].max() - coords[:, 0].min()
    y_span_m = coords[:, 1].max() - coords[:, 1].min()
    print(f"  X span: {x_span_m:.0f} m = {x_span_m/1000:.1f} km")
    print(f"  Y span: {y_span_m:.0f} m = {y_span_m/1000:.1f} km")
    
    # === FIND OPTIMAL ZOOM REGION (in REAL coordinates) ===
    print("\nFinding optimal zoom region...")
    
    # Multi-threshold approach
    if conc.sum() > 0:
        high_threshold = np.percentile(conc[conc > 0], 85)
    else:
        high_threshold = 0
    
    high_mask = conc > high_threshold
    high_coords = coords[high_mask]
    high_conc = conc[high_mask]
    
    print(f"  High-density points (>{high_threshold:.1f}): {len(high_coords)}")
    
    if len(high_coords) < 50:
        print("  WARNING: Insufficient high-density points, lowering threshold")
        high_threshold = np.percentile(conc[conc > 0], 70) if conc.sum() > 0 else 0
        high_mask = conc > high_threshold
        high_coords = coords[high_mask]
        high_conc = conc[high_mask]
        print(f"  Retry with threshold {high_threshold:.1f}: {len(high_coords)} points")
    
    if len(high_coords) < 20:
        print("  Using center of mass of all data")
        center = np.average(coords, weights=conc + 1e-10, axis=0)
        # Use REAL distance for zoom (5-10 km)
        zoom_size = 7000  # 7 km in meters
    else:
        # Weighted clustering with REAL distances
        # eps in meters (1000m = 1km)
        clustering = DBSCAN(eps=2000, min_samples=5).fit(high_coords)
        labels = set(clustering.labels_)
        labels.discard(-1)
        
        if not labels:
            print("  WARNING: No clusters found, using center of mass")
            center = np.average(high_coords, weights=high_conc, axis=0)
            zoom_size = 7000  # 7 km
        else:
            # Find largest cluster
            cluster_pops = []
            for label in labels:
                cluster_mask = clustering.labels_ == label
                cluster_pop = high_conc[cluster_mask].sum()
                cluster_pops.append((label, cluster_pop, cluster_mask.sum()))
            
            largest_cluster = max(cluster_pops, key=lambda x: x[1])[0]
            cluster_mask = clustering.labels_ == largest_cluster
            
            print(f"  Found {len(labels)} clusters, using largest with {cluster_pops[0][2]} points")
            
            # Weighted center
            cluster_coords = high_coords[cluster_mask]
            cluster_weights = high_conc[cluster_mask]
            center = np.average(cluster_coords, weights=cluster_weights, axis=0)
            
            # Adaptive zoom: 3-10 km based on cluster size
            cluster_std = np.std(cluster_coords, axis=0).mean()
            zoom_size = np.clip(cluster_std * 3, 3000, 10000)  # 3-10 km in meters
        
        print(f"  Cluster center: ({center[0]:.0f}, {center[1]:.0f}) meters")
        print(f"  Zoom size: {zoom_size:.0f} m = {zoom_size/1000:.1f} km")
    
    # Define zoom window (in REAL coordinates)
    x_min = center[0] - zoom_size/2
    x_max = center[0] + zoom_size/2
    y_min = center[1] - zoom_size/2
    y_max = center[1] + zoom_size/2
    
    # Filter to zoom region
    zoom_mask = ((coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                 (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max))
    
    zoom_coords = coords[zoom_mask]
    zoom_conc = conc[zoom_mask]
    
    pct_captured = zoom_mask.sum() / len(coords) * 100
    print(f"  Zoom region: {zoom_mask.sum()} DAUIDs ({pct_captured:.1f}% of total)")
    print(f"  Population in zoom: {zoom_conc.sum():.0f} ({zoom_conc.sum()/conc.sum()*100:.1f}% of total)")
    
    if len(zoom_coords) < 20:
        print("  ERROR: Zoom region has too few points! Using all data...")
        zoom_coords = coords
        zoom_conc = conc
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        zoom_size = max(x_max - x_min, y_max - y_min)
    
    # === HIGH-RESOLUTION INTERPOLATION ===
    print("\nGenerating high-resolution Turing pattern...")
    
    # Adaptive grid resolution
    grid_res = np.clip(int(400 * np.sqrt(10000 / zoom_size)), 300, 800)
    print(f"  Grid resolution: {grid_res}x{grid_res}")
    
    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Log transform
    zoom_conc_log = np.log1p(zoom_conc)
    conc_mean = zoom_conc_log.mean()
    conc_std = zoom_conc_log.std() + 1e-10
    conc_normalized = (zoom_conc_log - conc_mean) / conc_std
    
    # RBF interpolation
    print("  RBF interpolation...")
    try:
        rbf = Rbf(zoom_coords[:, 0], zoom_coords[:, 1], conc_normalized,
                 function='thin_plate', smooth=0.01)
        Zi = rbf(Xi, Yi)
    except Exception as e:
        print(f"  RBF failed ({e}), using cubic griddata")
        from scipy.interpolate import griddata
        Zi = griddata(zoom_coords, conc_normalized, (Xi, Yi),
                     method='cubic', fill_value=0)
    
    # Adaptive smoothing
    focal_idx = all_ethnicities.index(focal_ethnicity)
    D = params['diffusion_coefficients'].mean(axis=0)
    D_focal = np.abs(D[focal_idx])
    
    # Scale smoothing to geographic distances
    sigma_pixels = np.clip(np.sqrt(D_focal) * 3, 0.5, 2.0)
    print(f"  Diffusion smoothing: σ = {sigma_pixels:.2f} pixels")
    
    Zi_smooth = gaussian_filter(Zi, sigma=sigma_pixels)
    
    # Edge enhancement
    from scipy.ndimage import laplace
    edges = laplace(Zi_smooth)
    Zi_enhanced = Zi_smooth - 0.12 * edges
    
    # Normalize to [0, 1]
    Zi_final = (Zi_enhanced - np.min(Zi_enhanced)) / (np.max(Zi_enhanced) - np.min(Zi_enhanced) + 1e-10)
    
    # === CREATE FIGURE ===
    print("\nCreating figure...")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
    
    # Custom colormap
    TURING_CMAP = LinearSegmentedColormap.from_list('turing_enhanced',
        [(0.0, '#0a0a1e'), (0.15, '#1a1a3e'), (0.3, '#0f3460'), 
         (0.5, '#16537e'), (0.7, '#3a7ca5'), (0.85, '#e94560'), (1.0, '#fef5e7')],
        N=256)
    
    # Main image
    im = ax.imshow(Zi_final, cmap=TURING_CMAP, origin='lower',
                   extent=[x_min, x_max, y_min, y_max],
                   aspect='equal', interpolation='bilinear',
                   vmin=0, vmax=1)
    
    # Contours
    if np.std(Zi_final) > 0.05:
        levels_dark = np.linspace(0.15, 0.45, 6)
        ax.contour(Xi, Yi, Zi_final, levels=levels_dark,
                  colors='#1a1a3e', alpha=0.4, linewidths=0.6)
        
        levels_light = np.linspace(0.55, 0.9, 8)
        ax.contour(Xi, Yi, Zi_final, levels=levels_light,
                  colors='white', alpha=0.3, linewidths=0.8)
        
        peak_levels = [0.85, 0.90, 0.95]
        ax.contour(Xi, Yi, Zi_final, levels=peak_levels,
                  colors='#ffd700', alpha=0.6, linewidths=[0.8, 1.0, 1.3])
    
    # Panel label
    # ax.text(0.03, 0.97, 'B', transform=ax.transAxes,
    #        fontsize=28, fontweight='bold', va='top', color='white',
    #        bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50',
    #                 edgecolor='white', linewidth=2, alpha=0.95))
    
    # Title
    ax.set_title(f'Emergent Spatial Patterns: {focal_ethnicity} Settlement',
                fontsize=16, fontweight='bold', pad=15, color='#2C3E50')
    
    # Scale bar (REAL distance)
    scale_km = 2
    scale_m = scale_km * 1000
    scale_x_start = x_max - zoom_size * 0.25
    scale_x_end = scale_x_start + scale_m
    scale_y = y_min + zoom_size * 0.05
    
    ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
           'w-', linewidth=4, solid_capstyle='butt', zorder=10)
    ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
           'k-', linewidth=2, solid_capstyle='butt', zorder=11)
    ax.text((scale_x_start + scale_x_end)/2, scale_y + zoom_size * 0.02,
           f'{scale_km} km', ha='center', va='bottom',
           fontsize=12, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
           zorder=12)
    
    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Relative Concentration', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', '', 'Medium', '', 'High'])
    
    # Model info
    info_text = (f'Geographic Coordinates\n'
                 f'Zoom: {zoom_size/1000:.1f} km\n'
                 f'Diffusion: D={D_focal:.4f}')
    ax.text(0.97, 0.03, info_text,
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=9, style='italic', color='white',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    # Pattern type
    pattern_type = params.get('pattern_type', 'Spots')
    ax.text(0.97, 0.97, pattern_type,
           transform=ax.transAxes, ha='right', va='top',
           fontsize=11, fontweight='bold', color='#ffd700',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#2C3E50',
                    edgecolor='#ffd700', linewidth=1.5, alpha=0.9))
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=dpi, pad_inches=0.1,
               facecolor='white', edgecolor='none')
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel b'})
    
    plt.close()
    
    print(f"\n✓ Panel B saved: {output_path}")
    print(f"  Geographic zoom: {zoom_size/1000:.1f} km")
    
    return {
        'zoom_size_km': zoom_size/1000,
        'n_dauids': zoom_mask.sum(),
        'population': zoom_conc.sum()
    }


def main():
    parser = argparse.ArgumentParser(description='Create Panel B - Geographic Turing Zoom')
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
                       default='../figures/figure5/panel_b.png')
    parser.add_argument('--city', type=str, default='Toronto',
                       help='City to visualize (or None for all cities)')
    parser.add_argument('--simulation-years', type=int, default=50)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 10])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PANEL B: TURING PATTERNS WITH GEOGRAPHIC COORDINATES")
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
    
    city_filter = None if args.city.lower() == 'none' else args.city
    
    results = create_panel_b_turing_zoom(
        analyzer,
        focal_ethnicity=args.ethnicity,
        all_ethnicities=args.ethnicities,
        output_path=output_path,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        simulation_years=args.simulation_years,
        city_filter=city_filter
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)


if __name__ == "__main__":
    main()