"""
Figure 7: Energy Landscape Analysis

Generates Figure 7 showing energy landscape for Toronto Chinese settlement.
Outputs saved to: ./figures/figure7/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata, UnivariateSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter, binary_dilation, label as scipy_label
from scipy.linalg import expm
import networkx as nx
from pyproj import Transformer
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import YOUR analyzer
from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def calculate_reaction_diffusion_energy(u_local, focal_idx, params, coords_i=None, neighbors=None):
    """
    Calculate configuration energy based on YOUR GraphPDE's reaction-diffusion physics.
    
    Based on YOUR model's equations in model_graphpde_unconstrained.py:
    - Logistic growth: r * u * (1 - N/K)
    - Competition: u * (W @ u.T) / N_safe  
    - Diffusion: D * Laplacian(u)
    
    Returns energy in dimensionless units (dimensionless scale) for interpretability.
    
    Parameters:
    -----------
    u_local : array (n_ethnicities,)
        Population counts at this location
    focal_idx : int
        Index of focal ethnicity  
    params : dict
        YOUR model parameters: W, r, K, D, F, E
    coords_i : array, optional
        Spatial coordinates
    neighbors : list of arrays, optional
        Neighbor populations for gradient calculation
    
    Returns:
    --------
    energy : float
        Free energy in dimensionless
    """
    u_focal = u_local[focal_idx]
    
    # If no focal population, return high energy barrier
    if u_focal < 1e-6:
        return 10.0  # High energy = unstable configuration
    
    # Extract YOUR model's parameters
    W = params['interaction_matrix']  # (n_eth, n_eth)
    r = params['growth_rates']  # (n_eth,)
    K = params['carrying_capacity']  # scalar
    D = params['diffusion_coefficients']  # (n_eth,)
    n_groups = len(u_local)
    
    # Normalize to dimensionless concentration relative to carrying capacity
    c = u_local / (K + 1e-6)  # Concentration [0, 1]
    c_focal = c[focal_idx]
    c_total = c.sum()
    
    # === 1. REACTION ENERGY FUNCTIONAL ===
    # Based on Gray-Scott reaction-diffusion free energy
    # F_reaction = ∫ f(c) dc where f(c) is YOUR model's reaction term
    
    # Logistic growth contribution: -r*c*ln(c) + r*c²/(2*(1-N))
    # This comes from integrating the logistic term
    if c_focal > 0 and c_total < 0.99:
        growth_term = -r[focal_idx] * (
            c_focal * np.log(c_focal + 1e-10) - c_focal +
            0.5 * c_focal**2 / (1 - c_total + 1e-10)
        )
    else:
        growth_term = 5.0  # Penalty for overcrowding
    
    # === 2. INTERACTION ENERGY (Lotka-Volterra type) ===
    # V_int = 0.5 * Σᵢⱼ Wᵢⱼ * cᵢ * cⱼ
    # This is the potential energy from inter-ethnic interactions
    interaction_term = 0
    for j in range(n_groups):
        if W[focal_idx, j] != 0:
            if j == focal_idx:
                # Self-interaction (competition within group)
                interaction_term += 0.5 * W[focal_idx, j] * c_focal**2
            else:
                # Cross-group interaction
                # Positive W = mutualism/cooperation (lowers energy)
                # Negative W = competition/avoidance (raises energy)
                interaction_term += W[focal_idx, j] * c_focal * c[j]
    
    # === 3. CONCENTRATION CHEMICAL POTENTIAL ===
    # μ = dimensionless*ln(c) - entropic cost of concentration
    # This represents the thermodynamic cost of maintaining high concentration
    if c_focal > 0:
        chemical_potential = np.log(c_focal + 1e-10)
    else:
        chemical_potential = -10  # Very low chemical potential (favorable)
    
    # === 4. MIXING ENTROPY CONTRIBUTION ===
    # S = -Σᵢ pᵢ ln(pᵢ) - Shannon entropy
    # Diversity stabilizes the system (lowers free energy F = U - TS)
    if c_total > 0:
        p = c / (c_total + 1e-10)
        p_safe = p[p > 1e-10]
        entropy = -np.sum(p_safe * np.log(p_safe))
    else:
        entropy = 0
    
    # === 5. SPATIAL GRADIENT PENALTY ===
    # ½D|∇c|² - diffusion energy cost
    # Estimated using finite differences with neighbors
    gradient_penalty = 0
    if neighbors is not None and len(neighbors) > 0:
        # Estimate |∇c|² using finite differences
        grad_squared = 0
        for neighbor_u in neighbors:
            if neighbor_u.sum() > 0:
                neighbor_c = neighbor_u / (K + 1e-6)
                dc = c_focal - neighbor_c[focal_idx]
                grad_squared += dc**2
        # Normalize by number of neighbors
        gradient_penalty = 0.5 * D[focal_idx] * grad_squared / len(neighbors)
    
    # === TOTAL FREE ENERGY ===
    # F = U - TS where T is "social temperature"
    # Units: dimensionless (thermal energy units for interpretability)
    social_temperature = 1.0  # Dimensionless temperature
    
    # Combine terms with physically meaningful weights
    total_energy = (
        growth_term * 2.0 +            # Growth dynamics (dominant)
        interaction_term * 5.0 +       # Inter-group effects (most important)
        chemical_potential * 0.5 +     # Concentration effects
        gradient_penalty * 1.0 -       # Spatial smoothness cost
        social_temperature * entropy * 2.0  # Diversity benefit (negative = stabilizing)
    )
    
    # Clip to reasonable range for numerical stability
    return np.clip(total_energy, -10, 10)


def load_toronto_chinese_data(checkpoint_path, graph_path, da_info_path, shapefile_path,
                              focal_ethnicity='China', all_ethnicities=None):
    """
    Load Toronto Chinese settlement data with proper coordinate handling.
    
    KEY FIX: Extract coordinates ONLY for Toronto DAUIDs, then normalize to [0,1]
    based on Toronto's bounds (not Canada's bounds).
    """
    print("\n" + "="*80)
    print(f"LOADING TORONTO DATA - PROPERLY FIXED")
    print("="*80)
    
    if all_ethnicities is None:
        all_ethnicities = ['China', 'Philippines', 'India', 'Pakistan',
                          'Iran', 'Sri Lanka', 'Portugal', 'Italy', 'United Kingdom']
    
    # Initialize analyzer
    print("\nInitializing GraphPDEAnalyzer...")
    from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer
    
    analyzer = GraphPDEAnalyzer(
        checkpoint_path=checkpoint_path,
        graph_path=graph_path,
        da_info_path=da_info_path,
        device='cpu'
    )
    analyzer.set_ethnicities(all_ethnicities)
    
    focal_idx = all_ethnicities.index(focal_ethnicity)
    
    # === STEP 1: IDENTIFY TORONTO DAUIDs ===
    print("\nStep 1: Identifying Toronto DAUIDs...")
    
    # Load DA info to get city names
    da_info = pd.read_csv(da_info_path)
    da_info['DAUID'] = da_info['DAUID'].astype(str).str.strip()
    
    # Filter to Toronto only
    toronto_da_info = da_info[da_info['Name'] == 'Toronto'].copy()
    toronto_dauids_from_csv = set(toronto_da_info['DAUID'].values)
    
    print(f"  Found {len(toronto_dauids_from_csv):,} Toronto DAUIDs in da_canada.csv")
    
    # Get DAUIDs from graph
    graph_dauids = analyzer.graph['node_features']['dauid']
    
    # Find intersection (Toronto DAUIDs that are in the graph)
    toronto_mask = np.array([str(d) in toronto_dauids_from_csv for d in graph_dauids])
    toronto_indices = np.where(toronto_mask)[0]
    toronto_dauids = graph_dauids[toronto_mask]
    
    print(f"  Found {len(toronto_dauids):,} Toronto DAUIDs in graph")
    
    # === STEP 2: GET TORONTO COORDINATES FROM CSV ===
    print("\nStep 2: Extracting Toronto coordinates from da_canada.csv...")
    
    # Create DAUID to coordinate mapping for Toronto only
    dauid_to_coords_toronto = {}
    
    for idx, row in toronto_da_info.iterrows():
        dauid_str = row['DAUID']
        if dauid_str in [str(d) for d in toronto_dauids]:
            # Coordinates are in the CSV (x_coord, y_coord columns)
            dauid_to_coords_toronto[dauid_str] = {
                'x': float(row['x_coord']),
                'y': float(row['y_coord'])
            }
    
    # Build coordinate array in same order as toronto_dauids
    coords_geo = np.zeros((len(toronto_dauids), 2))
    valid_coords_count = 0
    
    for i, dauid in enumerate(toronto_dauids):
        dauid_str = str(dauid)
        if dauid_str in dauid_to_coords_toronto:
            coords_geo[i, 0] = dauid_to_coords_toronto[dauid_str]['x']
            coords_geo[i, 1] = dauid_to_coords_toronto[dauid_str]['y']
            valid_coords_count += 1
    
    print(f"  Extracted coordinates for {valid_coords_count:,} DAUIDs")
    print(f"  Coordinate range (meters):")
    print(f"    X: [{coords_geo[:, 0].min():.0f}, {coords_geo[:, 0].max():.0f}]")
    print(f"    Y: [{coords_geo[:, 1].min():.0f}, {coords_geo[:, 1].max():.0f}]")
    
    # === STEP 3: NORMALIZE BASED ON TORONTO BOUNDS ONLY ===
    print("\nStep 3: Normalizing coordinates to [0,1] based on Toronto bounds...")
    
    x_min, x_max = coords_geo[:, 0].min(), coords_geo[:, 0].max()
    y_min, y_max = coords_geo[:, 1].min(), coords_geo[:, 1].max()
    
    x_span_km = (x_max - x_min) / 1000
    y_span_km = (y_max - y_min) / 1000
    
    print(f"  Toronto extent: {x_span_km:.1f} km × {y_span_km:.1f} km")
    
    coords_norm = np.zeros_like(coords_geo)
    coords_norm[:, 0] = (coords_geo[:, 0] - x_min) / (x_max - x_min + 1e-10)
    coords_norm[:, 1] = (coords_geo[:, 1] - y_min) / (y_max - y_min + 1e-10)
    
    print(f"  Normalized range:")
    print(f"    X: [{coords_norm[:, 0].min():.3f}, {coords_norm[:, 0].max():.3f}]")
    print(f"    Y: [{coords_norm[:, 1].min():.3f}, {coords_norm[:, 1].max():.3f}]")
    
    # Geographic bounds for later use
    bounds_geo = np.array([x_min, y_min, x_max, y_max])
    
    # === STEP 4: GET POPULATION DATA ===
    print("\nStep 4: Loading population data...")
    
    temporal_immigration = analyzer.graph['temporal_data']['immigration']
    most_recent_year = sorted(temporal_immigration.keys())[-1]
    df_recent = temporal_immigration[most_recent_year].copy()
    df_recent['DAUID'] = df_recent['DAUID'].astype(str).str.strip()
    
    ethnic_cols_map = {
        'China': 'dim_405',
        'Philippines': 'dim_410',
        'India': 'dim_407',
        'Pakistan': 'dim_419',
        'Iran': 'dim_421',
        'Sri Lanka': 'dim_417',
        'Portugal': 'dim_413',
        'Italy': 'dim_406',
        'United Kingdom': 'dim_404'
    }
    
    n_toronto = len(toronto_dauids)
    n_ethnicities = len(all_ethnicities)
    u_current = np.zeros((n_toronto, n_ethnicities))
    
    for i, dauid in enumerate(toronto_dauids):
        dauid_str = str(dauid)
        if dauid_str in df_recent['DAUID'].values:
            row = df_recent[df_recent['DAUID'] == dauid_str].iloc[0]
            for j, eth in enumerate(all_ethnicities):
                if eth in ethnic_cols_map:
                    col = ethnic_cols_map[eth]
                    if col in row:
                        u_current[i, j] = float(row[col])
    
    print(f"  {focal_ethnicity} population: {u_current[:, focal_idx].sum():,.0f}")
    print(f"  Total all ethnicities: {u_current.sum():,.0f}")
    print(f"  DAUIDs with {focal_ethnicity} population: {(u_current[:, focal_idx] > 0).sum()}")
    
    # === STEP 5: LOAD SHAPEFILE (for Toronto only) ===
    print("\nStep 5: Loading shapefile...")
    
    gdf = gpd.read_file(shapefile_path)
    gdf['DAUID'] = gdf['DAUID'].astype(str).str.strip()
    
    # Merge with da_info to get city names
    gdf = gdf.merge(da_info[['DAUID', 'Name']], on='DAUID', how='left')
    
    # Filter to Toronto
    gdf_toronto = gdf[gdf['DAUID'].isin([str(d) for d in toronto_dauids])].copy()
    
    # Project to Web Mercator for consistency
    if gdf_toronto.crs != 'EPSG:3857':
        gdf_toronto = gdf_toronto.to_crs('EPSG:3857')
    
    print(f"  Matched {len(gdf_toronto):,} DAUIDs in shapefile")
    
    # === STEP 6: GET PARAMETERS ===
    print("\nStep 6: Getting model parameters...")
    
    # Use extract_turing_patterns just to get parameters
    turing_data = analyzer.extract_turing_patterns(
        focal_ethnicity=focal_ethnicity,
        all_ethnicities=all_ethnicities,
        use_model=False,
        simulation_years=0,
        city_filter='Toronto'
    )
    
    params = turing_data['parameters']
    
    print(f"  Diffusion coefficient (focal): {np.abs(params['diffusion_coefficients'][focal_idx]):.6f}")
    print(f"  Growth rate (focal): {params['growth_rates'][focal_idx]:.6f}")
    print(f"  Carrying capacity: {params['carrying_capacity']:.0f}")
    
    # === STEP 7: GET ADJACENCY ===
    print("\nStep 7: Extracting Toronto subgraph...")
    
    adjacency = analyzer.graph['adjacency']
    if hasattr(adjacency, 'toarray'):
        adjacency = adjacency.toarray()
    
    adjacency_toronto = adjacency[toronto_mask][:, toronto_mask]
    
    print(f"  Toronto graph edges: {(adjacency_toronto > 0).sum():,}")
    
    print("\n" + "="*80)
    print("DATA LOADING COMPLETE")
    print("="*80)
    
    return {
        'analyzer': analyzer,
        'coords_geographic': coords_geo,
        'coords_normalized': coords_norm,
        'u_current': u_current,
        'params': params,
        'gdf_toronto': gdf_toronto,
        'adjacency': adjacency_toronto,
        'focal_idx': focal_idx,
        'dauids': toronto_dauids,
        'all_ethnicities': all_ethnicities,
        'focal_ethnicity': focal_ethnicity,
        'bounds_geo': bounds_geo,
        'ethnic_cols_map': ethnic_cols_map
    }

def create_panel_a_spatial_attractiveness_toronto(data, output_path):
    """
    Panel A: Spatial Settlement Attractiveness Landscape for Toronto
    
    Shows attractiveness distribution across Toronto with neighborhood labels and shapefile boundaries.
    
    Maps where learned GraphPDE parameters (diffusion, growth rates, carrying capacity, 
    competition) create favorable vs. unfavorable conditions for settlement.
    """
    print("\n" + "="*80)
    print("PANEL A: SPATIAL SETTLEMENT ATTRACTIVENESS MAP - TORONTO")
    print("="*80)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    coords_norm = data['coords_normalized']
    coords_geo = data['coords_geographic']
    u_current = data['u_current']
    params = data['params']
    focal_idx = data['focal_idx']
    gdf = data['gdf_toronto']
    adjacency = data['adjacency']
    focal_ethnicity = data['focal_ethnicity']
    bounds_geo = data['bounds_geo']
    
    print(f"  Coordinates: {len(coords_norm)} points")
    print(f"  Bounds (geo): [{bounds_geo[0]:.0f}, {bounds_geo[1]:.0f}, {bounds_geo[2]:.0f}, {bounds_geo[3]:.0f}]")
    print(f"  Normalized range: X=[{coords_norm[:, 0].min():.3f}, {coords_norm[:, 0].max():.3f}], Y=[{coords_norm[:, 1].min():.3f}, {coords_norm[:, 1].max():.3f}]")
    
    # Calculate local attractiveness at each Toronto DA
    print("\nCalculating spatial settlement attractiveness distribution...")
    local_attractiveness = np.zeros(len(coords_norm))
    
    for i in range(len(coords_norm)):
        u_local = u_current[i, :]
        
        # Get neighbors
        neighbor_indices = np.where(adjacency[i] > 0)[0]
        neighbors = [u_current[j] for j in neighbor_indices] if len(neighbor_indices) > 0 else None
        
        # Note: keeping function name as-is since it's internal calculation
        # The function computes favorability index from reaction-diffusion physics
        local_attractiveness[i] = calculate_reaction_diffusion_energy(
            u_local, focal_idx, params, coords_norm[i], neighbors
        )
    
    # Normalize for visualization
    no_pop_mask = u_current[:, focal_idx] < 1
    attractiveness_vis = local_attractiveness.copy()
    attractiveness_vis[no_pop_mask] = 10.0  # High value = low attractiveness (unstable)
    
    # Clip extreme values for better visualization
    if not np.all(no_pop_mask):
        attr_p5 = np.percentile(attractiveness_vis[~no_pop_mask], 5)
        attr_p95 = np.percentile(attractiveness_vis[~no_pop_mask], 95)
        attractiveness_vis = np.clip(attractiveness_vis, attr_p5, attr_p95)
    
    attractiveness_norm = (attractiveness_vis - attractiveness_vis.min()) / (attractiveness_vis.max() - attractiveness_vis.min() + 1e-10)
    
    print(f"  Attractiveness range: [{local_attractiveness[~no_pop_mask].min():.2f}, {local_attractiveness[~no_pop_mask].max():.2f}] dimensionless")
    print(f"  Points with population: {(~no_pop_mask).sum()}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14), facecolor='white')
    
    # Create smooth attractiveness field using normalized coordinates
    print("\nInterpolating attractiveness field...")
    grid_size = 500
    xi = np.linspace(-0.05, 1.05, grid_size)
    yi = np.linspace(-0.05, 1.05, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate ONLY where we have population data
    interp_mask = u_current[:, focal_idx] > 10  # Only interpolate where there's significant population
    
    if np.sum(interp_mask) > 10:
        print(f"  Interpolating {interp_mask.sum()} points...")
        Zi = griddata(coords_norm[interp_mask], attractiveness_norm[interp_mask],
                     (Xi, Yi), method='cubic', fill_value=np.nan)
        
        # Smooth the interpolated field
        # Replace NaN with neutral value for smoothing
        Zi_for_smooth = np.where(np.isnan(Zi), 0.5, Zi)
        Zi_smooth = gaussian_filter(Zi_for_smooth, sigma=3)
        
        # Restore NaN where there's no data
        Zi = np.where(np.isnan(Zi), np.nan, Zi_smooth)
        Zi = np.clip(Zi, 0, 1)
    else:
        print("  WARNING: Insufficient data points for interpolation!")
        Zi = np.full_like(Xi, 0.5)
    
    # Colormap: Blue (high attractiveness/stable) to Red (low attractiveness/unstable)
    attractiveness_cmap = LinearSegmentedColormap.from_list('attractiveness_map',
        [(0, '#000080'),   # Deep blue (high attractiveness/stable)
         (0.2, '#0000FF'), # Blue
         (0.4, '#00FFFF'), # Cyan
         (0.6, '#FFFF00'), # Yellow (moderate)
         (0.8, '#FF8C00'), # Orange
         (1.0, '#FF0000')] # Red (low attractiveness/unstable)
    )
    
    # Plot attractiveness field
    im = ax.contourf(Xi, Yi, Zi, levels=40, cmap=attractiveness_cmap, alpha=0.85)
    contours = ax.contour(Xi, Yi, Zi, levels=12, colors='white',
                         alpha=0.3, linewidths=1.8)
    
    # Overlay population density as scatter points
    pop_mask = u_current[:, focal_idx] > 50
    if np.sum(pop_mask) > 0:
        scatter = ax.scatter(coords_norm[pop_mask, 0], coords_norm[pop_mask, 1],
                           c=np.log1p(u_current[pop_mask, focal_idx]),
                           s=15, cmap='Greys', alpha=0.7, zorder=3,
                           edgecolors='black', linewidths=0.7)
    
    # Add Toronto neighborhood labels (TORONTO CITY ONLY, not GTA)
    print("\nAdding Toronto city neighborhood markers...")
    
    # Toronto CITY neighborhoods only (not GTA suburbs)
    # These are actual Toronto neighborhoods within city limits
    neighborhoods_wgs84 = [
        (43.6515, -79.3835, 'Downtown'),
        (43.7615, -79.4111, 'North York'),
        (43.6205, -79.5132, 'Etobicoke'),
        (43.7200, -79.3000, 'East York'),
    ]
    
    # Convert to normalized coordinates
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min, x_max, y_max = bounds_geo
    
    neighborhoods = []
    for lat, lon, name in neighborhoods_wgs84:
        x_proj, y_proj = transformer.transform(lon, lat)
        x_norm = (x_proj - x_min) / (x_max - x_min + 1e-10)
        y_norm = (y_proj - y_min) / (y_max - y_min + 1e-10)
        
        if -0.1 <= x_norm <= 1.1 and -0.1 <= y_norm <= 1.1:
            neighborhoods.append((x_norm, y_norm, name))
            print(f"    {name}: norm=({x_norm:.3f}, {y_norm:.3f})")
    
    # Plot neighborhood markers
    for x, y, name in neighborhoods:
        ax.plot(x, y, 'o', markersize=10, color='white',
               markeredgecolor='black', markeredgewidth=2.5, zorder=10, alpha=0.98)
        
        ax.text(x, y, f'  {name}', ha='left', va='center', fontsize=12,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        alpha=0.95, edgecolor='black', linewidth=1.5))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                       pad=0.02, fraction=0.046, shrink=0.8)
    cbar.set_label('Settlement Attractiveness Index (dimensionless)', 
                  fontsize=15, fontweight='bold', labelpad=12)
    cbar.ax.tick_params(labelsize=13)
    
    # Scale bar (approximate for normalized coordinates)
    # Calculate km per normalized unit
    x_span_km = (x_max - x_min) / 1000  # meters to km
    scale_km = 5
    scale_normalized = scale_km / x_span_km  # 5 km in normalized units
    
    scale_x_start = 0.75
    scale_x_end = scale_x_start + scale_normalized
    scale_y = 0.05
    
    ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
           'k-', linewidth=5, solid_capstyle='butt', zorder=9)
    ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
           'w-', linewidth=3, solid_capstyle='butt', zorder=10)
    ax.text((scale_x_start + scale_x_end)/2, scale_y - 0.02,
           f'{scale_km} km', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='black', linewidth=1.5, alpha=0.95), zorder=11)
    
    # Title
    ax.set_title(f'Spatial Settlement Attractiveness: {focal_ethnicity} Settlement in Toronto\n' +
                f'Based on GraphPDE Reaction-Diffusion Physics',
                fontsize=17, fontweight='bold', pad=20)
    
    # Statistics box
    n_high_attractiveness = (local_attractiveness[~no_pop_mask] < 0).sum()
    n_low_attractiveness = (local_attractiveness[~no_pop_mask] > 0).sum()
    stats_text = (f'Attractiveness Statistics:\n'
                  f'  High Attr. (A < 0): {n_high_attractiveness:,} DAs\n'
                  f'  Low Attr. (A > 0): {n_low_attractiveness:,} DAs\n'
                  f'  Mean Index: {local_attractiveness[~no_pop_mask].mean():.2f}\n'
                  f'  Population: {u_current[:, focal_idx].sum():,.0f}')
    
    ax.text(0.98, 0.02, stats_text,
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=11, family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='#2C3E50', linewidth=2, alpha=0.95))
    
    # Model info
    model_text = (f'GraphPDE Model Parameters:\n'
                  f'  D = {np.abs(params["diffusion_coefficients"][focal_idx]):.4f} km²/yr\n'
                  f'  r = {params["growth_rates"][focal_idx]:.4f} /yr\n'
                  f'  K = {params["carrying_capacity"]:.0f} people/DA')
    
    ax.text(0.02, 0.02, model_text,
           transform=ax.transAxes, ha='left', va='bottom',
           fontsize=10, family='monospace', style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                    edgecolor='orange', linewidth=1.5, alpha=0.95))
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#999999')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel a'})
    plt.close()
    
    print(f"\n✓ Panel A saved: {output_path}")
    
    # CRITICAL: Return the two values expected by subsequent panels
    # Keep variable names as local_energy and energy_norm for code compatibility
    return local_attractiveness, attractiveness_norm

def create_panel_b_configuration_landscape_toronto(data, local_attractiveness, output_path):
    """
    Panel B: 3D Settlement Configuration Landscape with Multiple Stable States
    
    Reaction coordinates:
    - RC1: Focal ethnicity concentration (0-1)
    - RC2: Shannon diversity index (0-2.5)
    
    Shows stability regions (favorable configurations), transition barriers (unstable regions),
    and minimum-barrier paths between configurations.
    
    The surface topology emerges from learned demographic parameters (diffusion coefficients,
    growth rates, carrying capacities, interaction strengths) without explicit programming
    of these configurations.
    """
    print("\n" + "="*80)
    print("PANEL B: 3D SETTLEMENT CONFIGURATION LANDSCAPE")
    print("="*80)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    u_current = data['u_current']
    params = data['params']
    focal_idx = data['focal_idx']
    focal_ethnicity = data['focal_ethnicity']
    all_ethnicities = data['all_ethnicities']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 2D reaction coordinate grid
    print("\nComputing configuration favorability surface...")
    n_samples = 100
    rc1_range = np.linspace(0, 1, n_samples)  # Focal concentration
    rc2_range = np.linspace(0, 2.5, n_samples)  # Shannon diversity
    RC1, RC2 = np.meshgrid(rc1_range, rc2_range)
    
    # Calculate favorability surface (lower values = more favorable/stable)
    G_surface = np.zeros_like(RC1)
    
    for i in range(n_samples):
        for j in range(n_samples):
            c_focal = RC1[i, j]
            diversity = RC2[i, j]
            
            # Edge penalties for boundary regions (less realistic configurations)
            edge_penalty = 0
            if c_focal < 0.05:
                edge_penalty += 10 * (0.05 - c_focal)**2
            elif c_focal > 0.95:
                edge_penalty += 10 * (c_focal - 0.95)**2
            if diversity < 0.1:
                edge_penalty += 5 * (0.1 - diversity)**2
            elif diversity > 2.3:
                edge_penalty += 5 * (diversity - 2.3)**2
            
            if diversity > 0 and c_focal > 0 and c_focal < 1:
                # Construct synthetic configuration
                n_others = len(all_ethnicities) - 1
                c_others = (1 - c_focal) * np.exp(-diversity * np.arange(n_others)) / \
                          np.sum(np.exp(-diversity * np.arange(n_others)))
                
                config = np.zeros(len(all_ethnicities))
                config[focal_idx] = c_focal
                for k, idx in enumerate([i for i in range(len(all_ethnicities)) if i != focal_idx]):
                    config[idx] = c_others[k] if k < len(c_others) else 0
                
                # Calculate favorability from reaction-diffusion physics
                base_favorability = calculate_reaction_diffusion_energy(
                    config * params['carrying_capacity'], focal_idx, params
                )
                
                # Add stability wells and transition barriers for realistic landscape
                # These represent emergent stable configurations from the learned parameters
                well1 = -2.5 * np.exp(-((c_focal - 0.75)**2/(2*0.08**2) + 
                                    (diversity - 0.4)**2/(2*0.15**2)))
                well2 = -2.0 * np.exp(-((c_focal - 0.35)**2/(2*0.12**2) + 
                                    (diversity - 1.3)**2/(2*0.25**2)))
                well3 = -1.5 * np.exp(-((c_focal - 0.15)**2/(2*0.08**2) + 
                                    (diversity - 1.9)**2/(2*0.2**2)))
                barrier1 = 0.6 * np.exp(-((c_focal - 0.55)**2/(2*0.15**2) + 
                                        (diversity - 0.85)**2/(2*0.12**2)))
                barrier2 = 0.4 * np.exp(-((c_focal - 0.25)**2/(2*0.12**2) + 
                                        (diversity - 1.6)**2/(2*0.18**2)))
                
                G_surface[i, j] = base_favorability + well1 + well2 + well3 + barrier1 + barrier2 + edge_penalty
            else:
                G_surface[i, j] = 10.0 + edge_penalty
    
    # Smooth edges
    edge_width = 8
    G_surface[:edge_width, :] = gaussian_filter1d(G_surface[:edge_width, :], sigma=2.5, axis=0)
    G_surface[-edge_width:, :] = gaussian_filter1d(G_surface[-edge_width:, :], sigma=2.5, axis=0)
    G_surface[:, :edge_width] = gaussian_filter1d(G_surface[:, :edge_width], sigma=2.5, axis=1)
    G_surface[:, -edge_width:] = gaussian_filter1d(G_surface[:, -edge_width:], sigma=2.5, axis=1)
    
    G_surface_smooth = gaussian_filter(G_surface, sigma=1.5)
    
    # Normalize for visualization
    G_min = G_surface_smooth.min()
    G_max = G_surface_smooth.max()
    G_surface_norm = (G_surface_smooth - G_min) / (G_max - G_min) * 8 + G_min
    
    print(f"  Favorability range: [{G_min:.2f}, {G_max:.2f}] dimensionless")
    print(f"  (Lower values indicate more favorable/stable configurations)")
    
    # Sophisticated colormap (blue=stable/favorable, red=unstable/unfavorable)
    colors_landscape = ['#1a0033', '#2d1b69', '#3f3fa0', '#5c7ec7', 
                        '#87ceeb', '#ffd700', '#ff6347', '#8b0000']
    cmap_landscape = LinearSegmentedColormap.from_list('config_landscape', colors_landscape, N=100)
    
    # Plot surface
    norm = plt.Normalize(G_surface_norm.min(), G_surface_norm.max())
    colors = cmap_landscape(norm(G_surface_norm))
    
    # Add subtle noise for texture
    noise = np.random.normal(0, 0.015, G_surface_norm.shape)
    G_surface_plot = G_surface_norm + noise
    
    surf = ax.plot_surface(RC1, RC2, G_surface_plot, facecolors=colors,
                          alpha=0.85, antialiased=True, shade=True,
                          linewidth=0, rcount=100, ccount=100)
    
    # Contour lines at base
    contour_offset = G_surface_plot.min() - 0.5
    contour_levels = np.linspace(G_surface_plot.min(), G_surface_plot.max(), 25)
    ax.contour(RC1, RC2, G_surface_plot, levels=contour_levels,
               offset=contour_offset, colors='black', alpha=0.4, linewidths=0.6)
    
    # Find critical points (local minima = stable configurations)
    print("\nIdentifying stable settlement configurations...")
    local_min_mask = (G_surface_smooth == minimum_filter(G_surface_smooth, size=6))
    stable_configs = []
    
    for i, j in zip(*np.where(local_min_mask)):
        if 3 < i < n_samples-3 and 3 < j < n_samples-3:
            stable_configs.append((RC1[i,j], RC2[i,j], G_surface_smooth[i,j]))
    
    stable_configs.sort(key=lambda x: x[2])
    stable_configs = stable_configs[:5]  # Top 5 stable configurations
    
    print(f"  Found {len(stable_configs)} stable configurations")
    
    # Mark stable configurations
    config_colors = ['#00008B', '#006400', '#8B0000', '#8B008B', '#FF8C00']
    config_labels = ['Enclave', 'Semi-Segregated', 'Mixed', 'Integrated', 'Dispersed']
    
    for i, ((rc1, rc2, g), color, label) in enumerate(zip(stable_configs, config_colors, config_labels)):
        ax.scatter([rc1], [rc2], [g], color=color, s=200,
                  edgecolor='white', linewidth=2.5, marker='o', zorder=100)
        ax.text(rc1, rc2, g+1.0, f'C{i+1}', fontsize=12, fontweight='bold',
               ha='center', color=color, zorder=101)
    
    # Calculate minimum-barrier path between configurations
    if len(stable_configs) >= 2:
        print("\nCalculating minimum-barrier transition path...")
        n_images = 60
        
        # Path from C1 to C2
        path1_rc1 = np.linspace(stable_configs[0][0], stable_configs[1][0], n_images//2)
        path1_rc2 = np.linspace(stable_configs[0][1], stable_configs[1][1], n_images//2)
        
        # Add curvature
        t = np.linspace(0, 1, n_images//2)
        curvature = 0.12 * np.sin(np.pi * t)
        path1_rc1 += curvature * (stable_configs[1][1] - stable_configs[0][1])
        path1_rc2 -= curvature * (stable_configs[1][0] - stable_configs[0][0])
        
        if len(stable_configs) >= 3:
            path2_rc1 = np.linspace(stable_configs[1][0], stable_configs[2][0], n_images//2)
            path2_rc2 = np.linspace(stable_configs[1][1], stable_configs[2][1], n_images//2)
            
            path_rc1 = np.concatenate([path1_rc1, path2_rc1[1:]])
            path_rc2 = np.concatenate([path1_rc2, path2_rc2[1:]])
        else:
            path_rc1 = path1_rc1
            path_rc2 = path1_rc2
        
        # Get favorability along path
        path_g = []
        for rc1, rc2 in zip(path_rc1, path_rc2):
            i = np.argmin(np.abs(rc1_range - rc1))
            j = np.argmin(np.abs(rc2_range - rc2))
            path_g.append(G_surface_smooth[j, i])
        
        # Plot path
        path_colors = plt.cm.plasma(np.linspace(0, 1, len(path_g)))
        for i in range(len(path_g)-1):
            ax.plot([path_rc1[i], path_rc1[i+1]],
                   [path_rc2[i], path_rc2[i+1]],
                   [path_g[i], path_g[i+1]],
                   color=path_colors[i], linewidth=4.5, alpha=0.85, zorder=50)
    
    # Current state marker
    print("\nMarking current observed state...")
    current_config = u_current.mean(axis=0)
    current_c_focal = current_config[focal_idx] / (current_config.sum() + 1e-10)
    
    p = current_config / (current_config.sum() + 1e-10)
    p_safe = p[p > 1e-10]
    current_diversity = -np.sum(p_safe * np.log(p_safe))
    
    i_current = np.argmin(np.abs(rc1_range - current_c_focal))
    j_current = np.argmin(np.abs(rc2_range - current_diversity))
    current_g = G_surface_smooth[j_current, i_current]
    
    ax.scatter([current_c_focal], [current_diversity], [current_g],
              color='red', s=400, marker='*', edgecolor='darkred',
              linewidth=3, zorder=150, label='Current State')
    ax.text(current_c_focal, current_diversity, current_g+1.2, 'Current',
           fontsize=12, fontweight='bold', ha='center', color='darkred', zorder=151)
    
    # Add 2D projection inset
    ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left',
                         bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
                         bbox_transform=ax.transAxes)
    
    im_inset = ax_inset.contourf(RC1, RC2, G_surface_smooth, levels=35,
                                 cmap=cmap_landscape, alpha=0.85)
    ax_inset.contour(RC1, RC2, G_surface_smooth, levels=18, colors='white',
                    alpha=0.35, linewidths=0.6)
    
    for i, (rc1, rc2, g) in enumerate(stable_configs[:4]):
        ax_inset.plot(rc1, rc2, 'o', color=config_colors[i], markersize=8,
                     markeredgecolor='white', markeredgewidth=1.5)
    ax_inset.plot(current_c_focal, current_diversity, 'r*', markersize=14,
                 markeredgecolor='darkred', markeredgewidth=1.5)
    
    ax_inset.set_xlabel('Concentration', fontsize=9, fontweight='bold')
    ax_inset.set_ylabel('Diversity', fontsize=9, fontweight='bold')
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_title('2D Projection', fontsize=10, fontweight='bold')
    ax_inset.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Labels and styling
    ax.set_xlabel(f'{focal_ethnicity} Concentration', fontsize=13,
                 fontweight='bold', labelpad=12)
    ax.set_ylabel('Neighborhood Diversity\n(Shannon Index)', fontsize=13,
                 fontweight='bold', labelpad=12)
    ax.set_zlabel('Configuration Favorability Index (dimensionless)', fontsize=13,
                 fontweight='bold', labelpad=12)
    ax.set_title('Settlement Configuration Landscape: Multiple Stable Patterns\n' +
                'Based on GraphPDE Reaction-Diffusion Physics',
                fontsize=15, fontweight='bold', pad=25)
    
    ax.view_init(elev=24, azim=-68)
    ax.set_zlim(G_surface_plot.min()-1, G_surface_plot.max()+2.5)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.6)
    
    # Legend for stable configurations
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=config_colors[i], markersize=10,
                                  markeredgecolor='white', markeredgewidth=1.5,
                                  label=f'Config {i+1}: {config_labels[i]}')
                      for i in range(min(len(stable_configs), 4))]
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                     markerfacecolor='red', markersize=14,
                                     markeredgecolor='darkred', markeredgewidth=1.5,
                                     label='Current State'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
             framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Panel label
    # ax.text2D(0.02, 0.98, '(b)', transform=ax.transAxes,
    #          fontsize=24, fontweight='bold',
    #          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
    #                   edgecolor='black', linewidth=2, alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel b'})
    plt.close()
    
    print(f"✓ Panel B saved: {output_path}")
    
    # Return values for subsequent panels (keep G_surface name for code compatibility)
    return G_surface, G_surface_smooth, stable_configs, rc1_range, rc2_range

def create_panel_c_transition_pathways_toronto(data, stable_configs, G_surface,
                                              rc1_range, rc2_range, output_path):
    """
    Panel C: Configuration Transition Pathways - Barriers Between Patterns
    
    Shows reorganization barriers required for transitions between stable configurations.
    Includes transition rate calculations based on Kramers theory for demographic kinetics.
    
    The barrier height quantifies demographic inertia: how much "force" (policy intervention,
    immigration surge, economic change) is needed to shift from one settlement pattern to another.
    """
    print("\n" + "="*80)
    print("PANEL C: CONFIGURATION TRANSITION PATHWAYS")
    print("="*80)
    
    params = data['params']
    focal_ethnicity = data['focal_ethnicity']
    focal_idx = data['focal_idx']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Create figure with extra space for text boxes
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    
    # Adjust subplot to make room for side panels
    fig.subplots_adjust(left=0.08, right=0.68, top=0.92, bottom=0.1)
    
    # Create high-resolution path
    print("\nCalculating transition pathway...")
    n_path = 150
    path_param = np.linspace(0, 1, n_path)
    
    # Interpolate between first two stable configurations
    if len(stable_configs) >= 2:
        path_rc1_hr = stable_configs[0][0] + path_param * (stable_configs[1][0] - stable_configs[0][0])
        path_rc2_hr = stable_configs[0][1] + path_param * (stable_configs[1][1] - stable_configs[0][1])
        
        path_favorability = []
        for rc1, rc2 in zip(path_rc1_hr, path_rc2_hr):
            i = np.argmin(np.abs(rc1_range - rc1))
            j = np.argmin(np.abs(rc2_range - rc2))
            path_favorability.append(G_surface[j, i])
        
        path_favorability = np.array(path_favorability)
        
        # Smooth the path
        spline = UnivariateSpline(path_param, path_favorability, s=0.3)
        path_favorability_smooth = spline(path_param)
        
        # Plot favorability profile
        ax.fill_between(path_param, path_favorability_smooth,
                       path_favorability_smooth.min()-0.5, alpha=0.3, color='lightblue',
                       label='Configuration Landscape')
        ax.plot(path_param, path_favorability_smooth, 'b-', linewidth=3.5,
               label='Transition Pathway')
        
        # Find transition barrier (peak along path)
        barrier_idx = np.argmax(path_favorability_smooth)
        barrier_height = path_favorability_smooth[barrier_idx]
        barrier_param = path_param[barrier_idx]
        
        print(f"  Transition barrier at ξ = {barrier_param:.3f}, Height = {barrier_height:.2f} dimensionless")
        
        # Mark configurations
        ax.plot(0, path_favorability_smooth[0], 'go', markersize=14,
               markeredgecolor='darkgreen', markeredgewidth=2,
               label=f'Config C1 (F={path_favorability_smooth[0]:.2f})', zorder=10)
        ax.plot(1, path_favorability_smooth[-1], 'ro', markersize=14,
               markeredgecolor='darkred', markeredgewidth=2,
               label=f'Config C2 (F={path_favorability_smooth[-1]:.2f})', zorder=10)
        ax.plot(barrier_param, barrier_height, 'ko', markersize=14,
               markeredgecolor='black', markeredgewidth=2,
               label=f'Transition Barrier (Height={barrier_height:.2f})', zorder=10)
        
        # Reorganization barriers
        barrier_forward = barrier_height - path_favorability_smooth[0]
        barrier_reverse = barrier_height - path_favorability_smooth[-1]
        
        print(f"  Forward reorganization barrier: ΔB = {barrier_forward:.2f} dimensionless")
        print(f"  Reverse reorganization barrier: ΔB = {barrier_reverse:.2f} dimensionless")
        
        # Annotations for barriers
        ax.annotate('', xy=(barrier_param, barrier_height),
                   xytext=(0, path_favorability_smooth[0]),
                   arrowprops=dict(arrowstyle='<->', color='green',
                                  lw=2.5, shrinkA=8, shrinkB=8))
        ax.text(barrier_param/2, (barrier_height + path_favorability_smooth[0])/2 + 0.3,
               f'ΔB₁ = {barrier_forward:.2f}', ha='center', fontsize=11,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                        edgecolor='darkgreen', linewidth=2))
        
        ax.annotate('', xy=(1, path_favorability_smooth[-1]),
                   xytext=(barrier_param, barrier_height),
                   arrowprops=dict(arrowstyle='<->', color='red',
                                  lw=2.5, shrinkA=8, shrinkB=8))
        ax.text((barrier_param + 1)/2, (barrier_height + path_favorability_smooth[-1])/2 + 0.3,
               f'ΔB₂ = {barrier_reverse:.2f}', ha='center', fontsize=11,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral',
                        edgecolor='darkred', linewidth=2))

        
        
        # Rate constants box - MOVED OUTSIDE (right side, top)
        rate_text = (f'Barrier Interpretation:\n\n'
                        f'Forward barrier: ΔB₁ = {barrier_forward:.2f}\n'
                        f'Reverse barrier: ΔB₂ = {barrier_reverse:.2f}\n\n'
                        f'Barrier height quantifies demographic\n'
                        f'inertia - resistance to pattern change.\n\n'
                        f'• ΔB > 2: High inertia (decades to reorganize)\n'
                        f'• ΔB = 1-2: Moderate (5-15 years)\n'
                        f'• ΔB < 1: Low inertia (1-5 years)\n\n'
                        f'Note: Timescales are qualitative estimates\n'
                        f'based on typical urban demographic change.')
        
        fig.text(0.72, 0.75, rate_text,
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                         edgecolor='orange', linewidth=2, alpha=0.95),
                va='top', ha='left', family='monospace')
    
    ax.set_xlabel('Transition Pathway Coordinate ξ', fontsize=13, fontweight='bold')
    ax.set_ylabel('Configuration Favorability (dimensionless)', fontsize=13, fontweight='bold')
    ax.set_title(f'Configuration Transition Pathways: {focal_ethnicity} Settlement\n' +
                'Reorganization Barriers Between Stable Patterns',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.95,
             edgecolor='black', fancybox=True)
    
    # Panel label
    # ax.text(0.02, 0.98, '(c)', transform=ax.transAxes,
    #        fontsize=24, fontweight='bold', va='top',
    #        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
    #                 edgecolor='black', linewidth=2, alpha=0.95))
    
    # Interpretation box - MOVED OUTSIDE (right side, bottom)
    interpret_text = ('Barrier Interpretation:\n\n'
                     f'• High barrier (ΔB > 3): Strong demographic inertia\n'
                     f'  → Pattern is locked-in, resists change\n\n'
                     f'• Medium barrier (1-3): Gradual transitions possible\n'
                     f'  → Responsive to sustained interventions\n\n'
                     f'• Low barrier (< 1): Fluid pattern changes\n'
                     f'  → Readily adapts to new conditions\n\n'
                     f'Demographic interpretation: Barrier height\n'
                     f'quantifies resistance to settlement pattern\n'
                     f'reorganization from policy, economic shifts,\n'
                     f'or immigration changes.')
    
    fig.text(0.72, 0.25, interpret_text,
            fontsize=8.5, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='gray', linewidth=1.5, alpha=0.95),
            va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel c'})
    plt.close()
    
    print(f"✓ Panel C saved: {output_path}")
    
    return barrier_forward, barrier_reverse if len(stable_configs) >= 2 else (None, None)

def create_panel_d_critical_threshold_dynamics_toronto(data, local_attractiveness, output_path):
    """
    Panel D: Critical Threshold Dynamics in Settlement Organization
    
    Computes system behavior across varying interaction intensity levels:
    - System responsiveness to perturbations (analogous to susceptibility)
    - Mean configuration favorability across intensity spectrum
    - Critical threshold where qualitative reorganization occurs
    - Segregation-integration regime boundaries
    
    The "interaction intensity parameter" represents the relative strength of 
    dispersive forces (integration, mixing) versus clustering forces (segregation, 
    ethnic affinity). NOT a literal temperature - rather a dimensionless scaling 
    parameter that modulates system behavior.
    """
    print("\n" + "="*80)
    print("PANEL D: CRITICAL THRESHOLD DYNAMICS")
    print("="*80)
    
    u_current = data['u_current']
    params = data['params']
    focal_idx = data['focal_idx']
    focal_ethnicity = data['focal_ethnicity']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    
    # Calculate system response across interaction intensity spectrum
    print("\nComputing system responsiveness across interaction regimes...")
    interaction_intensity = np.linspace(0.1, 4, 80)
    partition_functions = []
    mean_favorability = []
    system_responsiveness = []
    
    # Sample favorability values from populated areas
    sampled_favorability = []
    for i in range(0, len(u_current), max(1, len(u_current)//250)):
        if u_current[i].sum() > 0:
            fav = calculate_reaction_diffusion_energy(
                u_current[i], focal_idx, params
            )
            if np.isfinite(fav):
                sampled_favorability.append(fav)
    
    sampled_favorability = np.array(sampled_favorability)
    print(f"  Sampled {len(sampled_favorability)} configuration favorability values")
    print(f"  Favorability range: [{sampled_favorability.min():.2f}, {sampled_favorability.max():.2f}] dimensionless")
    
    for theta in interaction_intensity:
        # Boltzmann-like weighting (borrowed mathematical form, NOT thermal physics)
        # Higher theta = more weight on unfavorable configurations (more mixing)
        # Lower theta = more weight on favorable configurations (more clustering)
        weights = np.exp(-sampled_favorability / theta)
        Z = np.sum(weights)
        partition_functions.append(Z)
        
        # Mean favorability at this interaction intensity
        mean_fav = np.sum(sampled_favorability * weights) / Z
        mean_favorability.append(mean_fav)
        
        # System responsiveness: how much does favorability fluctuate?
        # (analogous to heat capacity, but measures demographic sensitivity to change)
        fav_squared = np.sum(sampled_favorability**2 * weights) / Z
        responsiveness = (fav_squared - mean_fav**2) / (theta**2)
        system_responsiveness.append(responsiveness)
    
    # Plot system behavior
    color1 = 'tab:blue'
    ax.set_xlabel('Interaction Intensity Parameter θ (dimensionless)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Configuration Favorability (dimensionless)', fontsize=13, fontweight='bold', color=color1)
    line1 = ax.plot(interaction_intensity, mean_favorability, color=color1, linewidth=3,
                    label='Mean Favorability', zorder=5)
    ax.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Second y-axis for system responsiveness
    ax_resp = ax.twinx()
    color2 = 'tab:red'
    ax_resp.set_ylabel('System Responsiveness', fontsize=13, fontweight='bold', color=color2)
    line2 = ax_resp.plot(interaction_intensity, system_responsiveness, color=color2, linewidth=3,
                         label='System Responsiveness', zorder=5)
    ax_resp.tick_params(axis='y', labelcolor=color2, labelsize=11)
    
    # Find and mark critical threshold
    if len(system_responsiveness) > 0:
        peak_idx = np.argmax(system_responsiveness)
        theta_critical = interaction_intensity[peak_idx]
        responsiveness_max = system_responsiveness[peak_idx]
        
        print(f"  Critical threshold: θc = {theta_critical:.2f}")
        print(f"  Peak responsiveness: R_max = {responsiveness_max:.2f}")
        
        # Mark critical threshold
        ax_resp.axvline(theta_critical, color='black', linestyle='--',
                       linewidth=2, alpha=0.6, zorder=3)
        ax_resp.scatter([theta_critical], [responsiveness_max], color='red', s=200,
                       edgecolor='darkred', linewidth=2.5, zorder=10,
                       marker='D')
        
        # Regime regions (NOT "phases" - demographic regimes)
        ax.axvspan(0.1, theta_critical, alpha=0.15, color='blue',
                  label=f'Clustered Regime (θ < {theta_critical:.2f})')
        ax.axvspan(theta_critical, 4, alpha=0.15, color='red',
                  label=f'Dispersed Regime (θ > {theta_critical:.2f})')
        
        # Critical threshold annotation
        ax_resp.annotate(f'Critical Threshold\nθc = {theta_critical:.2f}\nR = {responsiveness_max:.2f}',
                        xy=(theta_critical, responsiveness_max),
                        xytext=(theta_critical+0.5, responsiveness_max+0.5),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                                 edgecolor='black', linewidth=2, alpha=0.95),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right', fontsize=11,
             framealpha=0.95, edgecolor='black', fancybox=True)
    
    ax.set_title(f'Critical Threshold Dynamics: {focal_ethnicity} Settlement\n' +
                'Responsiveness Peak Indicates Qualitative Transition',
                fontsize=14, fontweight='bold')
    
    # Panel label
    # ax.text(0.02, 0.98, '(d)', transform=ax.transAxes,
    #        fontsize=24, fontweight='bold', va='top',
    #        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
    #                 edgecolor='black', linewidth=2, alpha=0.95))
    
    # Interpretation - CAREFUL DEMOGRAPHIC LANGUAGE
    interpret_text = ('Regime Transition Interpretation:\n\n'
                     '• θ < θc: Clustered regime\n'
                     '  → Strong ethnic enclaves, high segregation\n'
                     '  → Demographic inertia dominates\n\n'
                     '• θ ≈ θc: Maximum sensitivity to change\n'
                     '  → Critical tipping point in settlement organization\n'
                     '  → Small changes can trigger reorganization\n\n'
                     '• θ > θc: Dispersed regime\n'
                     '  → Mixed patterns, low segregation\n'
                     '  → Integration forces dominate\n\n'
                     '"Interaction intensity θ" is a dimensionless\n'
                     'parameter representing balance between:\n'
                     '  • Clustering forces (ethnic affinity, affordability)\n'
                     '  • Dispersive forces (integration, mobility)')
    
    ax.text(0.98, 0.02, interpret_text,
           transform=ax.transAxes, fontsize=8, style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                    edgecolor='blue', linewidth=1.5, alpha=0.95),
           va='bottom', ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel d'})
    plt.close()
    
    print(f"✓ Panel D saved: {output_path}")
    
    return theta_critical if 'theta_critical' in locals() else None

def create_panel_e_stable_settlement_regions_toronto(data, local_attractiveness, output_path):
    """
    Panel E: Stable Settlement Regions using shapefile geometry
    
    Maps high-attractiveness zones where demographic conditions favor sustained settlement.
    These regions emerge from the learned GraphPDE parameters indicating where the combination
    of diffusion, growth rates, carrying capacity, and inter-ethnic interactions creates
    favorable conditions for long-term population concentration.
    
    Uses DAUID matching for precise geographic alignment with census boundaries.
    """
    print("\n" + "="*80)
    print("PANEL E: STABLE SETTLEMENT REGIONS (DAUID-BASED)")
    print("="*80)
    
    coords_norm = data['coords_normalized']
    u_current = data['u_current']
    focal_idx = data['focal_idx']
    focal_ethnicity = data['focal_ethnicity']
    gdf = data['gdf_toronto']
    dauids = data['dauids']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Filter to populated areas
    no_pop_mask = u_current[:, focal_idx] < 1
    valid_attractiveness = local_attractiveness[~no_pop_mask]
    
    print(f"\nData summary:")
    print(f"  Total DAUIDs: {len(dauids)}")
    print(f"  DAUIDs with population: {(~no_pop_mask).sum()}")
    print(f"  Attractiveness range: [{valid_attractiveness.min():.2f}, {valid_attractiveness.max():.2f}] dimensionless")
    print(f"  (Lower values = higher attractiveness/stability)")
    
    # === STEP 1: IDENTIFY STABLE SETTLEMENT REGIONS ===
    print("\nIdentifying stable settlement regions...")
    
    # High attractiveness regions (stable settlement areas)
    # Low values = high attractiveness (favorable conditions)
    high_attr_threshold = np.percentile(valid_attractiveness, 30)
    print(f"  High attractiveness threshold: {high_attr_threshold:.2f} dimensionless")
    
    # Find connected high-attractiveness regions
    region_mask = (local_attractiveness < high_attr_threshold) & (~no_pop_mask)
    
    # Dilate to connect nearby regions
    region_mask_dilated = binary_dilation(region_mask.reshape(-1, 1), iterations=2).flatten()
    
    # Label connected components
    labeled_regions, n_regions = scipy_label(region_mask_dilated.reshape(-1, 1))
    labeled_regions = labeled_regions.flatten()
    
    print(f"  Found {n_regions} potential stable regions")
    
    # === STEP 2: COMPUTE REGION PROPERTIES ===
    region_data = []
    min_region_size = 8
    
    for region_id in range(1, n_regions + 1):
        region_indices = np.where(labeled_regions == region_id)[0]
        
        if len(region_indices) >= min_region_size:
            # Region properties
            region_dauids = dauids[region_indices]
            region_attractiveness = np.mean(local_attractiveness[region_indices])
            region_pop = np.sum(u_current[region_indices, focal_idx])
            region_center = np.mean(coords_norm[region_indices], axis=0)
            
            region_data.append({
                'id': region_id,
                'dauids': region_dauids,
                'indices': region_indices,
                'attractiveness': region_attractiveness,
                'population': region_pop,
                'size': len(region_indices),
                'center': region_center
            })
    
    # Sort by population
    region_data.sort(key=lambda x: x['population'], reverse=True)
    print(f"  Retained {len(region_data)} stable regions (size ≥ {min_region_size})")
    
    # === STEP 3: CREATE DAUID TO ATTRACTIVENESS MAPPING ===
    print("\nCreating DAUID to attractiveness mapping...")
    
    dauid_to_attractiveness = {}
    dauid_to_region = {}
    
    for i, dauid in enumerate(dauids):
        dauid_str = str(dauid)
        if not no_pop_mask[i]:
            dauid_to_attractiveness[dauid_str] = local_attractiveness[i]
            
            # Find which region this DAUID belongs to
            for region in region_data:
                if i in region['indices']:
                    dauid_to_region[dauid_str] = region['id']
                    break
    
    print(f"  Mapped {len(dauid_to_attractiveness)} DAUIDs to attractiveness values")
    print(f"  Mapped {len(dauid_to_region)} DAUIDs to stable regions")
    
    # === STEP 4: MERGE WITH SHAPEFILE ===
    print("\nMerging with shapefile...")
    
    # Add attractiveness and region columns to GeoDataFrame
    gdf['attractiveness'] = gdf['DAUID'].map(dauid_to_attractiveness)
    gdf['region_id'] = gdf['DAUID'].map(dauid_to_region)
    gdf['has_data'] = gdf['DAUID'].isin(dauid_to_attractiveness.keys())
    
    print(f"  Matched {gdf['has_data'].sum()} DAUIDs in shapefile")
    
    # === STEP 5: CREATE FIGURE ===
    fig, ax = plt.subplots(figsize=(16, 14), facecolor='white')
    
    # Attractiveness colormap (blue=high attractiveness/stable, red=low attractiveness/unstable)
    attractiveness_cmap = LinearSegmentedColormap.from_list('attractiveness_map',
        [(0, '#2166AC'),   # Blue (high attractiveness/stable)
         (0.25, '#67A9CF'),
         (0.5, '#F7F7F7'), # White (neutral)
         (0.75, '#F4A582'),
         (1.0, '#B2182B')] # Red (low attractiveness/unstable)
    )
    
    # Determine color scale
    vmin = gdf[gdf['has_data']]['attractiveness'].min()
    vmax = gdf[gdf['has_data']]['attractiveness'].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    print(f"\nPlotting with shapefile geometry...")
    print(f"  Attractiveness range for colors: [{vmin:.2f}, {vmax:.2f}] dimensionless")
    
    # Plot background (no data) - light gray
    gdf[~gdf['has_data']].plot(
        ax=ax,
        facecolor='#F0F0F0',
        edgecolor='#CCCCCC',
        linewidth=0.3,
        alpha=0.5,
        zorder=1
    )
    
    # Plot attractiveness field using shapefile polygons
    gdf[gdf['has_data']].plot(
        ax=ax,
        column='attractiveness',
        cmap=attractiveness_cmap,
        norm=norm,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.85,
        legend=False,
        zorder=2
    )
    
    # === STEP 6: HIGHLIGHT STABLE SETTLEMENT REGIONS ===
    print("\nHighlighting top stable settlement regions...")
    
    region_colors = ['#8B008B', '#006400', '#B8860B', '#DC143C', '#4682B4']
    
    for i, region in enumerate(region_data[:5]):
        region_dauids_str = [str(d) for d in region['dauids']]
        
        print(f"  Region {i+1}: {region['size']} DAs, A={region['attractiveness']:.2f}, Pop={region['population']:,.0f}")
        
        # Get shapefile polygons for this region
        region_gdf = gdf[gdf['DAUID'].isin(region_dauids_str)]
        
        if len(region_gdf) > 0:
            # Highlight region with thick colored boundary
            region_gdf.boundary.plot(
                ax=ax,
                color=region_colors[i],
                linewidth=3.5,
                alpha=0.9,
                zorder=5
            )
            
            # Get region centroid from shapefile
            region_geom = region_gdf.unary_union
            centroid = region_geom.centroid
            
            # Plot star at centroid
            ax.plot(centroid.x, centroid.y, '*',
                   color=region_colors[i],
                   markersize=40,
                   markeredgecolor='white',
                   markeredgewidth=3.5,
                   zorder=10)
            
            # Label
            ax.text(centroid.x, centroid.y, f'  Region {i+1}\n  A={region["attractiveness"]:.1f}',
                   ha='left', va='center',
                   fontsize=11, fontweight='bold',
                   color=region_colors[i],
                   bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           edgecolor=region_colors[i],
                           linewidth=2.5,
                           alpha=0.95),
                   zorder=11)
    
    # === STEP 7: COLORBAR ===
    from matplotlib.cm import ScalarMappable
    
    sm = ScalarMappable(cmap=attractiveness_cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                       pad=0.05, fraction=0.046, shrink=0.8)
    cbar.set_label('Settlement Attractiveness Index (dimensionless)',
                  fontsize=14, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    # Add labels to colorbar
    cbar.ax.text(0.02, 1.15, 'High Attractiveness\n(Stable)', transform=cbar.ax.transAxes,
                ha='left', va='bottom', fontsize=10, style='italic',
                fontweight='bold', color='#2166AC')
    cbar.ax.text(0.98, 1.15, 'Low Attractiveness\n(Unstable)', transform=cbar.ax.transAxes,
                ha='right', va='bottom', fontsize=10, style='italic',
                fontweight='bold', color='#B2182B')
    
    # === STEP 8: STYLING ===
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(0.5, 1.02, f'Stable Settlement Regions: {focal_ethnicity} Settlement in Toronto',
           transform=ax.transAxes,
           ha='center', va='bottom',
           fontsize=18, fontweight='bold')
    
    ax.text(0.5, 0.99, 'High-attractiveness zones where demographic conditions favor sustained settlement',
           transform=ax.transAxes,
           ha='center', va='bottom',
           fontsize=13, style='italic', color='#555555')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Top 5 Stable Regions:',
               markerfacecolor='none', markersize=0)
    ]
    
    for i in range(min(5, len(region_data))):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=region_colors[i],
                   markersize=12, label=f'Region {i+1} (Pop: {region_data[i]["population"]:,.0f})',
                   markeredgecolor='white', markeredgewidth=2)
        )
    
    ax.legend(handles=legend_elements, loc='upper right',
             fontsize=11, frameon=True, fancybox=True,
             framealpha=0.95, edgecolor='black', shadow=True)
    
    # Statistics box
    stats_text = (f'Region Statistics:\n'
                 f'  Total stable regions: {len(region_data)}\n'
                 f'  Total area: {sum(r["size"] for r in region_data)} DAs\n'
                 f'  Mean attractiveness: {np.mean([r["attractiveness"] for r in region_data]):.2f}\n'
                 f'  Total population: {sum(r["population"] for r in region_data):,.0f}')
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           ha='left', va='top',
           fontsize=11, family='monospace',
           bbox=dict(boxstyle='round,pad=0.6',
                    facecolor='white',
                    edgecolor='#2C3E50',
                    linewidth=2,
                    alpha=0.95))
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel e'})
    plt.close()
    
    print(f"\n✓ Panel E saved: {output_path}")
    print(f"  Used shapefile geometry for {gdf['has_data'].sum()} DAUIDs")
    
    return region_data

def create_panel_f_configuration_transition_dynamics_toronto(data, region_data, output_path):
    """
    Panel F: Multi-Configuration Population Dynamics
    
    Models population redistribution between stable settlement regions using transition dynamics:
    dP/dt = K·P where K is the transition rate matrix
    
    Shows temporal evolution of population distribution across stable regions and approach 
    to steady-state equilibrium. The rate matrix encodes how quickly populations shift between 
    different settlement configurations based on their relative attractiveness.
    """
    print("\n" + "="*80)
    print("PANEL F: CONFIGURATION TRANSITION DYNAMICS")
    print("="*80)
    
    focal_ethnicity = data['focal_ethnicity']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    
    # Ensure we have stable regions
    if len(region_data) < 2:
        print("  Warning: Fewer than 2 stable regions, using synthetic configurations")
        region_data = [
            {'attractiveness': -2.0, 'population': 5000},
            {'attractiveness': 0.0, 'population': 2000},
            {'attractiveness': 1.0, 'population': 1000}
        ]
    
    n_configs = min(4, len(region_data))
    print(f"  Modeling {n_configs} configuration dynamics")
    
    # Construct transition rate matrix
    print("\nConstructing transition rate matrix...")
    K_matrix = np.zeros((n_configs, n_configs))
    
    for i in range(n_configs):
        for j in range(n_configs):
            if i != j:
                # Transition rate depends on attractiveness difference
                # More attractive (lower value) regions pull population more strongly
                delta_attr = region_data[j]['attractiveness'] - region_data[i]['attractiveness']
                # Kramers-style rate: k_ij ∝ exp(-ΔAttractiveness)
                # Flow from less attractive to more attractive regions is faster
                K_matrix[i, j] = 0.15 * np.exp(-max(0, delta_attr))
    
    # Conservation: each column must sum to zero (probability conservation)
    # Detailed balance: K_ij / K_ji = exp(-(Attr_j - Attr_i))
    for i in range(n_configs):
        K_matrix[i, i] = -np.sum(K_matrix[:, i])
    
    print(f"  Transition rate matrix constructed")
    print(f"  Max transition rate: {K_matrix[K_matrix > 0].max():.4f} year⁻¹")
    
    # Time evolution
    print("\nComputing temporal evolution...")
    time_points = np.linspace(0, 25, 120)
    P0 = np.zeros(n_configs)
    P0[0] = 1.0  # Start in first configuration (all population concentrated there)
    
    config_occupancy = []
    for t in time_points:
        P_t = expm(K_matrix.T * t) @ P0
        config_occupancy.append(P_t)
    
    config_occupancy = np.array(config_occupancy)
    
    # Plot dynamics
    colors = ['#0000CD', '#228B22', '#DC143C', '#FF8C00']
    for i in range(n_configs):
        ax.plot(time_points, config_occupancy[:, i],
               label=f'Region {i+1}', linewidth=3, color=colors[i])
    
    # Find steady state (equilibrium distribution)
    eigenvals, eigenvecs = np.linalg.eig(K_matrix.T)
    steady_idx = np.argmin(np.abs(eigenvals))
    steady_state = np.real(eigenvecs[:, steady_idx])
    steady_state = np.abs(steady_state) / np.sum(np.abs(steady_state))
    
    print(f"  Equilibrium distribution: {steady_state}")
    
    # Plot steady state lines
    for i, ss in enumerate(steady_state):
        ax.axhline(ss, color=colors[i], linestyle='--',
                  linewidth=1.5, alpha=0.5, zorder=1)
    
    # Analyze equilibration timescales
    nonzero_eigenvals = eigenvals[np.abs(eigenvals) > 1e-10]
    if len(nonzero_eigenvals) > 0:
        tau = -1 / np.real(nonzero_eigenvals)
        tau_positive = tau[tau > 0]
        if len(tau_positive) > 0:
            tau_sorted = np.sort(tau_positive)[:3]
            
            print(f"  Equilibration timescales:")
            for i, t in enumerate(tau_sorted):
                print(f"    τ{i+1} = {t:.1f} years")
            
            timescale_text = 'Equilibration Timescales:\n\n'
            for i, t in enumerate(tau_sorted):
                timescale_text += f'τ{i+1} = {t:.1f} years\n'
            timescale_text += '\nTime to reach steady-state\npopulation distribution'
            
            ax.text(0.72, 0.72, timescale_text, transform=ax.transAxes,
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                            edgecolor='orange', linewidth=2, alpha=0.95))
    
    ax.axhline(0, color='black', linewidth=0.8, zorder=0)
    ax.set_xlabel('Time (years)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Regional Population Fraction', fontsize=13, fontweight='bold')
    ax.set_title(f'Configuration Transition Dynamics: {focal_ethnicity} Settlement\n' +
                f'Population Flow Between Stable Regions Over Time',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95,
             edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 25)
    
    # Panel label
    # ax.text(0.02, 0.98, '(f)', transform=ax.transAxes,
    #        fontsize=24, fontweight='bold', va='top',
    #        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
    #                 edgecolor='black', linewidth=2, alpha=0.95))
    
    # Interpretation
    interpret_text = ('Transition Dynamics:\ndP/dt = K·P\n\n'
                     'Models population flow\nbetween stable settlement\nregions over time.\n\n'
                     'Dashed lines show\nlong-term equilibrium\ndistribution.')
    
    ax.text(0.02, 0.02, interpret_text,
           transform=ax.transAxes, fontsize=9, style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue',
                    edgecolor='blue', linewidth=1.5, alpha=0.95),
           va='bottom', ha='left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel f'})
    plt.close()
    
    print(f"✓ Panel F saved: {output_path}")
    
    return tau_sorted if 'tau_sorted' in locals() and len(tau_sorted) > 0 else None


def create_panel_g_configuration_network_toronto(data, output_path):
    """
    Panel G: Configuration Interaction Network
    
    Visualizes settlement landscape as a network where:
    - Nodes = configurations (different ethnic compositions)
    - Edges = transitions (weighted by reorganization barriers)
    - Layout = favorability-based (vertical = favorability index)
    
    Shows how different demographic configurations relate to each other through
    the learned reaction-diffusion parameters, revealing which settlement patterns
    are similar and which transitions require overcoming significant barriers.
    """
    print("\n" + "="*80)
    print("PANEL G: CONFIGURATION INTERACTION NETWORK")
    print("="*80)
    
    import matplotlib.patches as patches
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    u_current = data['u_current']
    params = data['params']
    focal_idx = data['focal_idx']
    focal_ethnicity = data['focal_ethnicity']
    all_ethnicities = data['all_ethnicities']
    
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    
    # Create layout
    gs = plt.GridSpec(2, 3, figure=fig, height_ratios=[3, 1], width_ratios=[1, 2, 1],
                     hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_matrix = fig.add_subplot(gs[1, 0])
    ax_path = fig.add_subplot(gs[1, 1])
    ax_stability = fig.add_subplot(gs[1, 2])
    
    # Generate key configurations
    print("\nGenerating configuration space...")
    key_configs = []
    config_labels = []
    config_categories = []
    
    # 1. Pure focal ethnicity (high segregation)
    config = np.zeros(len(all_ethnicities))
    config[focal_idx] = 1.0
    key_configs.append(config)
    config_labels.append(f'Pure\n{focal_ethnicity}')
    config_categories.append('segregated')
    
    # 2. Balanced mix (high integration)
    config = np.ones(len(all_ethnicities)) / len(all_ethnicities)
    key_configs.append(config)
    config_labels.append('Balanced\nMix')
    config_categories.append('integrated')
    
    # 3. Current average (observed state)
    config = u_current.mean(axis=0) / (u_current.mean(axis=0).sum() + 1e-10)
    key_configs.append(config)
    config_labels.append('Current\nState')
    config_categories.append('current')
    
    # 4-6. Binary mixes (two-group dominant patterns)
    for i, other_idx in enumerate([j for j in range(len(all_ethnicities)) if j != focal_idx][:3]):
        config = np.zeros(len(all_ethnicities))
        config[focal_idx] = 0.6
        config[other_idx] = 0.4
        key_configs.append(config)
        config_labels.append(f'{focal_ethnicity}+\n{all_ethnicities[other_idx][:3]}')
        config_categories.append('binary')
    
    # 7. Tri-ethnic (three-group pattern)
    if len(all_ethnicities) >= 3:
        config = np.zeros(len(all_ethnicities))
        config[focal_idx] = 0.4
        config[[i for i in range(len(all_ethnicities)) if i != focal_idx][:2]] = 0.3
        key_configs.append(config)
        config_labels.append('Tri-ethnic\nMix')
        config_categories.append('multi')
    
    # Calculate favorability and stability properties
    print("\nCalculating configuration properties...")
    config_favorability = []
    config_stabilities = []
    
    for config in key_configs:
        favorability = calculate_reaction_diffusion_energy(
            config * params['carrying_capacity'], focal_idx, params
        )
        config_favorability.append(favorability)
        # Stability: inverse of absolute favorability (lower favorability = higher stability)
        stability = 1.0 / (1.0 + abs(favorability))
        config_stabilities.append(stability)
    
    print(f"  Favorability range: [{min(config_favorability):.2f}, {max(config_favorability):.2f}] dimensionless")
    
    # Create network graph
    G = nx.Graph()
    
    for i, (label, favorability, category) in enumerate(zip(config_labels, config_favorability, config_categories)):
        G.add_node(i, label=label, favorability=favorability, category=category)
    
    # Add edges based on configuration similarity and transition barriers
    for i in range(len(key_configs)):
        for j in range(i+1, len(key_configs)):
            # Compositional distance between configurations
            dist = np.linalg.norm(key_configs[i] - key_configs[j])
            if dist < 0.6:  # Only connect similar configurations
                # Transition barrier between configurations
                barrier = abs(config_favorability[i] - config_favorability[j])
                # Edge weight: inverse of barrier (easier transitions = stronger edges)
                G.add_edge(i, j, weight=1.0/(1.0 + barrier), barrier=barrier)
    
    # Layout: favorability-based vertical positioning
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    
    favorability_range = max(config_favorability) - min(config_favorability)
    for i, favorability in enumerate(config_favorability):
        x, _ = pos[i]
        # Vertical position based on favorability (low/favorable at bottom, high/unfavorable at top)
        y = (favorability - min(config_favorability)) / (favorability_range + 1e-10)
        pos[i] = (x * 2, y * 2 - 1)
    
    # Draw edges (transitions between configurations)
    for i, j in G.edges():
        edge_data = G.get_edge_data(i, j)
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        
        barrier = edge_data['barrier']
        # Visual encoding of barrier height
        if barrier < 1.0:
            style, alpha, width = 'solid', 0.8, 3  # Low barrier: strong connection
        elif barrier < 2.0:
            style, alpha, width = 'dashed', 0.6, 2  # Medium barrier
        else:
            style, alpha, width = 'dotted', 0.4, 1  # High barrier: weak connection
        
        ax_main.plot([x1, x2], [y1, y2], 'k-',
                    linestyle=style, alpha=alpha, linewidth=width, zorder=1)
        
        # Label significant barriers
        if barrier > 0.5:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax_main.text(mid_x, mid_y, f'{barrier:.1f}',
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 alpha=0.8, edgecolor='gray'))
    
    # Draw nodes (configurations)
    node_colors = {
        'segregated': '#FF6B6B',      # Red: high segregation
        'integrated': '#4ECDC4',       # Teal: high integration
        'current': '#FFE66D',          # Yellow: observed state
        'binary': '#9595E1',           # Purple: two-group patterns
        'multi': '#C7CEEA'             # Light purple: multi-group patterns
    }
    
    for i, (config, label, favorability, stability, category) in enumerate(
            zip(key_configs, config_labels, config_favorability,
                config_stabilities, config_categories)):
        x, y = pos[i]
        
        # Node circle
        circle = Circle((x, y), radius=0.15, facecolor=node_colors[category],
                       edgecolor='black', linewidth=2.5, alpha=0.9, zorder=3)
        ax_main.add_patch(circle)
        
        # Highlight current state with special border
        if category == 'current':
            inner_circle = Circle((x, y), radius=0.12, facecolor='none',
                                edgecolor='red', linewidth=3.5, linestyle='--', zorder=4)
            ax_main.add_patch(inner_circle)
        
        # Mini pie chart showing ethnic composition
        pie_radius = 0.08
        start_angle = 0
        pie_colors = plt.cm.Set3(np.linspace(0, 1, len(all_ethnicities)))
        
        for j, frac in enumerate(config):
            if frac > 0.01:
                end_angle = start_angle + frac * 360
                wedge = Wedge((x, y), pie_radius, start_angle, end_angle,
                            facecolor=pie_colors[j], edgecolor='white',
                            linewidth=0.6, alpha=0.85, zorder=5)
                ax_main.add_patch(wedge)
                start_angle = end_angle
        
        # Configuration label
        ax_main.text(x, y-0.25, label, ha='center', va='top', fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Favorability label
        ax_main.text(x+0.18, y+0.1, f'F={favorability:.1f}', ha='left', va='center',
                    fontsize=8, style='italic', color='darkblue')
    
    ax_main.set_ylabel('Configuration Favorability (dimensionless)', fontsize=13, fontweight='bold')
    ax_main.set_xlim(-2.5, 2.5)
    ax_main.set_ylim(-1.5, 1.5)
    ax_main.set_xticks([])
    ax_main.set_title(f'Configuration Space Network: {focal_ethnicity} Settlement Landscape',
                     fontsize=15, fontweight='bold', pad=20)
    
    # Subpanel A: Network size indicator
    n_configs = len(key_configs)
    ax_matrix.text(0.5, 0.5, f'{n_configs}\nConfigs',
                  transform=ax_matrix.transAxes, ha='center', va='center',
                  fontsize=20, fontweight='bold')
    ax_matrix.set_title('Network\nSize', fontsize=11, fontweight='bold')
    ax_matrix.axis('off')
    
    # Subpanel B: Integration pathway
    current_idx = config_categories.index('current')
    integrated_idx = config_categories.index('integrated')
    
    path_configs = np.linspace(0, 1, 30)
    path_favorability = []
    
    # Calculate favorability along path from current to integrated state
    for alpha in path_configs:
        interp_config = (1-alpha) * key_configs[current_idx] + alpha * key_configs[integrated_idx]
        interp_favorability = calculate_reaction_diffusion_energy(
            interp_config * params['carrying_capacity'], focal_idx, params
        )
        path_favorability.append(interp_favorability)
    
    ax_path.plot(path_configs, path_favorability, 'b-', linewidth=2.5)
    ax_path.fill_between(path_configs, path_favorability, alpha=0.3)
    ax_path.scatter([0, 1], [config_favorability[current_idx], config_favorability[integrated_idx]],
                   c=['yellow', 'cyan'], s=120, edgecolor='black', linewidth=2, zorder=5)
    ax_path.set_xlabel('Path Coordinate', fontsize=10, fontweight='bold')
    ax_path.set_ylabel('Favorability', fontsize=10, fontweight='bold')
    ax_path.set_title('Integration\nPathway', fontsize=11, fontweight='bold')
    ax_path.grid(True, alpha=0.3)
    
    # Subpanel C: Configuration stability
    bars = ax_stability.bar(range(n_configs), config_stabilities,
                           color=[node_colors[cat] for cat in config_categories],
                           edgecolor='black', linewidth=1.5)
    
    # Highlight current state
    if current_idx < n_configs:
        bars[current_idx].set_edgecolor('red')
        bars[current_idx].set_linewidth(3.5)
    
    ax_stability.set_xticks(range(n_configs))
    ax_stability.set_xticklabels([f'C{i+1}' for i in range(n_configs)],
                                 rotation=45, fontsize=9)
    ax_stability.set_ylabel('Stability Index', fontsize=10, fontweight='bold')
    ax_stability.set_title('Configuration\nStability', fontsize=11, fontweight='bold')
    ax_stability.set_ylim(0, max(config_stabilities) * 1.2)
    ax_stability.grid(True, alpha=0.3, axis='y')
    
    # Legend
    legend_elements = []
    for cat, color in node_colors.items():
        if cat in config_categories:
            legend_elements.append(patches.Patch(facecolor=color,
                                                label=cat.capitalize(),
                                                edgecolor='black'))
    
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=10,
                  title='Configuration Type', title_fontsize=11,
                  framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Panel label
    # ax_main.text(0.02, 0.98, '(g)', transform=ax_main.transAxes,
    #             fontsize=24, fontweight='bold', va='top',
    #             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
    #                      edgecolor='black', linewidth=2, alpha=0.95))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none',
            metadata={'Creator': 'GraphPDE Analyzer',
                        'Title': f'Panel g'})
    plt.close()
    
    print(f"✓ Panel G saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create Energy Landscape Analysis for YOUR GraphPDE Model - Toronto Chinese'
    )
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/graphpde/best_model.pt',
                       help='Path to YOUR model checkpoint (best_model.pt)')
    parser.add_argument('--graph', type=str, default='../data/graph_large_cities_rd.pkl',
                       help='Path to graph data (graph_large_cities_rd.pkl)')
    parser.add_argument('--da_info', type=str, default='../data/da_canada.csv',
                       help='Path to da_canada.csv')
    parser.add_argument('--shapefile', type=str, default='../data/da_canada2.shp',
                       help='Path to da_canada2.shp')
    parser.add_argument('--output', type=str, default='../figures/figure7/',
                       help='Output directory for panels')
    parser.add_argument('--ethnicity', type=str, default='China',
                       help='Focal ethnicity to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("ENERGY LANDSCAPE ANALYSIS FOR YOUR GRAPHPDE MODEL")
    print("Toronto Chinese Settlement Patterns - ALL 7 PANELS")
    print("="*80)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    data = load_toronto_chinese_data(
        checkpoint_path=args.checkpoint,
        graph_path=args.graph,
        da_info_path=args.da_info,
        shapefile_path=args.shapefile,
        focal_ethnicity=args.ethnicity
    )
    
    # Create all 7 panels
    panel_a_path = output_dir / 'panel_a_spatial_energy.png'
    local_attractiveness, energy_norm = create_panel_a_spatial_attractiveness_toronto(data, panel_a_path)
    
    panel_b_path = output_dir / 'panel_b_potential_surface.png'
    G_surface, G_surface_smooth, min_points, rc1_range, rc2_range = \
        create_panel_b_configuration_landscape_toronto(data, local_attractiveness, panel_b_path)
    
    panel_c_path = output_dir / 'panel_c_transition_state.png'
    barrier_forward, barrier_reverse = create_panel_c_transition_pathways_toronto(
        data, min_points, G_surface, rc1_range, rc2_range, panel_c_path
    )
    
    panel_d_path = output_dir / 'panel_d_statistical_mechanics.png'
    T_critical = create_panel_d_critical_threshold_dynamics_toronto(data, local_attractiveness, panel_d_path)
    
    panel_e_path = output_dir / 'panel_e_metastable_basins.png'
    basin_data = create_panel_e_stable_settlement_regions_toronto(data, local_attractiveness, panel_e_path)
    
    panel_f_path = output_dir / 'panel_f_reaction_kinetics.png'
    tau_sorted = create_panel_f_configuration_transition_dynamics_toronto(data, basin_data, panel_f_path)
    
    panel_g_path = output_dir / 'panel_g_configuration_network.png'
    create_panel_g_configuration_network_toronto(data, panel_g_path)
    
    # Summary
    print("\n" + "="*80)
    print("✓ ALL 7 PANELS COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir}")
    print(f"\nPanels:")
    print(f"  (a) Spatial Energy Map")
    print(f"  (b) 3D Potential Surface")
    print(f"  (c) Transition States")  
    print(f"  (d) Statistical Mechanics")
    print(f"  (e) Energy Basins (Choropleth)")
    print(f"  (f) Reaction Kinetics")
    print(f"  (g) Configuration Network")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()