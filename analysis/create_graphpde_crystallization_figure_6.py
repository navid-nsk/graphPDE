"""
Figure 6: Settlement Crystallization Dynamics

Generates Figure 6 showing crystallization patterns in Toronto.
Outputs saved to: ./figures/figure6/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LogNorm
from scipy.spatial import Voronoi, distance_matrix
from scipy.ndimage import gaussian_filter, label as scipy_label
from scipy.interpolate import griddata, Rbf
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import geopandas as gpd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


# Enhanced color schemes
CRYSTAL_CMAP = LinearSegmentedColormap.from_list('crystal',
    [(0, '#000033'), (0.2, '#000055'), (0.4, '#0000BB'), 
     (0.6, '#5555FF'), (0.8, '#AAAAFF'), (1.0, '#FFFFFF')])

TEMPORAL_CMAP = LinearSegmentedColormap.from_list('temporal',
    [(0, '#440154'), (0.25, '#31688E'), (0.5, '#35B779'), 
     (0.75, '#FDE725'), (1.0, '#FFFFFF')])

PLASMA_ENHANCED = LinearSegmentedColormap.from_list('plasma_enhanced',
    [(0, '#0d0887'), (0.3, '#7e03a8'), (0.5, '#cc4778'), 
     (0.7, '#f89540'), (0.9, '#f0f921'), (1.0, '#fcffa4')])


class GraphPDECrystallizationVisualizer:
    """
    Visualizer for GraphPDE settlement crystallization patterns.
    """
    
    def __init__(self, graph_path, shapefile_path, da_info_path):
        """
        Args:
            graph_path: Path to graph_large_cities_rd.pkl
            shapefile_path: Path to da_canada2.shp (EPSG:3347)
            da_info_path: Path to da_canada.csv
        """
        print("Loading data for crystallization visualization...")
        
        # Load graph
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load shapefile
        self.gdf = gpd.read_file(shapefile_path)
        self.gdf['DAUID'] = self.gdf['DAUID'].astype(str).str.strip()
        
        # Load DA info
        self.da_info = pd.read_csv(da_info_path)
        self.da_info['DAUID'] = self.da_info['DAUID'].astype(str).str.strip()
        
        # Get node features from graph
        self.dauids = self.graph['node_features']['dauid']
        self.cities = self.graph['node_features']['city']
        
        print(f"  Loaded {len(self.dauids)} DAUIDs")
        print(f"  Loaded {len(self.gdf)} geometries from shapefile")
        print(f"  Graph adjacency: {self.graph['adjacency'].shape}")
    
    def get_toronto_data(self, focal_ethnicity='China'):
        """
        Extract Toronto-specific data for all census years.
        
        Returns:
            Dictionary with:
                - dauids_toronto: Array of Toronto DAUIDs
                - coords_projected: (n, 2) array in EPSG:3347 meters
                - coords_normalized: (n, 2) array in [0, 1]
                - geometries: GeoDataFrame of Toronto DA geometries
                - temporal_data: Dict[year -> population array]
        """
        print(f"\nExtracting Toronto data for {focal_ethnicity}...")
        
        # Filter to Toronto
        toronto_mask = np.array([city == 'Toronto' for city in self.cities])
        dauids_toronto = self.dauids[toronto_mask]
        
        print(f"  Toronto DAUIDs: {len(dauids_toronto)}")
        
        # Get geometries
        toronto_gdf = self.gdf[self.gdf['DAUID'].isin([str(d) for d in dauids_toronto])].copy()
        
        # Ensure correct CRS
        if toronto_gdf.crs != 'EPSG:3347':
            toronto_gdf = toronto_gdf.to_crs('EPSG:3347')
        
        print(f"  Matched {len(toronto_gdf)} geometries")
        
        # Get centroids in projected coordinates (meters)
        centroids = toronto_gdf.geometry.centroid
        
        # Create coordinate arrays
        coords_proj = np.zeros((len(dauids_toronto), 2))
        dauid_to_idx = {str(d): i for i, d in enumerate(dauids_toronto)}
        
        for idx, row in toronto_gdf.iterrows():
            dauid_str = row['DAUID']
            if dauid_str in dauid_to_idx:
                graph_idx = dauid_to_idx[dauid_str]
                coords_proj[graph_idx, 0] = centroids.loc[idx].x
                coords_proj[graph_idx, 1] = centroids.loc[idx].y
        
        # Normalize to [0, 1] for some visualizations
        coords_norm = np.zeros_like(coords_proj)
        coords_norm[:, 0] = (coords_proj[:, 0] - coords_proj[:, 0].min()) / (coords_proj[:, 0].max() - coords_proj[:, 0].min())
        coords_norm[:, 1] = (coords_proj[:, 1] - coords_proj[:, 1].min()) / (coords_proj[:, 1].max() - coords_proj[:, 1].min())
        
        # Get extent
        bounds = toronto_gdf.total_bounds
        x_span = (bounds[2] - bounds[0]) / 1000  # km
        y_span = (bounds[3] - bounds[1]) / 1000  # km
        
        print(f"  Toronto extent: {x_span:.1f} km × {y_span:.1f} km")
        
        # Extract temporal population data
        temporal_immigration = self.graph['temporal_data']['immigration']
        years = sorted(temporal_immigration.keys())
        
        # Map ethnicity to column
        ethnic_cols_map = {
            'China': 'dim_405', 'Philippines': 'dim_410', 'India': 'dim_407',
            'Pakistan': 'dim_419', 'Iran': 'dim_421', 'Sri Lanka': 'dim_417',
            'Portugal': 'dim_413', 'Italy': 'dim_406', 'United Kingdom': 'dim_404'
        }
        
        if focal_ethnicity not in ethnic_cols_map:
            raise ValueError(f"Unknown ethnicity: {focal_ethnicity}")
        
        col_name = ethnic_cols_map[focal_ethnicity]
        
        temporal_data = {}
        for year in years:
            df_year = temporal_immigration[year].copy()
            df_year['DAUID'] = df_year['DAUID'].astype(str).str.strip()
            
            # Initialize array
            pop_array = np.zeros(len(dauids_toronto))
            
            # Fill in values
            for i, dauid in enumerate(dauids_toronto):
                dauid_str = str(dauid)
                if dauid_str in df_year['DAUID'].values:
                    row = df_year[df_year['DAUID'] == dauid_str].iloc[0]
                    if col_name in row:
                        pop_array[i] = float(row[col_name])
            
            temporal_data[year] = pop_array
            print(f"  Year {year}: {pop_array.sum():.0f} total, {np.sum(pop_array > 0)} non-zero DAs")
        
        return {
            'dauids': dauids_toronto,
            'coords_projected': coords_proj,
            'coords_normalized': coords_norm,
            'geometries': toronto_gdf,
            'temporal_data': temporal_data,
            'bounds': bounds,
            'extent_km': (x_span, y_span),
            'toronto_mask': toronto_mask
        }
    
    def create_visualization(self, focal_ethnicity='China', output_dir='./figures'):
        """
        Create comprehensive crystallization visualization.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get Toronto data
        toronto_data = self.get_toronto_data(focal_ethnicity)
        
        coords = toronto_data['coords_normalized']
        coords_proj = toronto_data['coords_projected']
        temporal_data = toronto_data['temporal_data']
        geometries = toronto_data['geometries']
        
        years = sorted(temporal_data.keys())
        concentrations = [temporal_data[year] for year in years]
        
        # Create figure
        fig = plt.figure(figsize=(32, 20), facecolor='#0A0A0A')
        
        # Title
        fig.text(0.5, 0.97, 
                f'Settlement Crystallization Dynamics: {focal_ethnicity} in Toronto', 
                color='white', fontsize=32, ha='center', fontweight='300')
        fig.text(0.5, 0.95, 
                'Multi-Scale Analysis of Ethnic Enclave Formation (2001-2021)', 
                color='#CCCCCC', fontsize=20, ha='center', style='italic')
        
        # ===================================================================
        # PANEL A: MACRO VIEW - Full Toronto with Real Geography
        # ===================================================================
        print("\n" + "="*80)
        print("Creating Panel A: Macro View")
        print("="*80)
        
        ax_macro = fig.add_axes([0.02, 0.50, 0.25, 0.40], facecolor='#1A1A1A')
        
        # Sum concentration across all years
        conc_all_years = np.sum(concentrations, axis=0)
        mask_populated = conc_all_years > 0
        
        print(f"  Total {focal_ethnicity} population (all years): {conc_all_years.sum():.0f}")
        print(f"  DAs with population: {np.sum(mask_populated)}")
        
        # Plot Toronto boundary (faint)
        geometries.boundary.plot(ax=ax_macro, color='#444444', linewidth=0.3, alpha=0.3)
        
        # Plot concentration with logarithmic scale
        if np.sum(mask_populated) > 0:
            scatter = ax_macro.scatter(
                coords[mask_populated, 0], 
                coords[mask_populated, 1],
                c=np.log1p(conc_all_years[mask_populated]),
                s=8, 
                cmap=PLASMA_ENHANCED, 
                alpha=0.9,
                vmin=0, 
                vmax=np.log1p(np.percentile(conc_all_years[mask_populated], 95)),
                edgecolors='none'
            )
        
        # Identify major settlement regions using DBSCAN-like clustering
        if np.sum(mask_populated) > 50:
            from sklearn.cluster import DBSCAN
            
            # Use high-concentration areas
            high_threshold = np.percentile(conc_all_years[mask_populated], 70)
            high_mask = conc_all_years > high_threshold
            high_coords = coords[high_mask]
            high_conc = conc_all_years[high_mask]
            
            if len(high_coords) > 20:
                # Cluster in normalized space
                clustering = DBSCAN(eps=0.03, min_samples=5).fit(high_coords)
                labels = set(clustering.labels_)
                labels.discard(-1)
                
                print(f"  Found {len(labels)} major settlement clusters")
                
                # Label top clusters
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7DC6F', '#95E1D3']
                cluster_info = []
                
                for label in labels:
                    cluster_mask = clustering.labels_ == label
                    cluster_pop = high_conc[cluster_mask].sum()
                    cluster_coords = high_coords[cluster_mask]
                    
                    # Weighted centroid
                    centroid = np.average(cluster_coords, weights=high_conc[cluster_mask], axis=0)
                    
                    cluster_info.append({
                        'label': label,
                        'population': cluster_pop,
                        'centroid': centroid,
                        'n_das': cluster_mask.sum()
                    })
                
                # Sort by population
                cluster_info.sort(key=lambda x: x['population'], reverse=True)
                
                # Label top 5 clusters
                for i, cluster in enumerate(cluster_info[:5]):
                    ax_macro.text(
                        cluster['centroid'][0], cluster['centroid'][1],
                        f"C{i+1}",
                        color=colors[i % len(colors)],
                        fontsize=11,
                        ha='center', va='center',
                        weight='bold',
                        bbox=dict(
                            boxstyle='circle,pad=0.3',
                            facecolor='black',
                            edgecolor=colors[i % len(colors)],
                            linewidth=2,
                            alpha=0.9
                        )
                    )
                
                # Summary text
                ax_macro.text(
                    0.02, 0.98,
                    f'{len(labels)} major clusters\n{np.sum(mask_populated)} DAs with population',
                    transform=ax_macro.transAxes,
                    color='white',
                    fontsize=9,
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
                )
        
        ax_macro.set_xlim(0, 1)
        ax_macro.set_ylim(0, 1)
        ax_macro.set_aspect('equal')
        ax_macro.set_title('a Macro View: Settlement Distribution', color='white', fontsize=16)
        ax_macro.set_xticks([])
        ax_macro.set_yticks([])
        
        # Scale bar
        ax_macro.plot([0.7, 0.9], [0.05, 0.05], 'w-', linewidth=3)
        ax_macro.text(0.8, 0.02, '10 km', color='white', ha='center', fontsize=12)
        
        # ===================================================================
        # PANEL B: MESO VIEW - Temporal Evolution
        # ===================================================================
        print("\n" + "="*80)
        print("Creating Panel B: Meso View")
        print("="*80)
        
        ax_meso = fig.add_axes([0.30, 0.35, 0.40, 0.55], facecolor='black')
        
        # Focus on highest concentration region
        sig_mask = conc_all_years > np.percentile(conc_all_years[conc_all_years > 0], 75)
        
        if np.sum(sig_mask) > 20:
            sig_coords = coords[sig_mask]
            sig_conc = conc_all_years[sig_mask]
            
            # Weighted center
            center_x = np.average(sig_coords[:, 0], weights=sig_conc)
            center_y = np.average(sig_coords[:, 1], weights=sig_conc)
            
            # Standard deviation for zoom
            std_x = np.sqrt(np.average((sig_coords[:, 0] - center_x)**2, weights=sig_conc))
            std_y = np.sqrt(np.average((sig_coords[:, 1] - center_y)**2, weights=sig_conc))
            
            zoom_factor = 2.5
            x_min = np.clip(center_x - zoom_factor * std_x, 0, 1)
            x_max = np.clip(center_x + zoom_factor * std_x, 0, 1)
            y_min = np.clip(center_y - zoom_factor * std_y, 0, 1)
            y_max = np.clip(center_y + zoom_factor * std_y, 0, 1)
            
            # Make square
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range > y_range:
                y_center = (y_min + y_max) / 2
                y_min = y_center - x_range / 2
                y_max = y_center + x_range / 2
            else:
                x_center = (x_min + x_max) / 2
                x_min = x_center - y_range / 2
                x_max = x_center + y_range / 2
            
            x_min = np.clip(x_min, 0, 1)
            x_max = np.clip(x_max, 0, 1)
            y_min = np.clip(y_min, 0, 1)
            y_max = np.clip(y_max, 0, 1)
        else:
            x_min, x_max, y_min, y_max = 0.3, 0.7, 0.3, 0.7
        
        print(f"  Meso zoom: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        
        # Add zoom indicator on macro view
        zoom_rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor='yellow', linewidth=2, linestyle='--', alpha=0.8
        )
        ax_macro.add_patch(zoom_rect)
        
        # Extract region data
        region_mask = ((coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                       (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max))
        
        print(f"  DAs in meso region: {np.sum(region_mask)}")
        
        # High-resolution interpolation
        resolution = 500
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate 2021 data as base layer
        conc_2021 = concentrations[-1]
        valid_points = region_mask & (conc_2021 > 0)
        
        if np.sum(valid_points) > 10:
            Zi = griddata(
                coords[valid_points], 
                conc_2021[valid_points],
                (Xi, Yi), 
                method='cubic', 
                fill_value=0
            )
            Zi = gaussian_filter(Zi, sigma=2.5)
            
            # Display with power norm
            norm = PowerNorm(gamma=0.6, vmin=0, vmax=np.percentile(Zi[Zi > 0], 95) if np.any(Zi > 0) else 1)
            im = ax_meso.imshow(
                Zi, 
                extent=[x_min, x_max, y_min, y_max],
                cmap=CRYSTAL_CMAP, 
                norm=norm, 
                origin='lower',
                interpolation='bilinear', 
                alpha=0.9
            )
            
            # Overlay temporal contours
            year_colors = ['#8B0000', '#FF6B6B', '#FFA500', '#FFD700', '#ADFF2F', '#00FA9A']
            
            for i, (year, conc) in enumerate(zip(years, concentrations)):
                valid_year = region_mask & (conc > 0)
                
                if np.sum(valid_year) > 10:
                    Zi_year = griddata(
                        coords[valid_year], 
                        conc[valid_year],
                        (Xi, Yi), 
                        method='cubic', 
                        fill_value=0
                    )
                    Zi_year = gaussian_filter(Zi_year, sigma=2.5)
                    
                    # Contour at 60th percentile
                    if np.sum(Zi_year > 0) > 100:
                        level = np.percentile(Zi_year[Zi_year > 0], 60)
                        contour = ax_meso.contour(
                            Xi, Yi, Zi_year, 
                            levels=[level],
                            colors=[year_colors[i]], 
                            linewidths=2.5, 
                            alpha=0.9
                        )
                        
                        # Label contours
                        if contour.collections and len(contour.collections[0].get_paths()) > 0:
                            try:
                                path = contour.collections[0].get_paths()[0]
                                v = path.vertices
                                if len(v) > 10:
                                    label_idx = len(v) // 4
                                    ax_meso.text(
                                        v[label_idx, 0], v[label_idx, 1],
                                        str(year),
                                        color=year_colors[i],
                                        fontsize=13,
                                        fontweight='bold',
                                        bbox=dict(
                                            boxstyle='round,pad=0.3',
                                            facecolor='black',
                                            alpha=0.8,
                                            edgecolor=year_colors[i],
                                            linewidth=1.5
                                        )
                                    )
                            except:
                                pass
        
        ax_meso.set_xlim(x_min, x_max)
        ax_meso.set_ylim(y_min, y_max)
        ax_meso.set_aspect('equal')
        ax_meso.axis('off')
        ax_meso.set_title(
            'b Meso View: Temporal Evolution of Settlement Pattern',
            color='white', fontsize=18, pad=20
        )
        
        # Scale bar
        scale_length = (x_max - x_min) * 0.2
        scale_km = scale_length * toronto_data['extent_km'][0]
        ax_meso.plot(
            [x_max - scale_length - 0.02, x_max - 0.02],
            [y_min + 0.02, y_min + 0.02],
            'w-', linewidth=4
        )
        ax_meso.text(
            x_max - scale_length/2 - 0.02, y_min + 0.01,
            f'{scale_km:.1f} km',
            color='white', ha='center', fontsize=14, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
        )
        
        # ===================================================================
        # PANEL C: MICRO VIEW - Voronoi Tessellation
        # ===================================================================
        print("\n" + "="*80)
        print("Creating Panel C: Micro View")
        print("="*80)
        
        ax_micro = fig.add_axes([0.73, 0.55, 0.24, 0.35], facecolor='black')
        
        # Focus on densest sub-region
        if np.sum(region_mask & (conc_2021 > 0)) > 20:
            region_conc = conc_2021[region_mask]
            region_conc_nonzero = region_conc[region_conc > 0]
            
            if len(region_conc_nonzero) > 10:
                high_threshold = np.percentile(region_conc_nonzero, 75)
                dense_mask = region_mask & (conc_2021 > high_threshold)
                
                if np.sum(dense_mask) > 5:
                    dense_coords = coords[dense_mask]
                    dense_conc = conc_2021[dense_mask]
                    
                    micro_center = np.average(dense_coords, weights=dense_conc, axis=0)
                    micro_size = 0.08  # 8% of space
                    
                    micro_x_min = max(x_min, micro_center[0] - micro_size/2)
                    micro_x_max = min(x_max, micro_center[0] + micro_size/2)
                    micro_y_min = max(y_min, micro_center[1] - micro_size/2)
                    micro_y_max = min(y_max, micro_center[1] + micro_size/2)
                    
                    # Zoom indicator on meso
                    micro_rect = Rectangle(
                        (micro_x_min, micro_y_min),
                        micro_x_max - micro_x_min,
                        micro_y_max - micro_y_min,
                        fill=False, edgecolor='cyan', linewidth=2.5,
                        linestyle='--', alpha=0.9
                    )
                    ax_meso.add_patch(micro_rect)
                    
                    # Get micro DAs
                    micro_threshold = np.percentile(conc_2021[conc_2021 > 0], 50)
                    micro_mask = ((coords[:, 0] >= micro_x_min) &
                                 (coords[:, 0] <= micro_x_max) &
                                 (coords[:, 1] >= micro_y_min) &
                                 (coords[:, 1] <= micro_y_max) &
                                 (conc_2021 > micro_threshold))
                    
                    print(f"  Micro DAs: {np.sum(micro_mask)}")
                    
                    if np.sum(micro_mask) >= 4:
                        micro_coords = coords[micro_mask]
                        micro_conc = conc_2021[micro_mask]
                        
                        # Voronoi tessellation
                        try:
                            vor = Voronoi(micro_coords)
                            
                            # Plot cells
                            for i, point in enumerate(micro_coords):
                                region_idx = vor.point_region[i]
                                if region_idx >= 0:
                                    region = vor.regions[region_idx]
                                    if -1 not in region and len(region) > 0:
                                        vertices = [vor.vertices[j] for j in region if j < len(vor.vertices)]
                                        if len(vertices) > 2:
                                            color_val = micro_conc[i] / (np.max(micro_conc) + 1e-10)
                                            poly = Polygon(
                                                vertices,
                                                facecolor=CRYSTAL_CMAP(color_val),
                                                edgecolor='white',
                                                linewidth=1.5,
                                                alpha=0.8
                                            )
                                            ax_micro.add_patch(poly)
                            
                            # DA centroids
                            scatter = ax_micro.scatter(
                                micro_coords[:, 0], micro_coords[:, 1],
                                c=micro_conc,
                                s=120,
                                cmap='hot',
                                edgecolors='white',
                                linewidths=2.5,
                                zorder=10,
                                alpha=1.0
                            )
                            
                            ax_micro.text(
                                0.5, 0.95,
                                f'{len(micro_coords)} DAs',
                                transform=ax_micro.transAxes,
                                color='white', fontsize=11, ha='center', va='top',
                                weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
                            )
                            
                        except Exception as e:
                            print(f"  Voronoi failed: {e}")
                            ax_micro.scatter(
                                micro_coords[:, 0], micro_coords[:, 1],
                                c=micro_conc, s=300, cmap='hot',
                                edgecolors='white', linewidths=3,
                                marker='h', alpha=0.9
                            )
                    
                    ax_micro.set_xlim(micro_x_min, micro_x_max)
                    ax_micro.set_ylim(micro_y_min, micro_y_max)
                    
                    # Scale bar
                    scale_len = (micro_x_max - micro_x_min) * 0.3
                    scale_km_micro = scale_len * toronto_data['extent_km'][0]
                    ax_micro.plot(
                        [micro_x_min + 0.1*scale_len, micro_x_min + 0.1*scale_len + scale_len],
                        [micro_y_min + 0.05*(micro_y_max-micro_y_min), micro_y_min + 0.05*(micro_y_max-micro_y_min)],
                        'w-', linewidth=3
                    )
                    ax_micro.text(
                        micro_x_min + 0.1*scale_len + scale_len/2,
                        micro_y_min + 0.02*(micro_y_max-micro_y_min),
                        f'{scale_km_micro:.1f} km',
                        color='white', ha='center', fontsize=11, weight='bold'
                    )
        
        ax_micro.set_aspect('equal')
        ax_micro.axis('off')
        ax_micro.set_title('c Micro View: Neighborhood Structure', color='white', fontsize=16)
        
        # ===================================================================
        # PANEL D: GROWTH KINETICS
        # ===================================================================
        print("\n" + "="*80)
        print("Creating Panel D: Growth Kinetics")
        print("="*80)
        
        ax_growth = fig.add_axes([0.30, 0.05, 0.40, 0.25])
        ax_growth.set_facecolor('white')
        
        # Analyze cluster growth
        n_clusters = []
        total_pop = []
        avg_cluster_size = []
        
        for year, conc in zip(years, concentrations):
            region_conc = conc[region_mask]
            
            if np.sum(region_conc > 0) > 10:
                threshold = np.percentile(region_conc[region_conc > 0], 50)
                binary_mask = region_conc > threshold
                
                # Connected components
                labeled, n = scipy_label(binary_mask.reshape(-1, 1))
                labeled = labeled.flatten()
                
                n_clusters.append(n)
                total_pop.append(np.sum(region_conc))
                
                if n > 0:
                    sizes = []
                    for i in range(1, n+1):
                        cluster_pop = np.sum(region_conc[labeled == i])
                        if cluster_pop > 0:
                            sizes.append(cluster_pop)
                    avg_cluster_size.append(np.mean(sizes) if sizes else 0)
                else:
                    avg_cluster_size.append(0)
            else:
                n_clusters.append(0)
                total_pop.append(0)
                avg_cluster_size.append(0)
        
        print(f"  Clusters over time: {n_clusters}")
        print(f"  Total population: {total_pop}")
        
        # Plot
        years_arr = np.array(years)
        
        ax1 = ax_growth
        color1 = '#1E88E5'
        ax1.plot(years_arr, n_clusters, 'o-', color=color1,
                linewidth=3.5, markersize=12, label='Number of Clusters',
                markeredgecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Year', fontsize=15, weight='bold', color='white')
        ax1.set_ylabel('Number of Settlement Clusters', color=color1, fontsize=14, weight='bold')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.tick_params(axis='x', labelsize=12, labelcolor='white')
        
        ax2 = ax1.twinx()
        color2 = '#D32F2F'
        ax2.plot(years_arr, avg_cluster_size, 's-', color=color2,
                linewidth=3.5, markersize=12, label='Avg Cluster Size',
                markeredgecolor='white', markeredgewidth=2)
        ax2.set_ylabel('Average Population per Cluster', color=color2, fontsize=14, weight='bold')
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
        
        ax1.set_title('d Nucleation and Growth Kinetics', fontsize=17, pad=10, weight='bold', color='white')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(years[0] - 1, years[-1] + 1)
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        # JMAK fit
        if len(years) > 2 and np.max(total_pop) > 0:
            X = np.array(total_pop) / np.max(total_pop)
            t = years_arr - years_arr[0]
            
            try:
                def avrami(t, k, n):
                    return 1 - np.exp(-(k * t)**n)
                
                popt, _ = curve_fit(avrami, t[t > 0], X[t > 0], p0=[0.1, 2], bounds=(0, [1, 5]))
                
                growth_type = ('Interface-controlled (n≈1)' if 0.8 < popt[1] < 1.2 else
                              '3D growth (n≈3)' if 2.5 < popt[1] < 3.5 else
                              '2D growth (n≈2)' if 1.5 < popt[1] < 2.5 else
                              f'Complex (n={popt[1]:.1f})')
                
                ax1.text(
                    0.05, 0.88,
                    f'JMAK Analysis:\nn = {popt[1]:.2f}\n{growth_type}',
                    transform=ax1.transAxes,
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.95, edgecolor='black', linewidth=1.5)
                )
            except:
                pass
        
        # ===================================================================
        # PANEL E: STRUCTURE FACTOR
        # ===================================================================
        print("\n" + "="*80)
        print("Creating Panel E: Structure Factor")
        print("="*80)
        
        ax_structure = fig.add_axes([0.75, 0.05, 0.24, 0.25])
        ax_structure.set_facecolor('white')
        
        structure_computed = False
        
        if np.sum(region_mask & (conc_2021 > 0)) > 30:
            settled_mask = region_mask & (conc_2021 > 0)
            settled_coords = coords[settled_mask]
            settled_conc = conc_2021[settled_mask]
            
            try:
                # Grid for settled area
                x_s_min, x_s_max = settled_coords[:, 0].min(), settled_coords[:, 0].max()
                y_s_min, y_s_max = settled_coords[:, 1].min(), settled_coords[:, 1].max()
                
                grid_size = 128
                xi_s = np.linspace(x_s_min, x_s_max, grid_size)
                yi_s = np.linspace(y_s_min, y_s_max, grid_size)
                Xi_s, Yi_s = np.meshgrid(xi_s, yi_s)
                
                Zi_s = griddata(settled_coords, settled_conc, (Xi_s, Yi_s), method='cubic', fill_value=0)
                
                # Window
                window = np.outer(np.hanning(grid_size), np.hanning(grid_size))
                Zi_windowed = Zi_s * window
                
                # FFT
                fft_2d = np.fft.fft2(Zi_windowed)
                fft_shifted = np.fft.fftshift(fft_2d)
                power_spectrum = np.abs(fft_shifted)**2
                
                # Radial average
                center = grid_size // 2
                y, x = np.ogrid[:grid_size, :grid_size]
                r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
                
                radial_prof = np.bincount(r.ravel(), power_spectrum.ravel()) / np.bincount(r.ravel())
                
                # Physical units
                physical_size = (x_s_max - x_s_min) * toronto_data['extent_km'][0]
                k_values = np.arange(len(radial_prof)) * 2 * np.pi / physical_size
                
                # Plot
                valid_k = (k_values > 0.05) & (k_values < 5)
                ax_structure.loglog(k_values[valid_k], radial_prof[valid_k],
                                   'b-', linewidth=3, label='S(k)', alpha=0.9)
                structure_computed = True
                
                # Find peaks
                peaks, _ = find_peaks(np.log10(radial_prof[1:30]+1e-10), prominence=0.5, distance=3)
                
                if len(peaks) > 0:
                    for peak in peaks[:3]:
                        k_peak = k_values[peak+1]
                        wavelength = 2 * np.pi / k_peak
                        ax_structure.axvline(k_peak, color='red', linestyle='--', linewidth=2, alpha=0.7)
                        ax_structure.text(k_peak*1.15, radial_prof[peak+1],
                                        f'λ={wavelength:.1f} km',
                                        fontsize=11, rotation=0, va='center', weight='bold')
                
                # Power law fit
                k_fit = k_values[10:40]
                S_fit = radial_prof[10:40]
                
                if len(k_fit) > 10 and np.all(S_fit > 0):
                    p = np.polyfit(np.log10(k_fit), np.log10(S_fit), 1)
                    ax_structure.plot(k_fit, 10**np.poly1d(p)(np.log10(k_fit)),
                                    'r--', linewidth=2.5, alpha=0.8,
                                    label=f'k^{{{p[0]:.1f}}}')
                    
                    fractal_dim = -p[0] / 2
                    ax_structure.text(
                        0.95, 0.88,
                        f'Fractal Dimension:\nD = {fractal_dim:.2f}',
                        transform=ax_structure.transAxes,
                        ha='right', va='top', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', alpha=0.95, edgecolor='black', linewidth=1.5)
                    )
            except Exception as e:
                print(f"  Structure factor error: {e}")
        
        if not structure_computed:
            ax_structure.text(0.5, 0.5,
                            'Insufficient data\nfor structure factor',
                            transform=ax_structure.transAxes,
                            ha='center', va='center',
                            fontsize=13, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))
        
        ax_structure.set_xlabel('Wavenumber k (km⁻¹)', fontsize=14, weight='bold', color='white')
        ax_structure.set_ylabel('Structure Factor S(k)', fontsize=14, weight='bold', color='white')
        ax_structure.tick_params(axis='y', labelcolor='white', labelsize=12)
        ax_structure.tick_params(axis='x', labelsize=12, labelcolor='white')
        ax_structure.set_title('e Spatial Correlations', fontsize=17, pad=10, weight='bold', color='white')
        ax_structure.legend(fontsize=12, loc='upper right')
        ax_structure.grid(True, alpha=0.3, which='both', linestyle='--')
        ax_structure.set_xlim(0.1, 5)
        
        # ===================================================================
        # PANEL F: PHASE DIAGRAM
        # ===================================================================
        ax_phase = fig.add_axes([0.75, 0.35, 0.12, 0.12], facecolor='white')
        
        T = np.linspace(0, 1, 100)
        c_critical = 0.5 - 0.5 * np.sqrt(np.clip(1 - T, 0, 1))
        c_critical2 = 0.5 + 0.5 * np.sqrt(np.clip(1 - T, 0, 1))
        
        ax_phase.fill_betweenx(T, c_critical, c_critical2, color='lightblue', alpha=0.4)
        ax_phase.plot(c_critical, T, 'b-', linewidth=2.5)
        ax_phase.plot(c_critical2, T, 'b-', linewidth=2.5)
        
        # Current state
        if total_pop and total_pop[-1] > 0:
            current_c = min(total_pop[-1] / (total_pop[-1] * 5 + 1e-10), 0.9)
        else:
            current_c = 0.3
        
        current_T = 0.4
        ax_phase.plot(current_c, current_T, 'r*', markersize=15, markeredgecolor='black', markeredgewidth=1.5)
        
        ax_phase.set_xlabel('Concentration', fontsize=11, weight='bold', color='white')
        ax_phase.set_ylabel('T/T_c', fontsize=11, weight='bold', color='white')
        ax_phase.tick_params(axis='y', labelcolor='white', labelsize=12)
        ax_phase.tick_params(axis='x', labelsize=12, labelcolor='white')
        ax_phase.set_title('f Phase Diagram', fontsize=13, pad=5, weight='bold', color='white')
        ax_phase.text(0.5, 0.12, 'Mixed', ha='center', fontsize=10, weight='bold')
        ax_phase.text(0.12, 0.65, 'Sep.', ha='center', fontsize=10, weight='bold')
        ax_phase.text(0.88, 0.65, 'Sep.', ha='center', fontsize=10, weight='bold')
        ax_phase.set_xlim(0, 1)
        ax_phase.set_ylim(0, 1)
        
        # ===================================================================
        # PANEL G: SUMMARY
        # ===================================================================
        ax_summary = fig.add_axes([0.02, 0.05, 0.25, 0.40], facecolor='#1A1A1A')
        ax_summary.axis('off')
        
        # Calculate metrics
        if len(n_clusters) > 0 and n_clusters[-1] > 0:
            if n_clusters[-1] > n_clusters[0]:
                pattern = "Nucleation & Growth"
            elif n_clusters[-1] < n_clusters[0]:
                pattern = "Coalescence"
            else:
                pattern = "Stable"
        else:
            pattern = "Emerging"
        
        # Moran's I
        morans_i = 0
        if np.sum(conc_2021 > 0) > 10:
            settled_mask = conc_2021 > 0
            settled_coords = coords[settled_mask]
            settled_conc = conc_2021[settled_mask]
            
            if len(settled_coords) > 10:
                W = 1 / (distance_matrix(settled_coords, settled_coords) + 1e-10)
                np.fill_diagonal(W, 0)
                W = W / W.sum()
                
                z = (settled_conc - settled_conc.mean()) / (settled_conc.std() + 1e-10)
                morans_i = len(z) * np.sum(W * np.outer(z, z)) / np.sum(z**2)
        
        total_clusters = n_clusters[-1] if n_clusters else 0
        avg_size = avg_cluster_size[-1] if avg_cluster_size else 0
        growth_rate = ((total_pop[-1]/total_pop[0])**(1/20)-1)*100 if (total_pop and total_pop[0] > 0) else 0
        nucleation_rate = (n_clusters[-1]-n_clusters[0])/20 if n_clusters else 0
        
        occupied_fraction = np.sum(conc_2021 > 0) / len(conc_2021)
        
        summary_text = f"""CRYSTALLIZATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━

Settlement Pattern: {pattern}

Spatial Statistics (2021):
• Clusters: {total_clusters}
• Avg Size: {avg_size:.0f}
• Moran's I: {morans_i:.3f}
• Type: {"Clustered" if morans_i > 0.3 else "Random" if morans_i > -0.3 else "Dispersed"}
• Coverage: {occupied_fraction:.1%}

Temporal Evolution:
• Growth: {growth_rate:.1f}%/year
• Nucleation: {nucleation_rate:.1f}/year

Physics Insights:
• Reaction-diffusion dynamics
• Self-organizing patterns
• Spatial autocorrelation
• Multi-scale structure"""
        
        ax_summary.text(
            0.05, 0.95,
            summary_text,
            transform=ax_summary.transAxes,
            color='white',
            fontsize=13,
            va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#2A2A2A', alpha=0.95, edgecolor='white', linewidth=2)
        )
        
        # Save
        output_file = output_dir / 'figure6.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#0A0A0A')
        pdf_path = Path(output_file).with_suffix('.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none',
                metadata={'Creator': 'GraphPDE Analyzer',
                            'Title': f'Crystall'})
        print(f"\n{'='*80}")
        print(f"Saved: {output_file}")
        print(f"{'='*80}\n")
        
        plt.close(fig)
        
        return {
            'n_clusters': n_clusters,
            'pattern_type': pattern,
            'morans_i': morans_i,
            'output_file': str(output_file)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create crystallization visualization for GraphPDE')
    parser.add_argument('--graph', type=str, default='../data/graph_large_cities_rd.pkl')
    parser.add_argument('--shapefile', type=str, default='../data/da_canada2.shp')
    parser.add_argument('--da-info', type=str, default='../data/da_canada.csv')
    parser.add_argument('--ethnicity', type=str, default='China')
    parser.add_argument('--output-dir', type=str, default='../figures/figure6')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GraphPDE Crystallization Visualization")
    print("="*80)
    
    visualizer = GraphPDECrystallizationVisualizer(
        graph_path=args.graph,
        shapefile_path=args.shapefile,
        da_info_path=args.da_info
    )
    
    results = visualizer.create_visualization(
        focal_ethnicity=args.ethnicity,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"Pattern type: {results['pattern_type']}")
    print(f"Moran's I: {results['morans_i']:.3f}")
    print(f"Final clusters: {results['n_clusters'][-1] if results['n_clusters'] else 0}")
    print(f"Output: {results['output_file']}")


if __name__ == "__main__":
    main()
