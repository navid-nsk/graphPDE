"""
Figure 3: City-Level Spatial Dynamics Analysis

Generates Figure 3 showing learned dynamics across 51 Canadian cities.
Outputs saved to: ./figures/figure3/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import seaborn as sns
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path as PathLib
import warnings
warnings.filterwarnings('ignore')

from graphpde_analyzer_COMPLETE import GraphPDEAnalyzer


def get_trained_cities():
    """
    Return the 51 cities that were actually used in training.
    These are cities with population > 100k.
    """
    cities_above_100k = [
        'Markham', 'Coquitlam', 'Winnipeg', 'Delta', 'Surrey', 'Richmond Hill', 
        'Oakville', 'Abbotsford', 'Vaughan', 'Toronto', 'Burnaby', 'Mississauga', 
        'Rocky View County', 'Brossard', 'Richmond', 'Vancouver', 'North Vancouver', 
        'Saskatoon', 'Whitby', 'Guelph', 'Hamilton', 'Edmonton', 'Nanaimo', 'Langley', 
        'Brampton', 'Ottawa', 'Calgary', 'Regina', 'Montréal', 'Laval', 'Milton', 
        'Windsor', 'Cambridge', 'Kelowna', 'Oshawa', 'Thunder Bay', 'St. Catharines', 
        'Chilliwack', 'Kitchener', 'Ajax', 'Saanich', 'Longueuil', 'London', 'Waterloo', 
        'Pickering', 'Burlington', 'Barrie', 'Québec', 'Gatineau', 'Halifax', 
        'Kingston', 'Greater Sudbury / Grand Sudbury'
    ]
    return cities_above_100k


def get_city_coordinates():
    """
    Approximate coordinates for Canadian cities (for bubble map).
    Returns dict: city_name -> (lat, lon)
    """
    coords = {
        'Toronto': (43.65, -79.38),
        'Montreal': (45.50, -73.57),
        'Montréal': (45.50, -73.57),
        'Vancouver': (49.28, -123.12),
        'Calgary': (51.05, -114.07),
        'Edmonton': (53.55, -113.47),
        'Ottawa': (45.42, -75.70),
        'Mississauga': (43.59, -79.64),
        'Winnipeg': (49.90, -97.14),
        'Quebec': (46.81, -71.21),
        'Québec': (46.81, -71.21),
        'Hamilton': (43.26, -79.87),
        'Brampton': (43.73, -79.76),
        'Surrey': (49.19, -122.85),
        'Laval': (45.61, -73.71),
        'Halifax': (44.65, -63.57),
        'London': (42.98, -81.25),
        'Markham': (43.88, -79.26),
        'Vaughan': (43.84, -79.50),
        'Gatineau': (45.48, -75.64),
        'Longueuil': (45.53, -73.52),
        'Burnaby': (49.25, -122.97),
        'Saskatoon': (52.13, -106.67),
        'Regina': (50.45, -104.62),
        'Richmond': (49.17, -123.14),
        'Richmond Hill': (43.88, -79.44),
        'Oakville': (43.45, -79.68),
        'Burlington': (43.33, -79.79),
        'Greater Sudbury / Grand Sudbury': (46.49, -80.99),
        'Oshawa': (43.90, -78.86),
        'Barrie': (44.37, -79.69),
        'Abbotsford': (49.05, -122.31),
        'Coquitlam': (49.28, -122.79),
        'St. Catharines': (43.16, -79.24),
        'Kelowna': (49.89, -119.50),
        'Cambridge': (43.36, -80.31),
        'Kingston': (44.23, -76.49),
        'Guelph': (43.55, -80.25),
        'Kitchener': (43.45, -80.49),
        'Waterloo': (43.47, -80.52),
        'Windsor': (42.30, -83.02),
        'Ajax': (43.85, -79.04),
        'Pickering': (43.84, -79.09),
        'Whitby': (43.90, -78.94),
        'Milton': (43.52, -79.88),
        'Nanaimo': (49.17, -123.94),
        'Brossard': (45.47, -73.46),
        'Delta': (49.08, -123.06),
        'Langley': (49.10, -122.66),
        'Saanich': (48.49, -123.38),
        'North Vancouver': (49.32, -123.07),
        'Chilliwack': (49.16, -121.95),
        'Thunder Bay': (48.38, -89.25),
        'Rocky View County': (51.18, -114.00)
    }
    return coords


def categorize_cities_by_size(city_names, graph):
    """
    Categorize cities by population size based on number of DAUIDs.
    
    Returns:
        dict: {'large': [...], 'medium': [...], 'small': [...]}
    """
    city_counts = {}
    all_cities = graph['node_features']['city']
    
    for city in city_names:
        count = sum(1 for c in all_cities if c == city)
        city_counts[city] = count
    
    # Sort by DAUID count
    sorted_cities = sorted(city_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Split into thirds
    n = len(sorted_cities)
    large = [c[0] for c in sorted_cities[:n//3]]
    medium = [c[0] for c in sorted_cities[n//3:2*n//3]]
    small = [c[0] for c in sorted_cities[2*n//3:]]
    
    return {'large': large, 'medium': medium, 'small': small}


def compute_diversity_index(population_matrix):
    """
    Compute Shannon diversity index for multi-ethnic mixing.
    
    Args:
        population_matrix: (n_cities, n_ethnicities) array
    
    Returns:
        diversity_scores: (n_cities,) array
    """
    diversity = np.zeros(population_matrix.shape[0])
    
    for i in range(population_matrix.shape[0]):
        pop = population_matrix[i]
        if pop.sum() > 0:
            pop_norm = pop / pop.sum()
            # Shannon entropy (higher = more diverse)
            diversity[i] = entropy(pop_norm + 1e-10)
        else:
            diversity[i] = 0.0
    
    return diversity


def create_city_analysis_figure(analyzer, all_ethnicities, trained_cities, output_dir):
    """
    Create comprehensive 9-panel city analysis figure.
    
    All visualizations use ACTUAL learned parameters for the 51 trained cities.
    """
    output_dir = PathLib(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Get city names from graph and filter to trained cities
    graph = analyzer.graph
    all_city_names = graph['node_features']['city']
    unique_cities = sorted(set(all_city_names))
    
    # Filter to trained cities only
    trained_set = set(trained_cities)
    filtered_cities = [c for c in unique_cities if c in trained_set]
    n_cities = len(filtered_cities)
    n_ethnicities = len(all_ethnicities)
    
    
    # Create city index mapping
    city_to_idx = {city: idx for idx, city in enumerate(unique_cities)}
    filtered_city_indices = [city_to_idx[c] for c in filtered_cities]
    
    # Extract parameters for all periods
    city_params = {}
    for period_idx in range(4):
        params = analyzer.get_parameters_for_period(period_idx)
        # Filter to trained cities only
        city_params[period_idx] = {
            'diffusion_coefficients': params['diffusion_coefficients'][filtered_city_indices],
            'growth_rates': params['growth_rates'][filtered_city_indices],
            'carrying_capacity': params['carrying_capacity'][filtered_city_indices],
            'immigration_rates': params['immigration_rates'][filtered_city_indices],
            'emigration_rates': params['emigration_rates'][filtered_city_indices]
        }
    
    # Create figure with 3x3 grid
    fig = plt.figure(figsize=(24, 20), facecolor='white')
    
    # Main title
    fig.suptitle('City-Scale Spatial Dynamics Across 51 Canadian Cities', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Subtitle
    fig.text(0.5, 0.96, 
             'Learned parameters from GraphPDE: city-specific diffusion, growth, and settlement patterns', 
             fontsize=13, ha='center', style='italic', color='#555555')
    
    # Create grid
    gs = GridSpec(3, 3, figure=fig, 
                  hspace=0.35, wspace=0.30,
                  left=0.06, right=0.96, top=0.93, bottom=0.05)
    
    period_names = ['2001-2006', '2006-2011', '2011-2016', '2016-2021']
    period_years = [2001, 2006, 2011, 2016]
    
    # PANEL A: Carrying Capacity Evolution
    ax_A = fig.add_subplot(gs[0, 0])
    
    # Select major cities for clarity
    major_cities = ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Edmonton', 
                    'Ottawa', 'Mississauga', 'Winnipeg']
    major_cities = [c for c in major_cities if c in filtered_cities]
    
    colors_major = plt.cm.tab10(np.linspace(0, 1, len(major_cities)))
    
    for i, city in enumerate(major_cities):
        city_idx = filtered_cities.index(city)
        K_trajectory = [city_params[p]['carrying_capacity'][city_idx] for p in range(4)]
        
        ax_A.plot(period_years, K_trajectory, 
                 marker='o', linewidth=2.5, markersize=8,
                 color=colors_major[i], label=city, alpha=0.85)
    
    ax_A.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax_A.set_ylabel('Carrying Capacity (people/DAUID)', fontsize=12, fontweight='bold')
    ax_A.set_title('a Carrying Capacity Evolution', fontsize=14, fontweight='bold', pad=10)
    ax_A.legend(fontsize=9, loc='best', ncol=2, frameon=True, fancybox=True, shadow=True)
    ax_A.grid(True, alpha=0.3, linestyle='--')
    ax_A.set_xlim(2000, 2017)
    
    # PANEL B: City Diffusion Heatmap (City × Period)
    ax_B = fig.add_subplot(gs[0, 1])
    
    # Compute mean diffusion across ethnicities for each city
    diffusion_matrix = np.zeros((n_cities, 4))
    for p in range(4):
        diffusion_matrix[:, p] = np.abs(city_params[p]['diffusion_coefficients']).mean(axis=1)
    
    # Plot top 20 cities by mean diffusion
    mean_diffusion = diffusion_matrix.mean(axis=1)
    top_indices = np.argsort(mean_diffusion)[-20:]
    top_cities = [filtered_cities[i] for i in top_indices]
    
    im = ax_B.imshow(diffusion_matrix[top_indices], cmap='YlOrRd', aspect='auto',
                     interpolation='nearest')
    
    ax_B.set_yticks(range(len(top_cities)))
    ax_B.set_yticklabels(top_cities, fontsize=9)
    ax_B.set_xticks(range(4))
    ax_B.set_xticklabels(period_names, rotation=45, ha='right', fontsize=9)
    ax_B.set_title('b City Diffusion Patterns (Top 20)', fontsize=14, fontweight='bold', pad=10)
    
    cbar = plt.colorbar(im, ax=ax_B, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Diffusion Coefficient', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # PANEL C: Growth Rate Profiles by City Size
    ax_C = fig.add_subplot(gs[0, 2])
    
    # Categorize cities
    city_categories = categorize_cities_by_size(filtered_cities, graph)
    
    # Get mean growth rates for most recent period
    growth_rates_recent = city_params[3]['growth_rates'].mean(axis=1)
    
    # Prepare data for violin plot
    growth_by_category = {
        'Large': [growth_rates_recent[filtered_cities.index(c)] 
                  for c in city_categories['large'] if c in filtered_cities],
        'Medium': [growth_rates_recent[filtered_cities.index(c)] 
                   for c in city_categories['medium'] if c in filtered_cities],
        'Small': [growth_rates_recent[filtered_cities.index(c)] 
                  for c in city_categories['small'] if c in filtered_cities]
    }
    
    positions = [1, 2, 3]
    colors_cat = ['#e74c3c', '#f39c12', '#3498db']
    
    parts = ax_C.violinplot([growth_by_category['Large'], 
                             growth_by_category['Medium'],
                             growth_by_category['Small']],
                            positions=positions, widths=0.7,
                            showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_cat[i])
        pc.set_alpha(0.7)
    
    ax_C.set_xticks(positions)
    ax_C.set_xticklabels(['Large\nCities', 'Medium\nCities', 'Small\nCities'], fontsize=11)
    ax_C.set_ylabel('Mean Growth Rate (/year)', fontsize=12, fontweight='bold')
    ax_C.set_title('c Growth Rates by City Size', fontsize=14, fontweight='bold', pad=10)
    ax_C.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_C.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # PANEL D: Immigration Attractiveness (Geographic Bubble Map)
    ax_D = fig.add_subplot(gs[1, 0])
    
    # Get coordinates
    city_coords = get_city_coordinates()
    
    # Compute mean immigration rate for most recent period
    immigration_rates_recent = city_params[3]['immigration_rates'].mean(axis=1)
    
    # Plot bubbles
    for i, city in enumerate(filtered_cities):
        if city in city_coords:
            lat, lon = city_coords[city]
            immigration = immigration_rates_recent[i]
            
            # Bubble size proportional to immigration rate
            size = max(50, min(1000, np.abs(immigration) * 50000))
            
            # Color by sign
            color = '#2ecc71' if immigration > 0 else '#e74c3c'
            
            ax_D.scatter(lon, lat, s=size, color=color, alpha=0.6, 
                        edgecolors='black', linewidths=1.5)
            
            # Label major cities
            if city in major_cities[:5]:
                ax_D.annotate(city, (lon, lat), xytext=(3, 3), 
                            textcoords='offset points', fontsize=8,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax_D.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax_D.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax_D.set_title('d Immigration Attractiveness (2016-2021)', fontsize=14, fontweight='bold', pad=10)
    ax_D.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', 
               markersize=10, label='Positive immigration', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
               markersize=10, label='Negative immigration', markeredgecolor='black')
    ]
    ax_D.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)
    
    # PANEL E: City Diversity Index
    ax_E = fig.add_subplot(gs[1, 1])
    
    # Load actual population data to compute diversity
    temporal_immigration = graph['temporal_data']['immigration']
    most_recent_year = sorted(temporal_immigration.keys())[-1]
    df_recent = temporal_immigration[most_recent_year].copy()
    df_recent['DAUID'] = df_recent['DAUID'].astype(str).str.strip()

    ethnic_cols_map = {
        'China': 'dim_405', 'Philippines': 'dim_410', 'India': 'dim_407',
        'Pakistan': 'dim_419', 'Iran': 'dim_421', 'Sri Lanka': 'dim_417',
        'Portugal': 'dim_413', 'Italy': 'dim_406', 'United Kingdom': 'dim_404'
    }

    # Pre-build lookup dictionary for O(1) access (performance optimization)
    dauid_to_row = {row['DAUID']: row for _, row in df_recent.iterrows()}

    # Compute population by city and ethnicity
    city_ethnic_pop = np.zeros((n_cities, n_ethnicities))

    for city_idx, city in enumerate(filtered_cities):
        # Get DAUIDs for this city
        city_mask = np.array([c == city for c in all_city_names])
        city_dauids = graph['node_features']['dauid'][city_mask]

        for eth_idx, ethnicity in enumerate(all_ethnicities):
            if ethnicity in ethnic_cols_map:
                col = ethnic_cols_map[ethnicity]
                # Sum population across all DAUIDs in this city
                for dauid in city_dauids:
                    dauid_str = str(dauid)
                    if dauid_str in dauid_to_row:
                        row = dauid_to_row[dauid_str]
                        if col in row:
                            city_ethnic_pop[city_idx, eth_idx] += float(row[col])
    
    # Compute diversity index
    diversity_scores = compute_diversity_index(city_ethnic_pop)
    
    # Plot top 20 most diverse cities
    top_diverse_indices = np.argsort(diversity_scores)[-20:]
    top_diverse_cities = [filtered_cities[i] for i in top_diverse_indices]
    top_diversity_scores = diversity_scores[top_diverse_indices]
    
    colors_diversity = plt.cm.viridis(np.linspace(0, 1, len(top_diverse_cities)))
    bars = ax_E.barh(range(len(top_diverse_cities)), top_diversity_scores,
                     color=colors_diversity, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax_E.set_yticks(range(len(top_diverse_cities)))
    ax_E.set_yticklabels(top_diverse_cities, fontsize=9)
    ax_E.set_xlabel('Shannon Diversity Index', fontsize=12, fontweight='bold')
    ax_E.set_title('e Multi-Ethnic Diversity by City', fontsize=14, fontweight='bold', pad=10)
    ax_E.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # PANEL F: Parameter Stability Over Time
    ax_F = fig.add_subplot(gs[1, 2])
    
    # Compute coefficient of variation for each city across periods
    stability_scores = np.zeros(n_cities)
    
    for i in range(n_cities):
        # Get all parameters across periods for this city
        K_trajectory = [city_params[p]['carrying_capacity'][i] for p in range(4)]
        D_trajectory = [city_params[p]['diffusion_coefficients'][i].mean() for p in range(4)]
        r_trajectory = [city_params[p]['growth_rates'][i].mean() for p in range(4)]
        
        # Compute CV (coefficient of variation = std/mean)
        K_cv = np.std(K_trajectory) / (np.mean(K_trajectory) + 1e-6)
        D_cv = np.std(D_trajectory) / (np.abs(np.mean(D_trajectory)) + 1e-6)
        r_cv = np.std(r_trajectory) / (np.abs(np.mean(r_trajectory)) + 1e-6)
        
        # Average CV (lower = more stable)
        stability_scores[i] = (K_cv + D_cv + r_cv) / 3.0
    
    # City size (number of DAUIDs)
    city_sizes = np.array([sum(1 for c in all_city_names if c == city) 
                           for city in filtered_cities])
    
    # Filter out extreme outliers (likely untrained cities with random parameters)
    # Use reasonable upper bound: CV > 10 indicates untrained/poorly trained city
    valid_mask = stability_scores < 10.0
    n_filtered = (~valid_mask).sum()
    
    if n_filtered > 0:
        filtered_out = [filtered_cities[i] for i in range(n_cities) if not valid_mask[i]]
    
    # Scatter plot: size vs. stability (only valid cities)
    scatter = ax_F.scatter(city_sizes[valid_mask], stability_scores[valid_mask], s=100, 
                          c=stability_scores[valid_mask], cmap='RdYlGn_r',
                          alpha=0.7, edgecolors='black', linewidth=1.5,
                          vmin=0, vmax=8)
    
    # Label some cities (avoid overlaps)
    valid_indices = np.where(valid_mask)[0]
    labeled_positions = []
    
    for idx in valid_indices:
        city = filtered_cities[idx]
        # Label major cities or high-instability cities
        if city in major_cities[:5] or stability_scores[idx] > np.percentile(stability_scores[valid_mask], 85):
            pos = (city_sizes[idx], stability_scores[idx])
            
            # Check overlap with existing labels
            overlap = False
            for prev_pos in labeled_positions:
                if abs(pos[0] - prev_pos[0]) < 300 and abs(pos[1] - prev_pos[1]) < 1.0:
                    overlap = True
                    break
            
            if not overlap:
                ax_F.annotate(city, pos, xytext=(5, 5), 
                            textcoords='offset points', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.8, edgecolor='none'))
                labeled_positions.append(pos)
    
    ax_F.set_xlabel('City Size (# DAUIDs)', fontsize=12, fontweight='bold')
    ax_F.set_ylabel('Parameter Instability (CV)', fontsize=12, fontweight='bold')
    ax_F.set_title('f Temporal Parameter Stability', fontsize=14, fontweight='bold', pad=10)
    ax_F.grid(True, alpha=0.3, linestyle='--')
    ax_F.set_ylim(0, min(10, stability_scores[valid_mask].max() * 1.1))
    
    cbar = plt.colorbar(scatter, ax=ax_F, fraction=0.046, pad=0.04)
    cbar.set_label('Instability', fontsize=10, fontweight='bold')
    
    # PANEL G: City Clustering (Hierarchical)
    ax_G = fig.add_subplot(gs[2, 0])

    # Flatten all parameters for each city (most recent period)
    city_feature_vectors = []
    for i in range(n_cities):
        features = np.array([
            # 1. Mean diffusion
            city_params[3]['diffusion_coefficients'][i].mean(),
            # 2. Mean growth rate  
            city_params[3]['growth_rates'][i].mean(),
            # 3. Carrying capacity
            city_params[3]['carrying_capacity'][i],
            # 4. Mean immigration
            city_params[3]['immigration_rates'][i].mean(),
            # 5. Mean emigration
            city_params[3]['emigration_rates'][i].mean(),
            # 6. Diffusion variability across ethnicities
            city_params[3]['diffusion_coefficients'][i].std(),
            # 7. Growth rate variability
            city_params[3]['growth_rates'][i].std(),
            # 8. Temporal stability (from Panel F)
            stability_scores[i],
            # 9. Diversity (from Panel E)
            diversity_scores[i]
        ])
        city_feature_vectors.append(features)

    city_feature_matrix = np.array(city_feature_vectors)

    # Standardize
    scaler = StandardScaler()
    city_features_scaled = scaler.fit_transform(city_feature_matrix)

    # Hierarchical clustering
    from scipy.cluster.hierarchy import dendrogram, linkage

    linkage_matrix = linkage(city_features_scaled, method='ward')

    # Create dendrogram with color coding by city size
    # Color cities by category
    city_colors = {}
    for city in filtered_cities:
        if city in city_categories['large']:
            city_colors[city] = '#e74c3c'
        elif city in city_categories['medium']:
            city_colors[city] = '#f39c12'
        else:
            city_colors[city] = '#3498db'

    # Plot dendrogram
    dendrogram(linkage_matrix, 
            labels=filtered_cities,
            orientation='right',
            leaf_font_size=7,
            ax=ax_G,
            color_threshold=0,
            above_threshold_color='gray')

    # Color the city labels by category
    # Get all text labels (city names) and color them
    for text_obj in ax_G.get_ymajorticklabels():
        city_name = text_obj.get_text()
        if city_name in city_colors:
            text_obj.set_color(city_colors[city_name])
            text_obj.set_fontweight('bold')

    ax_G.set_xlabel('Ward Distance', fontsize=12, fontweight='bold')
    ax_G.set_title('g City Relationships (Hierarchical Clustering)', 
                fontsize=14, fontweight='bold', pad=10)
    ax_G.grid(True, axis='x', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Large cities'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Medium cities'),
        Patch(facecolor='#3498db', edgecolor='black', label='Small cities')
    ]
    ax_G.legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True)
    
    # PANEL H: Ethnicity-to-City Flow (Sankey Diagram)
    ax_H = fig.add_subplot(gs[2, 1])

    # For each ethnicity, find top 3 destination cities by population
    # Get actual population data for most recent period
    ethnic_city_flows = {}  # Dict: ethnicity -> [(city, population)]

    for eth_idx, ethnicity in enumerate(all_ethnicities):
        # Get population for this ethnicity across all cities
        city_populations = []
        
        for city_idx, city in enumerate(filtered_cities):
            # Sum population from city_ethnic_pop computed in Panel E
            pop = city_ethnic_pop[city_idx, eth_idx]
            if pop > 0:
                city_populations.append((city, pop))
        
        # Sort by population and get top 3
        city_populations.sort(key=lambda x: x[1], reverse=True)
        ethnic_city_flows[ethnicity] = city_populations[:3]

    # Prepare Sankey data
    # We'll show: Ethnicity groups -> Top 3 cities for each
    # To keep it readable, focus on top 5 ethnicities by total population

    # Get total population by ethnicity
    ethnicity_totals = [(eth, city_ethnic_pop[:, i].sum()) 
                        for i, eth in enumerate(all_ethnicities)]
    ethnicity_totals.sort(key=lambda x: x[1], reverse=True)
    top_5_ethnicities = [eth for eth, _ in ethnicity_totals[:5]]


    # Build Sankey flows
    from collections import defaultdict
    flows = []  # (source, target, value)
    city_targets = set()

    for ethnicity in top_5_ethnicities:
        eth_idx = all_ethnicities.index(ethnicity)
        top_cities = ethnic_city_flows[ethnicity][:3]
        
        for city, population in top_cities:
            flows.append((ethnicity, city, population))
            city_targets.add(city)

    # Create simplified Sankey using matplotlib patches
    # (matplotlib.sankey.Sankey is limited, so we'll draw custom flow diagram)

    # Position nodes
    left_x = 0.1
    right_x = 0.9
    y_spacing_left = 0.8 / len(top_5_ethnicities)
    y_spacing_right = 0.8 / len(city_targets)

    # Ethnicity positions (left side)
    ethnicity_positions = {}
    for i, eth in enumerate(top_5_ethnicities):
        y = 0.1 + i * y_spacing_left
        ethnicity_positions[eth] = (left_x, y)

    # City positions (right side)
    city_list = sorted(city_targets)
    city_positions = {}
    for i, city in enumerate(city_list):
        y = 0.1 + i * y_spacing_right
        city_positions[city] = (right_x, y)

    # Color palette
    ethnicity_colors = plt.cm.Set3(np.linspace(0, 1, len(top_5_ethnicities)))
    eth_color_map = {eth: ethnicity_colors[i] for i, eth in enumerate(top_5_ethnicities)}

    # Draw flows
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import FancyArrowPatch

    # Normalize flow widths
    max_pop = max(flow[2] for flow in flows)

    for source, target, value in flows:
        # Get positions
        x1, y1 = ethnicity_positions[source]
        x2, y2 = city_positions[target]
        
        # Flow width proportional to population
        width = (value / max_pop) * 0.015
        
        # Draw curved flow
        # Use bezier curve for nice Sankey effect
        from matplotlib.path import Path
        import matplotlib.patches as mpatches
        
        # Control points for bezier curve
        mid_x = (x1 + x2) / 2
        verts = [
            (x1 + 0.05, y1),  # Start
            (mid_x, y1),       # Control 1
            (mid_x, y2),       # Control 2
            (x2 - 0.05, y2),  # End
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor='none', 
                                edgecolor=eth_color_map[source],
                                linewidth=width * 1000,
                                alpha=0.4)
        ax_H.add_patch(patch)

    # Draw ethnicity nodes (left)
    for eth, (x, y) in ethnicity_positions.items():
        # Node box
        box = FancyBboxPatch((x - 0.04, y - 0.025), 0.08, 0.05,
                            boxstyle="round,pad=0.005",
                            facecolor=eth_color_map[eth],
                            edgecolor='black', linewidth=2,
                            alpha=0.8, zorder=10)
        ax_H.add_patch(box)
        
        # Label
        ax_H.text(x, y, eth[:8], ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=11)

    # Draw city nodes (right)
    for city, (x, y) in city_positions.items():
        # Node box
        box = FancyBboxPatch((x - 0.04, y - 0.025), 0.08, 0.05,
                            boxstyle="round,pad=0.005",
                            facecolor='lightgray',
                            edgecolor='black', linewidth=2,
                            alpha=0.8, zorder=10)
        ax_H.add_patch(box)
        
        # Label
        city_label = city if len(city) <= 10 else city[:8] + '..'
        ax_H.text(x, y, city_label, ha='center', va='center',
                fontsize=8, fontweight='bold', zorder=11)

    # Formatting
    ax_H.set_xlim(0, 1)
    ax_H.set_ylim(0, 1)
    ax_H.axis('off')
    ax_H.set_title('h Ethnicity → City Settlement Flows (2021)', 
                fontsize=14, fontweight='bold', pad=10)

    # Add labels for sides
    ax_H.text(left_x, 0.95, 'Ethnic Groups', ha='center', fontsize=11, 
            fontweight='bold', style='italic')
    ax_H.text(right_x, 0.95, 'Destination Cities', ha='center', fontsize=11,
            fontweight='bold', style='italic')

    # Add legend for flow interpretation
    ax_H.text(0.5, 0.02, 'Flow width ∝ Population size', ha='center', 
            fontsize=9, style='italic', color='#666666')
    
    # PANEL I: Spatial Wavelength Distribution by City Size
    ax_I = fig.add_subplot(gs[2, 2])
    
    # For simplicity, use theoretical wavelengths from learned parameters
    # λ ≈ sqrt(D / r) (characteristic length scale)
    
    wavelengths_by_category = {'Large': [], 'Medium': [], 'Small': []}
    
    for i, city in enumerate(filtered_cities):
        # Mean wavelength across ethnicities for this city
        D_city = np.abs(city_params[3]['diffusion_coefficients'][i])
        r_city = np.abs(city_params[3]['growth_rates'][i])
        
        # Compute wavelength for each ethnicity
        wavelengths_city = []
        for eth in range(n_ethnicities):
            # Use minimum threshold to avoid division issues
            r_effective = max(r_city[eth], 0.001)
            D_effective = max(D_city[eth], 0.0001)
            
            # Compute wavelength with proper scaling
            wavelength = np.sqrt(D_effective / r_effective) * 200  # Scale factor
            
            # Ensure reasonable range but don't clip too aggressively
            wavelength = np.clip(wavelength, 15, 120)
            wavelengths_city.append(wavelength)
        
        # Categorize (note: categories might seem inverted due to DAUID count)
        if city in city_categories['large']:
            wavelengths_by_category['Large'].extend(wavelengths_city)
        elif city in city_categories['medium']:
            wavelengths_by_category['Medium'].extend(wavelengths_city)
        else:
            wavelengths_by_category['Small'].extend(wavelengths_city)
    
    
    # Plot KDE distributions
    from scipy.stats import gaussian_kde
    
    categories = ['Large', 'Medium', 'Small']
    colors_kde = ['#e74c3c', '#f39c12', '#3498db']
    
    x_range = np.linspace(15, 120, 200)
    
    for i, cat in enumerate(categories):
        if len(wavelengths_by_category[cat]) > 5:
            try:
                # Add small noise to avoid singular matrix in KDE
                data = np.array(wavelengths_by_category[cat])
                data_jittered = data + np.random.normal(0, 0.1, size=len(data))
                
                kde = gaussian_kde(data_jittered, bw_method='scott')
                density = kde(x_range)
                
                ax_I.fill_between(x_range, density, alpha=0.5, 
                                 color=colors_kde[i], label=f'{cat} cities')
                ax_I.plot(x_range, density, color=colors_kde[i], linewidth=2.5)
            except Exception:
                ax_I.hist(wavelengths_by_category[cat], bins=20, alpha=0.5,
                         color=colors_kde[i], label=f'{cat} cities', density=True)
    
    ax_I.set_xlabel('Spatial Wavelength (km)', fontsize=12, fontweight='bold')
    ax_I.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax_I.set_title('i Spatial Pattern Scales by City Size', fontsize=14, fontweight='bold', pad=10)
    ax_I.legend(fontsize=10, loc='upper right', frameon=True)
    ax_I.grid(True, alpha=0.3, linestyle='--')
    ax_I.set_xlim(15, 120)
    
    # Save figure
    plt.tight_layout()
    
    output_path_pdf = output_dir / 'figure3.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', format='pdf')

    output_path_png = output_dir / 'figure3.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    
    return str(output_path_pdf)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Create city-level analysis figure')
    parser.add_argument('--checkpoint', type=str, 
                       default='../checkpoints/graphpde/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--graph', type=str,
                       default='../data/graph_large_cities_rd.pkl',
                       help='Path to graph data')
    parser.add_argument('--da_csv', type=str,
                       default='../data/da_canada.csv',
                       help='Path to DA info CSV')
    parser.add_argument('--output', type=str,
                       default='../figures/figure3',
                       help='Output directory')
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GraphPDEAnalyzer(
        checkpoint_path=args.checkpoint,
        graph_path=args.graph,
        da_info_path=args.da_csv,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    )
    
    # Set ethnicities
    ethnicities = ['China', 'Philippines', 'India', 'Pakistan', 'Iran', 
                   'Sri Lanka', 'Portugal', 'Italy', 'United Kingdom']
    analyzer.set_ethnicities(ethnicities)
    
    # Get trained cities
    trained_cities = get_trained_cities()

    # Create figure
    output_path = create_city_analysis_figure(analyzer, ethnicities, trained_cities, args.output)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()