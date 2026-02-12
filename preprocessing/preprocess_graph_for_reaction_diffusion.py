"""
preprocess_graph_for_reaction_diffusion.py

Creates a properly indexed graph structure for physics-informed modeling.
Filters to cities above population threshold and ensures complete graph connectivity.
"""

import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_raw_graph(graph_path='./data/canada_graph_multilayer_temporal.pkl'):
    """Load the raw multi-layer temporal graph"""
    print("="*80)
    print("LOADING RAW GRAPH")
    print("="*80)
    
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"Raw graph nodes: {len(graph_data['node_features']['dauid']):,}")
    print(f"Temporal years available: {graph_data['temporal_data']['years']}")
    
    return graph_data


def identify_large_cities(graph_data, da_csv_path, min_population=100000):
    """
    Identify cities that meet population threshold in at least one census year.
    Uses actual census data to determine city sizes.
    """
    print(f"\n{'='*80}")
    print(f"IDENTIFYING CITIES WITH POPULATION >= {min_population:,}")
    print("="*80)
    
    # Load DA to city mapping
    da_df = pd.read_csv(da_csv_path)
    da_df['DAUID'] = da_df['DAUID'].astype(str).str.strip()
    dauid_to_city = dict(zip(da_df['DAUID'], da_df['Name']))
    
    # Calculate city populations across all years
    temporal_features = graph_data['temporal_data']['features']
    city_populations = defaultdict(lambda: defaultdict(float))
    
    for year in sorted(temporal_features.keys()):
        df = temporal_features[year].copy()
        df['DAUID'] = df['DAUID'].astype(str).str.strip()
        df['City'] = df['DAUID'].map(dauid_to_city)
        df = df[df['City'].notna()]
        
        # dim_2 is total population
        if 'dim_2' in df.columns:
            df['dim_2'] = pd.to_numeric(df['dim_2'], errors='coerce').fillna(0)
            city_totals = df.groupby('City')['dim_2'].sum()
            
            for city, pop in city_totals.items():
                city_populations[city][year] = pop
    
    # Find cities meeting threshold in ANY year
    large_cities = set()
    city_max_pops = {}
    
    for city, years in city_populations.items():
        max_pop = max(years.values()) if years else 0
        city_max_pops[city] = max_pop
        if max_pop >= min_population:
            large_cities.add(city)
    
    print(f"\nFound {len(large_cities)} cities meeting threshold:")
    for city in sorted(large_cities):
        print(f"  {city}: {city_max_pops[city]:,.0f} peak population")
    
    return large_cities, dauid_to_city


def filter_graph_to_large_cities(graph_data, large_cities, dauid_to_city):
    """
    Filter graph to only include nodes from large cities.
    Creates proper indexing for matrix operations.
    """
    print(f"\n{'='*80}")
    print("FILTERING GRAPH")
    print("="*80)
    
    # Get original data
    all_dauids = graph_data['node_features']['dauid']
    all_cities = graph_data['node_features']['city']
    all_coords = graph_data['node_features']['coordinates']
    old_adj = graph_data['layers']['adjacency']
    
    # Convert DAUIDs to strings for consistency
    all_dauids = np.array([str(d) for d in all_dauids])
    
    # Filter to large cities
    large_city_mask = np.array([city in large_cities for city in all_cities])
    
    filtered_dauids = all_dauids[large_city_mask]
    filtered_cities = all_cities[large_city_mask]
    filtered_coords = all_coords[large_city_mask]
    
    n_filtered = len(filtered_dauids)
    
    print(f"Nodes: {len(all_dauids):,} → {n_filtered:,}")
    
    # Create index mappings (CRITICAL for reaction-diffusion)
    dauid_to_idx = {dauid: idx for idx, dauid in enumerate(filtered_dauids)}
    idx_to_dauid = {idx: dauid for idx, dauid in enumerate(filtered_dauids)}
    
    # Filter adjacency matrix
    print("Filtering adjacency matrix...")
    
    # Map old indices to new indices
    old_to_new = {}
    for new_idx, old_idx in enumerate(np.where(large_city_mask)[0]):
        old_to_new[old_idx] = new_idx
    
    # Extract edges
    old_adj_coo = old_adj.tocoo()
    
    new_rows, new_cols, new_data = [], [], []
    for i, j, v in zip(old_adj_coo.row, old_adj_coo.col, old_adj_coo.data):
        if i in old_to_new and j in old_to_new:
            new_rows.append(old_to_new[i])
            new_cols.append(old_to_new[j])
            new_data.append(v)
    
    # Create new adjacency matrix
    new_adj = sp.csr_matrix(
        (new_data, (new_rows, new_cols)),
        shape=(n_filtered, n_filtered)
    )
    
    print(f"Edges: {old_adj.nnz:,} → {new_adj.nnz:,}")
    print(f"Average degree: {new_adj.nnz / n_filtered:.2f}")
    
    # Check connectivity
    n_components, labels = connected_components(new_adj, directed=False)
    
    if n_components > 1:
        print(f"\nWARNING: Graph has {n_components} disconnected components")
        print("Connecting components...")
        new_adj = connect_components(new_adj, filtered_coords, labels)
        
        # Verify
        n_components_after, _ = connected_components(new_adj, directed=False)
        print(f"After connection: {n_components_after} component(s)")
    
    return {
        'dauids': filtered_dauids,
        'cities': filtered_cities,
        'coordinates': filtered_coords,
        'adjacency': new_adj,
        'dauid_to_idx': dauid_to_idx,
        'idx_to_dauid': idx_to_dauid,
        'n_nodes': n_filtered
    }


def connect_components(adj, coords, labels):
    """Connect disconnected graph components via nearest neighbors"""
    unique_components = np.unique(labels)
    main_component = unique_components[np.argmax(np.bincount(labels))]
    
    adj_lil = adj.tolil()
    
    for comp in unique_components:
        if comp == main_component:
            continue
        
        comp_nodes = np.where(labels == comp)[0]
        main_nodes = np.where(labels == main_component)[0]
        
        # Find closest pair
        min_dist = float('inf')
        best_pair = None
        
        for node_i in comp_nodes[:min(10, len(comp_nodes))]:
            distances = np.linalg.norm(coords[main_nodes] - coords[node_i], axis=1)
            nearest_idx = np.argmin(distances)
            nearest_main = main_nodes[nearest_idx]
            dist = distances[nearest_idx]
            
            if dist < min_dist:
                min_dist = dist
                best_pair = (node_i, nearest_main)
        
        if best_pair:
            i, j = best_pair
            adj_lil[i, j] = 1.0
            adj_lil[j, i] = 1.0
    
    return adj_lil.tocsr()


def compute_graph_laplacian(adj):
    """Compute normalized Laplacian for diffusion operator"""
    print("\nComputing graph Laplacian...")
    
    # Degree matrix
    degrees = np.array(adj.sum(axis=1)).flatten()
    
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    with np.errstate(divide='ignore', invalid='ignore'):
        D_sqrt_inv = sp.diags(1.0 / np.sqrt(degrees))
        D_sqrt_inv.data[~np.isfinite(D_sqrt_inv.data)] = 0
    
    L_norm = sp.identity(adj.shape[0]) - D_sqrt_inv @ adj @ D_sqrt_inv
    
    print("Laplacian computed (normalized for reaction-diffusion)")
    
    return L_norm


def save_filtered_graph(filtered_data, graph_data, output_path):
    """Save filtered graph with all necessary components for reaction-diffusion"""
    print(f"\n{'='*80}")
    print("SAVING FILTERED GRAPH")
    print("="*80)
    
    # Compute Laplacian
    laplacian = compute_graph_laplacian(filtered_data['adjacency'])
    
    # Package for reaction-diffusion model
    output = {
        # Core graph structure
        'adjacency': filtered_data['adjacency'],
        'laplacian': laplacian,
        
        # Node data
        'node_features': {
            'dauid': filtered_data['dauids'],
            'city': filtered_data['cities'],
            'coordinates': filtered_data['coordinates'],
        },
        
        # Critical for indexing
        'dauid_to_idx': filtered_data['dauid_to_idx'],
        'idx_to_dauid': filtered_data['idx_to_dauid'],
        
        # Temporal data (keep for feature extraction)
        'temporal_data': graph_data['temporal_data'],
        
        # Metadata
        'metadata': {
            'n_nodes': filtered_data['n_nodes'],
            'n_edges': filtered_data['adjacency'].nnz,
            'filtered': True,
            'crs': str(graph_data['metadata']['crs']),
            'years_available': graph_data['temporal_data']['years']
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved to: {output_path}")
    print(f"  Nodes: {filtered_data['n_nodes']:,}")
    print(f"  Edges: {filtered_data['adjacency'].nnz:,}")
    print(f"  Components: {sp.csgraph.connected_components(filtered_data['adjacency'], directed=False)[0]}")


def main():
    # Configuration
    RAW_GRAPH_PATH = './data/canada_graph_multilayer_temporal.pkl'
    DA_CSV_PATH = './data/da_canada.csv'
    OUTPUT_PATH = './data/graph_large_cities_rd.pkl'
    MIN_POPULATION = 100000  # 100k threshold (more cities than 200k)
    
    # Load raw graph
    graph_data = load_raw_graph(RAW_GRAPH_PATH)
    
    # Identify large cities
    large_cities, dauid_to_city = identify_large_cities(
        graph_data, DA_CSV_PATH, MIN_POPULATION
    )
    
    # Filter graph
    filtered_data = filter_graph_to_large_cities(
        graph_data, large_cities, dauid_to_city
    )
    
    # Save
    save_filtered_graph(filtered_data, graph_data, OUTPUT_PATH)
    
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nFiltered graph ready for reaction-diffusion modeling")
    print(f"Next: Run create_reaction_diffusion_splits.py")


if __name__ == "__main__":
    main()