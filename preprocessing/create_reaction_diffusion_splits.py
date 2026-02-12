"""
create_reaction_diffusion_splits.py

Creates train/val/test splits for reaction-diffusion model.
- Uses RAW population counts (not log-transformed)
- Filters zero→zero transitions
- Ensures spatial separation
- Creates graph-aligned samples for efficient training
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# Ethnic groups to model
ETHNIC_COLS = {
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

# Census features for model
# FEATURE_COLS = [
#     'dim_2', 'dim_581', 'dim_604', 'dim_785', 'dim_776',
#     'dim_706', 'dim_707', 'dim_708',
#     'dim_713', 'dim_763', 'dim_765', 'dim_766',
#     'dim_793', 'dim_795', 'dim_799', 'dim_803',
#     'dim_900', 'dim_901', 'dim_903', 'dim_904', 'dim_905',
#     'dim_1204', 'dim_1205', 'dim_1250', 'dim_1251', 'dim_1302',
#     'dim_1325', 'dim_1326', 'dim_1344', 'dim_1314'
# ]

FEATURE_COLS = [
    'dim_2',    # Total population by sex and age groups
    'dim_581',  # Total visible minority population
    'dim_594',  # Not a visible minority
    'dim_582',  # Chinese
    'dim_583',  # South Asian
    'dim_584',  # Black
    'dim_585',  # Filipino
    'dim_586',  # Latin American
    'dim_587',  # Southeast Asian
    'dim_588',  # Arab
    'dim_589',  # West Asian
    'dim_590',  # Korean
    'dim_591',  # Japanese
    'dim_601',  # Non-immigrants
    'dim_604',  # Immigrants
    'dim_645',  # Non-permanent residents
    'dim_649',  # Total immigrant population by period of immigration 1971 to 1980
    'dim_650',  # Total immigrant population by period of immigration 1981 to 1990
    'dim_651',  # Total immigrant population by period of immigration 1991 to 2000
    'dim_655',  # Total immigrant population by age at immigration Under 5 years
    'dim_656',  # Total immigrant population by age at immigration 5 to 14 years
    'dim_657',  # Total immigrant population by age at immigration 15 to 24 years
    'dim_658',  # Total immigrant population by age at immigration 25 to 44 years
    'dim_659',  # Total immigrant population by age at immigration 45 years and over
    'dim_713',  # Non-official languages (mother tongue)
    'dim_763',  # English only (knowledge of official languages)
    'dim_765',  # English and French (knowledge of official languages)
    'dim_766',  # Neither English nor French (knowledge of official languages)
    'dim_715',  # Chinese (mother tongue)
    'dim_722',  # Punjabi (mother tongue)
    'dim_723',  # Tagalog (Pilipino Filipino) (mother tongue)
    'dim_724',  # Arabic (mother tongue)
    'dim_717',  # Spanish (mother tongue)
    'dim_719',  # Portuguese (mother tongue)
    'dim_1325', # Owned (dwelling tenure)
    'dim_1326', # Rented (dwelling tenure)
    'dim_1320', # Apartment building that has five or more storeys
    'dim_1321', # Apartment building that has fewer than five storeys
    'dim_1316', # Single-detached house
    'dim_1333', # Spending 30% or more of household income on shelter costs
    'dim_1345', # Tenant households in subsidized housing
    'dim_1204', # Average income $
    'dim_1205', # Median income $
    'dim_1250', # Average household income $
    'dim_1251', # Median household income $
    'dim_1302', # Incidence of low income - total population in private households
    'dim_1303', # Incidence of low income - persons less than 18 years of age
    'dim_1344', # Average gross rent $
    'dim_1314', # Average value of dwelling $
    'dim_793',  # No certificate diploma or degree
    'dim_795',  # High school certificate or equivalent
    'dim_799',  # University certificate diploma or degree
    'dim_803',  # Master's degree
    'dim_900',  # Employed
    'dim_901',  # Unemployed
    'dim_903',  # Participation rate
    'dim_904',  # Employment rate
    'dim_905',  # Unemployment rate
    'dim_776',  # Movers (mobility status 1 year ago)
    'dim_782',  # External migrants (mobility status 1 year ago)
    'dim_785',  # Movers (mobility status 5 years ago)
    'dim_791',  # External migrants (mobility status 5 years ago)
    'dim_52',   # Married couples
    'dim_60',   # With children at home
    'dim_142',  # 1 person household
    'dim_143',  # 2 persons household
    'dim_145',  # 4 persons household
    'dim_146',  # 5 or more persons household
    'dim_706',  # 1st generation
    'dim_707',  # 2nd generation
    'dim_708',  # 3rd generation or more
    'dim_1354', # 1971 to 1980 (period of construction)
    'dim_1356', # 1986 to 1990 (period of construction)
    'dim_1358', # 1996 to 2000 (period of construction)
]


def load_filtered_graph(graph_path='./data/graph_large_cities_rd.pkl'):
    """Load preprocessed graph"""
    print("="*80)
    print("LOADING FILTERED GRAPH")
    print("="*80)
    
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Nodes: {graph['metadata']['n_nodes']:,}")
    print(f"Edges: {graph['metadata']['n_edges']:,}")
    print(f"Years: {graph['metadata']['years_available']}")
    
    return graph


def create_spatial_splits(graph, n_clusters_per_city=15, 
                         train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Create spatially-separated train/val/test splits.
    Clusters nodes within each city, then assigns clusters to splits.
    """
    print(f"\n{'='*80}")
    print("CREATING SPATIAL SPLITS")
    print("="*80)
    
    coords = graph['node_features']['coordinates']
    cities = graph['node_features']['city']
    dauids = graph['node_features']['dauid']
    
    dauid_to_split = {}
    
    for city in np.unique(cities):
        city_mask = cities == city
        city_dauids = dauids[city_mask]
        city_coords = coords[city_mask]
        
        n_dauids = len(city_dauids)
        n_clusters = min(n_clusters_per_city, max(3, n_dauids // 20))
        
        if n_dauids < 10:
            # Small city: assign randomly
            np.random.seed(42)
            splits = np.random.choice(['train', 'val', 'test'], size=n_dauids,
                                     p=[train_ratio, val_ratio, test_ratio])
            for dauid, split in zip(city_dauids, splits):
                dauid_to_split[str(dauid)] = split
        else:
            # Cluster spatially
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(city_coords)
            
            # Assign clusters to splits
            unique_clusters = np.unique(cluster_labels)
            np.random.seed(42)
            shuffled = np.random.permutation(unique_clusters)
            
            n_train = int(len(unique_clusters) * train_ratio)
            n_val = int(len(unique_clusters) * val_ratio)
            
            train_clusters = set(shuffled[:n_train])
            val_clusters = set(shuffled[n_train:n_train + n_val])
            test_clusters = set(shuffled[n_train + n_val:])
            
            # Assign DAUIDs based on cluster
            for dauid, cluster in zip(city_dauids, cluster_labels):
                if cluster in train_clusters:
                    split = 'train'
                elif cluster in val_clusters:
                    split = 'val'
                else:
                    split = 'test'
                dauid_to_split[str(dauid)] = split
        
        n_train = sum(1 for d in city_dauids if dauid_to_split[str(d)] == 'train')
        n_val = sum(1 for d in city_dauids if dauid_to_split[str(d)] == 'val')
        n_test = sum(1 for d in city_dauids if dauid_to_split[str(d)] == 'test')
        
        print(f"{city}: {n_train} train, {n_val} val, {n_test} test")
    
    return dauid_to_split


def create_temporal_samples(graph, dauid_to_split, filter_zero_transitions=True):
    """
    Create samples for reaction-diffusion model.
    
    Key points:
    - Uses RAW population counts (reaction-diffusion needs actual concentrations)
    - Filters zero→zero (no useful gradient information)
    - Includes neighbor features for graph convolution
    """
    print(f"\n{'='*80}")
    print("CREATING TEMPORAL SAMPLES")
    print("="*80)
    
    temporal_features = graph['temporal_data']['features']
    temporal_immigration = graph['temporal_data']['immigration']
    years = sorted(temporal_features.keys())
    
    valid_dauids = set(graph['dauid_to_idx'].keys())
    adjacency = graph['adjacency']
    
    all_samples = []
    zero_filtered = 0
    
    for i in range(len(years) - 1):
        year_t = years[i]
        year_t1 = years[i + 1]
        
        print(f"\nTransition: {year_t} → {year_t1}")
        
        # Load data
        features_t = temporal_features[year_t1].copy()
        immigration_t = temporal_immigration[year_t].copy()
        immigration_t1 = temporal_immigration[year_t1].copy()
        
        # Standardize
        for df in [features_t, immigration_t, immigration_t1]:
            df['DAUID'] = df['DAUID'].astype(str).str.strip()
        
        # Filter to valid DAUIDs
        features_t = features_t[features_t['DAUID'].isin(valid_dauids)]
        immigration_t = immigration_t[immigration_t['DAUID'].isin(valid_dauids)]
        immigration_t1 = immigration_t1[immigration_t1['DAUID'].isin(valid_dauids)]
        
        # Merge
        data_t = features_t.merge(immigration_t, on='DAUID', how='inner')
        data = data_t.merge(immigration_t1[['DAUID'] + list(ETHNIC_COLS.values())],
                           on='DAUID', suffixes=('_t', '_t1'))
        
        # Filter to split DAUIDs
        data = data[data['DAUID'].isin(dauid_to_split.keys())]
        
        print(f"  DAUIDs: {len(data):,}")
        
        transition_zero_filtered = 0
        
        # Create samples
        for _, row in data.iterrows():
            dauid = row['DAUID']
            split = dauid_to_split[dauid]
            node_idx = graph['dauid_to_idx'][dauid]
            
            # Get neighbor indices
            neighbors = adjacency[node_idx].indices
            
            # Extract census features
            census_feats = {}
            for feat in FEATURE_COLS:
                col_name = feat if feat in row.index else f"{feat}_t"
                if col_name in row.index:
                    census_feats[f'census_{feat}'] = pd.to_numeric(row[col_name], errors='coerce')
                else:
                    census_feats[f'census_{feat}'] = np.nan
            
            # Process each ethnicity
            for eth_name, eth_col in ETHNIC_COLS.items():
                col_t = f"{eth_col}_t"
                col_t1 = f"{eth_col}_t1"
                
                if col_t in row.index and col_t1 in row.index:
                    pop_t = pd.to_numeric(row[col_t], errors='coerce')
                    pop_t1 = pd.to_numeric(row[col_t1], errors='coerce')
                    
                    pop_t = 0.0 if pd.isna(pop_t) else float(pop_t)
                    pop_t1 = 0.0 if pd.isna(pop_t1) else float(pop_t1)
                    
                    # Filter zero→zero
                    if filter_zero_transitions and pop_t == 0.0 and pop_t1 == 0.0:
                        zero_filtered += 1
                        transition_zero_filtered += 1
                        continue
                    
                    # Get other ethnic populations (for interaction terms)
                    other_pops = {}
                    for other_eth, other_col in ETHNIC_COLS.items():
                        if other_eth != eth_name:
                            other_col_t = f"{other_col}_t"
                            if other_col_t in row.index:
                                other_pop = pd.to_numeric(row[other_col_t], errors='coerce')
                                other_pops[f'other_{other_eth}'] = 0.0 if pd.isna(other_pop) else float(other_pop)
                    
                    sample = {
                        'dauid': dauid,
                        'node_idx': node_idx,
                        'n_neighbors': len(neighbors),
                        'year_t': year_t,
                        'year_t1': year_t1,
                        'ethnicity': eth_name,
                        'split': split,
                        'pop_t': pop_t,          # RAW counts for reaction-diffusion
                        'pop_t1': pop_t1,
                        'change': pop_t1 - pop_t,
                        **census_feats,
                        **other_pops
                    }
                    
                    all_samples.append(sample)
        
        print(f"  Samples: {len(all_samples):,}, Filtered (0→0): {transition_zero_filtered:,}")
    
    print(f"\nTotal zero→zero filtered: {zero_filtered:,}")
    
    df = pd.DataFrame(all_samples)
    
    # Fill NaNs
    feature_cols = [c for c in df.columns if c.startswith('census_') or c.startswith('other_')]
    df[feature_cols] = df[feature_cols].fillna(0)
    
    return df


def save_splits(samples_df, output_dir='./data'):
    """Save splits as pickle (more efficient for large datasets)"""
    print(f"\n{'='*80}")
    print("SAVING SPLITS")
    print("="*80)
    
    train = samples_df[samples_df['split'] == 'train'].drop('split', axis=1)
    val = samples_df[samples_df['split'] == 'val'].drop('split', axis=1)
    test = samples_df[samples_df['split'] == 'test'].drop('split', axis=1)
    
    # Save as pickle (preserves dtypes, faster loading)
    train.to_pickle(f'{output_dir}/train_rd.pkl')
    val.to_pickle(f'{output_dir}/val_rd.pkl')
    test.to_pickle(f'{output_dir}/test_rd.pkl')
    
    # Also save as CSV for inspection
    train.to_csv(f'{output_dir}/train_rd.csv', index=False)
    val.to_csv(f'{output_dir}/val_rd.csv', index=False)
    test.to_csv(f'{output_dir}/test_rd.csv', index=False)
    
    print(f"\nTrain: {len(train):,} samples ({train['dauid'].nunique():,} nodes)")
    print(f"Val:   {len(val):,} samples ({val['dauid'].nunique():,} nodes)")
    print(f"Test:  {len(test):,} samples ({test['dauid'].nunique():,} nodes)")
    
    print(f"\nSaved: train_rd.pkl, val_rd.pkl, test_rd.pkl")
    print(f"Also:  train_rd.csv, val_rd.csv, test_rd.csv (for inspection)")


def main():
    # Load preprocessed graph
    graph = load_filtered_graph()
    
    # Create spatial splits
    dauid_to_split = create_spatial_splits(graph)
    
    # Create temporal samples
    samples = create_temporal_samples(graph, dauid_to_split, filter_zero_transitions=True)
    
    # Save
    save_splits(samples)
    
    print(f"\n{'='*80}")
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print("\nReady for reaction-diffusion model training")
    print("Samples use RAW population counts (not log-transformed)")
    print("All nodes have valid graph indices and neighbors")


if __name__ == "__main__":
    main()