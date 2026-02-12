"""
dataloader_with_city_filter.py

Data loading with city filtering capability.
Thread-safe implementation using pre-extracted numpy arrays.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class ReactionDiffusionDatasetFiltered(Dataset):
    """
    Dataset with city filtering capability.
    Thread-safe implementation using pre-extracted numpy arrays.
    """

    def __init__(self, samples_path, graph_path, ethnicity_filter=None,
                 census_stats=None, is_training=False,
                 city_filter=None, exclude_cities=None,
                 min_city_r2=None, city_metrics_path=None,
                 da_csv_path='./data/da_canada.csv'):

        # Load samples
        if samples_path.endswith('.pkl'):
            df = pd.read_pickle(samples_path)
        else:
            df = pd.read_csv(samples_path)

        # Load graph
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        self.adjacency = self.graph['adjacency']
        self.dauid_to_idx = self.graph['dauid_to_idx']

        if 'node_features' in self.graph and 'coordinates' in self.graph['node_features']:
            self.coordinates = self.graph['node_features']['coordinates']
        else:
            n_nodes = self.adjacency.shape[0]
            self.coordinates = np.random.randn(n_nodes, 2).astype(np.float32)

        # Load DAUID to city name mapping
        da_df = pd.read_csv(da_csv_path)
        da_df['DAUID'] = da_df['DAUID'].astype(str).str.strip()
        self.dauid_to_city_name = dict(zip(da_df['DAUID'], da_df['Name']))

        # Add city names to samples (convert dauid to string for mapping)
        df['city_name'] = df['dauid'].astype(str).map(self.dauid_to_city_name)

        # APPLY CITY FILTERING
        original_len = len(df)
        cities_to_exclude = set()

        # Option 1: Filter by R² threshold
        if min_city_r2 is not None and city_metrics_path is not None:
            city_metrics = pd.read_csv(city_metrics_path)
            poor_cities = city_metrics[city_metrics['r2'] < min_city_r2]['city'].tolist()
            cities_to_exclude.update(poor_cities)

        # Option 2: Explicit exclusion list
        if exclude_cities is not None:
            cities_to_exclude.update(exclude_cities)

        # Option 3: Include only specific cities
        if city_filter is not None:
            all_cities = set(df['city_name'].unique())
            cities_to_exclude.update(all_cities - set(city_filter))

        # Apply filtering
        if cities_to_exclude:
            df = df[~df['city_name'].isin(cities_to_exclude)]

        # Filter ethnicities if specified
        if ethnicity_filter:
            df = df[df['ethnicity'].isin(ethnicity_filter)]

        # Identify feature columns
        self.census_cols = [c for c in df.columns if c.startswith('census_')]
        self.other_ethnic_cols = [c for c in df.columns if c.startswith('other_')]

        # Compute or use provided normalization statistics
        if is_training or census_stats is None:
            census_data = df[self.census_cols].values.astype(np.float32)
            self.census_mean = np.nan_to_num(census_data.mean(axis=0), nan=0.0)
            self.census_std = np.nan_to_num(census_data.std(axis=0), nan=1.0)
            self.census_std = np.clip(self.census_std, 1.0, None)
        else:
            self.census_mean = census_stats['mean']
            self.census_std = census_stats['std']

        # Build ethnicity mapping
        self.ethnicity_map = {eth: i for i, eth in enumerate(sorted(df['ethnicity'].unique()))}
        self.n_ethnicities = len(self.ethnicity_map)

        # Store samples DataFrame temporarily for graph filtering and cache building
        self.samples = df

        # Filter graph to match filtered samples
        if city_filter is not None or exclude_cities is not None:
            self._filter_graph_to_cities(city_filter, exclude_cities, da_csv_path)

        self.n_nodes = self.adjacency.shape[0]

        # Precompute full graph states for each year
        self._precompute_full_graph_states()

        # Pre-extract all data as numpy arrays
        self._extract_data_to_numpy(df)

        # Store length
        self._len = len(df)

        # Delete DataFrame to free memory (keep only for city stats)
        # Store minimal info needed for get_city_statistics and other external access
        self._city_names = df['city_name'].values
        self._dauids_for_stats = df['dauid'].values
        self._pop_t_for_stats = df['pop_t'].values
        self._pop_t1_for_stats = df['pop_t1'].values

        # Store ethnicities for external access (e.g., evaluate scripts)
        self._ethnicities = df['ethnicity'].values

        # Clean up
        del df
        del self.samples

    @property
    def n_census_features(self):
        """Number of census features."""
        return len(self.census_cols)

    @property
    def n_other_ethnic_features(self):
        """Number of other ethnic features."""
        return len(self.other_ethnic_cols)

    def get_n_ethnicities(self):
        """Get number of unique ethnicities."""
        return self.n_ethnicities

    def get_census_cols(self):
        """Get list of census column names."""
        return self.census_cols

    def get_population_for_year(self, year_t, return_t1=False):
        """
        Get population tensor for a specific year.

        Args:
            year_t: The year_t value to filter by
            return_t1: If True, returns pop_t1 (next census), else pop_t

        Returns:
            Tensor of shape (n_nodes, n_ethnicities) with population values
        """
        import torch

        pop_tensor = torch.zeros(self.n_nodes, self.n_ethnicities, dtype=torch.float32)

        # Find samples for this year
        year_mask = self._year_t == year_t

        if not year_mask.any():
            return pop_tensor

        # Get relevant data
        node_indices = self._node_idx[year_mask]
        ethnicity_indices = self._ethnicity_idx[year_mask]
        populations = self._pop_t1[year_mask] if return_t1 else self._pop_t[year_mask]

        # Fill tensor (convert numpy values to Python float)
        for i in range(len(node_indices)):
            node_idx = int(node_indices[i])
            eth_idx = int(ethnicity_indices[i])
            pop_tensor[node_idx, eth_idx] = float(populations[i])

        return pop_tensor

    def get_unique_years(self):
        """Get list of unique year_t values."""
        return sorted(set(self._year_t))

    def _extract_data_to_numpy(self, df):
        """Pre-extract all data as numpy arrays for thread-safe access."""
        n_samples = len(df)

        # Node indices
        self._node_idx = df['node_idx'].values.astype(np.int64)

        # Census features (normalized)
        census_values = df[self.census_cols].values.astype(np.float32)
        census_values = np.nan_to_num(census_values, nan=0.0)
        self._census_normalized = (census_values - self.census_mean) / self.census_std
        self._census_normalized = np.clip(self._census_normalized, -5.0, 5.0)

        # Other ethnic features (raw)
        other_ethnic_values = df[self.other_ethnic_cols].values.astype(np.float32)
        self._other_ethnic = np.nan_to_num(other_ethnic_values, nan=0.0)

        # Populations
        self._pop_t = df['pop_t'].values.astype(np.float32)
        self._pop_t1 = df['pop_t1'].values.astype(np.float32)

        # Zero/sparse flags
        self._is_zero_t = (self._pop_t == 0.0).astype(np.float32)
        self._is_zero_t1 = (self._pop_t1 == 0.0).astype(np.float32)
        self._is_sparse = (self._pop_t1 < 10.0).astype(np.float32)

        # Ethnicity indices
        self._ethnicity_idx = np.array([self.ethnicity_map[eth] for eth in df['ethnicity']], dtype=np.int64)

        # Year and dauid
        self._year_t = df['year_t'].values
        self._dauid = df['dauid'].values

    def _filter_graph_to_cities(self, city_filter, exclude_cities, da_csv_path):
        """
        Filter graph to only include nodes belonging to specified cities.
        This is based on CITY membership, not on which samples exist.
        """
        from scipy import sparse

        # Load city mapping
        da_df = pd.read_csv(da_csv_path)
        da_df['DAUID'] = da_df['DAUID'].astype(str).str.strip()
        dauid_to_city = dict(zip(da_df['DAUID'], da_df['Name']))

        # Determine which cities to keep
        all_cities = set(dauid_to_city.values())
        if city_filter is not None:
            cities_to_keep = set(city_filter)
        else:
            cities_to_keep = all_cities

        if exclude_cities is not None:
            cities_to_keep = cities_to_keep - set(exclude_cities)

        # Find ALL node indices belonging to kept cities
        idx_to_dauid = {v: k for k, v in self.dauid_to_idx.items()}
        n_original = self.adjacency.shape[0]

        nodes_to_keep = []
        for idx in range(n_original):
            dauid = idx_to_dauid.get(idx)
            if dauid and dauid_to_city.get(dauid) in cities_to_keep:
                nodes_to_keep.append(idx)

        nodes_to_keep = sorted(nodes_to_keep)

        # Create old -> new index mapping
        old_to_new = {old: new for new, old in enumerate(nodes_to_keep)}

        # Filter adjacency
        adj_csr = self.adjacency.tocsr()
        self.adjacency = adj_csr[nodes_to_keep, :][:, nodes_to_keep].tocsr()

        # Filter coordinates
        if hasattr(self, 'coordinates'):
            self.coordinates = self.coordinates[nodes_to_keep]

        # Filter node_features
        if 'node_features' in self.graph:
            for key, values in self.graph['node_features'].items():
                if isinstance(values, np.ndarray) and len(values) == n_original:
                    self.graph['node_features'][key] = values[nodes_to_keep]
                elif isinstance(values, list) and len(values) == n_original:
                    self.graph['node_features'][key] = [values[i] for i in nodes_to_keep]

        # Update dauid_to_idx
        new_dauid_to_idx = {}
        for old_idx in nodes_to_keep:
            dauid = idx_to_dauid[old_idx]
            new_dauid_to_idx[dauid] = old_to_new[old_idx]
        self.dauid_to_idx = new_dauid_to_idx

        # Update graph object
        self.graph['adjacency'] = self.adjacency
        self.graph['dauid_to_idx'] = self.dauid_to_idx

        # Remap node_idx in samples
        self.samples['node_idx'] = self.samples['node_idx'].map(old_to_new)
        self.samples = self.samples.dropna(subset=['node_idx'])
        self.samples['node_idx'] = self.samples['node_idx'].astype(int)

    def _precompute_full_graph_states(self):
        """Precompute full graph population and census for each year"""
        self.full_graph_cache = {}

        unique_years = self.samples['year_t'].unique()
        n_census = len(self.census_cols)

        for year in unique_years:
            year_samples = self.samples[self.samples['year_t'] == year]

            # Initialize full graph tensors
            full_population = np.zeros((self.n_nodes, self.n_ethnicities), dtype=np.float32)
            full_census = np.zeros((self.n_nodes, n_census), dtype=np.float32)

            # Fill in actual values from all samples at this year
            for _, row in year_samples.iterrows():
                node_idx = int(row['node_idx'])
                ethnicity_idx = self.ethnicity_map[row['ethnicity']]

                # Population at year_t (raw, not normalized)
                full_population[node_idx, ethnicity_idx] = float(row['pop_t'])

                # Census features (normalized)
                census_values = row[self.census_cols].values.astype(np.float32)
                census_values = np.nan_to_num(census_values, nan=0.0)
                census_normalized = (census_values - self.census_mean) / self.census_std
                census_normalized = np.clip(census_normalized, -5.0, 5.0)
                full_census[node_idx] = census_normalized

            # Cache as tensors
            self.full_graph_cache[year] = {
                'population': torch.tensor(full_population, dtype=torch.float32),
                'census': torch.tensor(full_census, dtype=torch.float32)
            }

    def get_census_stats(self):
        """Return normalization statistics for use by val/test sets"""
        return {
            'mean': self.census_mean,
            'std': self.census_std
        }

    def get_city_statistics(self):
        """Return statistics about cities in this dataset"""
        # Build a minimal DataFrame for groupby
        stats_df = pd.DataFrame({
            'city_name': self._city_names,
            'dauid': self._dauids_for_stats,
            'pop_t': self._pop_t_for_stats,
            'pop_t1': self._pop_t1_for_stats
        })
        city_stats = stats_df.groupby('city_name').agg({
            'dauid': 'count',
            'pop_t': 'mean',
            'pop_t1': 'mean'
        }).rename(columns={'dauid': 'n_samples'})
        return city_stats

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """Thread-safe data access using pre-extracted numpy arrays."""
        # Node index
        node_idx = self._node_idx[idx]
        coords = self.coordinates[node_idx]
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        # Census features (already normalized)
        census_features = torch.from_numpy(self._census_normalized[idx])

        # Other ethnic features
        other_ethnic_features = torch.from_numpy(self._other_ethnic[idx])

        # Combine features
        features = torch.cat([census_features, other_ethnic_features])

        # Populations
        pop_t = torch.tensor(self._pop_t[idx], dtype=torch.float32)
        pop_t1 = torch.tensor(self._pop_t1[idx], dtype=torch.float32)

        # Zero/sparse flags
        is_zero_t = torch.tensor(self._is_zero_t[idx], dtype=torch.float32)
        is_zero_t1 = torch.tensor(self._is_zero_t1[idx], dtype=torch.float32)
        is_sparse = torch.tensor(self._is_sparse[idx], dtype=torch.float32)

        # Neighbor indices
        neighbors = torch.tensor(
            self.adjacency[node_idx].indices,
            dtype=torch.long
        )

        # Ethnicity
        ethnicity_idx = self._ethnicity_idx[idx]

        return {
            'node_idx': torch.tensor(node_idx, dtype=torch.long),
            'features': features,
            'pop_t': pop_t,
            'target': pop_t1,
            'is_zero_t': is_zero_t,
            'is_zero_t1': is_zero_t1,
            'is_sparse': is_sparse,
            'neighbors': neighbors,
            'ethnicity': torch.tensor(ethnicity_idx, dtype=torch.long),
            'year_t': self._year_t[idx],
            'dauid': self._dauid[idx],
            'coordinates': coords_tensor
        }


class GraphBatchCollator:
    """
    Picklable collate function for graph batches.
    Handles variable-length neighbor lists and builds full graph state.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        """
        Collate a batch of samples and add full graph state.
        """
        # Standard batch fields
        batch_dict = {
            'node_idx': torch.stack([b['node_idx'] for b in batch]),
            'features': torch.stack([b['features'] for b in batch]),
            'pop_t': torch.stack([b['pop_t'] for b in batch]),
            'target': torch.stack([b['target'] for b in batch]),
            'is_zero_t': torch.stack([b['is_zero_t'] for b in batch]),
            'is_zero_t1': torch.stack([b['is_zero_t1'] for b in batch]),
            'is_sparse': torch.stack([b['is_sparse'] for b in batch]),
            'neighbors': [b['neighbors'] for b in batch],
            'ethnicity': torch.stack([b['ethnicity'] for b in batch]),
            'year_t': [b['year_t'] for b in batch],
            'dauid': [b['dauid'] for b in batch],
            'coordinates': torch.stack([b['coordinates'] for b in batch])
        }

        # Get full graph state from cache
        year_t = batch[0]['year_t']

        full_graph_data = self.dataset.full_graph_cache[year_t]
        batch_dict['full_graph_population_t'] = full_graph_data['population'].clone()
        batch_dict['full_graph_census'] = full_graph_data['census'].clone()

        return batch_dict


def create_dataloaders_with_city_filter(
        train_path, val_path, test_path, graph_path,
        batch_size=512, num_workers=4, ethnicity_filter=None,
        # NEW: City filtering options
        city_filter=None, exclude_cities=None,
        min_city_r2=None, city_metrics_path=None,
        da_csv_path='./data/da_canada.csv'):
    """
    Create train/val/test dataloaders with city filtering capability.

    NEW City Filtering Options:
        city_filter: List of cities to INCLUDE (None = all)
        exclude_cities: List of cities to EXCLUDE (None = no exclusions)
        min_city_r2: Minimum R² threshold for cities (None = no threshold)
        city_metrics_path: Path to city metrics CSV (required if min_city_r2 is set)
        da_csv_path: Path to DA-to-city mapping CSV

    Example usage:
        # Exclude specific poorly-performing cities
        exclude_cities = ['Rocky View County', 'Gatineau', 'Longueuil']

        # Or filter by R² threshold
        min_city_r2 = 0.3
        city_metrics_path = './city_metrics.csv'

        # Or include only specific cities
        city_filter = ['Toronto', 'Vancouver', 'Montreal', 'Calgary']

    Returns:
        train_loader, val_loader, test_loader, graph
    """
    # Create TRAINING dataset first (computes normalization stats)
    train_dataset = ReactionDiffusionDatasetFiltered(
        train_path, graph_path, ethnicity_filter,
        census_stats=None,
        is_training=True,
        city_filter=city_filter,
        exclude_cities=exclude_cities,
        min_city_r2=min_city_r2,
        city_metrics_path=city_metrics_path,
        da_csv_path=da_csv_path
    )

    # Get normalization stats from training set
    census_stats = train_dataset.get_census_stats()

    # Create VAL and TEST datasets
    val_dataset = ReactionDiffusionDatasetFiltered(
        val_path, graph_path, ethnicity_filter,
        census_stats=census_stats,
        is_training=False,
        city_filter=city_filter,
        exclude_cities=exclude_cities,
        min_city_r2=min_city_r2,
        city_metrics_path=city_metrics_path,
        da_csv_path=da_csv_path
    )

    test_dataset = ReactionDiffusionDatasetFiltered(
        test_path, graph_path, ethnicity_filter,
        census_stats=census_stats,
        is_training=False,
        city_filter=city_filter,
        exclude_cities=exclude_cities,
        min_city_r2=min_city_r2,
        city_metrics_path=city_metrics_path,
        da_csv_path=da_csv_path
    )

    # Create picklable collate functions
    train_collate_fn = GraphBatchCollator(train_dataset)
    val_collate_fn = GraphBatchCollator(val_dataset)
    test_collate_fn = GraphBatchCollator(test_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader, train_dataset.graph


# Convenience function to get poorly performing cities from metrics
def get_cities_below_threshold(city_metrics_path, r2_threshold=0.0):
    """
    Helper function to get list of cities below R² threshold.

    Args:
        city_metrics_path: Path to city metrics CSV
        r2_threshold: R² threshold (default: 0.0 for negative R²)

    Returns:
        List of city names with R² < threshold
    """
    df = pd.read_csv(city_metrics_path)
    poor_cities = df[df['r2'] < r2_threshold]['city'].tolist()
    return poor_cities


# Example usage function
def example_usage():
    """
    Example of how to use the city filtering functionality.
    """
    print("EXAMPLE 1: Exclude cities with negative R²")
    print("-" * 80)

    poor_cities = get_cities_below_threshold('./city_metrics.csv', r2_threshold=0.0)
    print(f"Cities with negative R²: {poor_cities}")

    train_loader, val_loader, test_loader, graph = create_dataloaders_with_city_filter(
        train_path='./data/train_rd.pkl',
        val_path='./data/val_rd.pkl',
        test_path='./data/test_rd.pkl',
        graph_path='./data/graph_large_cities_rd.pkl',
        batch_size=512,
        exclude_cities=poor_cities
    )

    print("\n" + "="*80)
    print("EXAMPLE 2: Only include high-performing cities (R² > 0.5)")
    print("-" * 80)

    train_loader, val_loader, test_loader, graph = create_dataloaders_with_city_filter(
        train_path='./data/train_rd.pkl',
        val_path='./data/val_rd.pkl',
        test_path='./data/test_rd.pkl',
        graph_path='./data/graph_large_cities_rd.pkl',
        batch_size=512,
        min_city_r2=0.5,
        city_metrics_path='./city_metrics.csv'
    )

    print("\n" + "="*80)
    print("EXAMPLE 3: Focus on major metropolitan areas only")
    print("-" * 80)

    major_cities = ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Edmonton', 'Ottawa']

    train_loader, val_loader, test_loader, graph = create_dataloaders_with_city_filter(
        train_path='./data/train_rd.pkl',
        val_path='./data/val_rd.pkl',
        test_path='./data/test_rd.pkl',
        graph_path='./data/graph_large_cities_rd.pkl',
        batch_size=512,
        city_filter=major_cities
    )


if __name__ == "__main__":
    # Run examples if executed directly
    example_usage()
