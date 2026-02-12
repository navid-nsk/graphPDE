"""
model_deeponet_baseline.py

DeepONet (Deep Operator Network) baseline for multi-ethnic population prediction.
Based on Lu et al. (2021) "Learning nonlinear operators via DeepONet"

DeepONet learns operators that map between function spaces:
G: (u(x), y) -> s(y)

In our context:
- Branch net: encodes spatial context (census features, neighbor populations)
- Trunk net: encodes query location and ethnicity
- Output: predicted population for that ethnicity at that location
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BranchNet(nn.Module):
    """
    Branch network: encodes the input function (spatial context).
    
    Takes as input the "sensors" - features observed at the location:
    - Census features (74 features)
    - Aggregated population signal from neighbors
    - Spatial coordinates (lat, lon)
    
    Outputs: latent encoding of the spatial context
    """
    def __init__(self, n_input_features, hidden_dims=[256, 256, 256], output_dim=128, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = n_input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_input_features) - spatial context features
        Returns:
            branch_output: (batch_size, output_dim) - latent encoding
        """
        return self.network(x)


class TrunkNet(nn.Module):
    """
    Trunk network: encodes the query coordinates.
    
    Takes as input:
    - Ethnicity embedding (which population are we predicting?)
    - Spatial coordinates (optional - already encoded in branch)
    - Time step (for temporal prediction, if needed)
    
    Outputs: basis functions in the same latent space as branch
    """
    def __init__(self, n_ethnicities, ethnicity_embed_dim=32, 
                 hidden_dims=[128, 128, 128], output_dim=128, dropout=0.1):
        super().__init__()
        
        # Ethnicity embedding
        self.ethnicity_embedding = nn.Embedding(n_ethnicities, ethnicity_embed_dim)
        
        # Trunk network
        layers = []
        prev_dim = ethnicity_embed_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer to latent space (same dim as branch output)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, ethnicity_idx):
        """
        Args:
            ethnicity_idx: (batch_size,) - ethnicity indices
        Returns:
            trunk_output: (batch_size, output_dim) - basis functions
        """
        ethnicity_embed = self.ethnicity_embedding(ethnicity_idx)
        return self.network(ethnicity_embed)


class DeepONet(nn.Module):
    """
    Complete DeepONet model for operator learning.
    
    Architecture:
    1. Branch net encodes spatial context u(x)
    2. Trunk net encodes query coordinates y
    3. Inner product + bias gives prediction G(u)(y)
    
    For our problem:
    - u(x) = spatial context at location x (census, populations, coordinates)
    - y = (ethnicity, location) - what we're querying
    - G(u)(y) = predicted population of ethnicity at that location
    """
    
    def __init__(
        self,
        n_census_features,
        n_ethnicities,
        branch_hidden_dims=[256, 256, 256],
        trunk_hidden_dims=[128, 128, 128],
        latent_dim=128,
        ethnicity_embed_dim=32,
        dropout=0.1,
        use_bias=True
    ):
        super().__init__()
        
        self.n_census_features = n_census_features
        self.n_ethnicities = n_ethnicities
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        
        # Branch net input: census (74) + aggregated pop (1) + coordinates (2) = 77
        branch_input_dim = n_census_features + 1 + 2
        
        self.branch_net = BranchNet(
            n_input_features=branch_input_dim,
            hidden_dims=branch_hidden_dims,
            output_dim=latent_dim,
            dropout=dropout
        )
        
        self.trunk_net = TrunkNet(
            n_ethnicities=n_ethnicities,
            ethnicity_embed_dim=ethnicity_embed_dim,
            hidden_dims=trunk_hidden_dims,
            output_dim=latent_dim,
            dropout=dropout
        )
        
        # Optional bias term
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, batch):
        """
        Forward pass using DeepONet architecture.
        
        Args:
            batch: Dictionary with keys:
                - 'features': (batch_size, n_features) - contains census + other populations
                - 'ethnicity': (batch_size,) - ethnicity index to predict
                - 'coordinates': (batch_size, 2) - (lat, lon) coordinates
                - 'node_idx': (batch_size,) - node indices
        
        Returns:
            predictions: (batch_size,) - predicted populations
        """
        # Extract features for branch net
        census_features = batch['features'][:, :self.n_census_features]
        
        # Aggregate population signal (sum of other ethnicities)
        # Features are: [census (74), other_ethnicities (8)]
        other_pop_features = batch['features'][:, self.n_census_features:]
        aggregated_pop = other_pop_features.sum(dim=1, keepdim=True)
        
        # Get coordinates
        coordinates = batch['coordinates']
        
        # Branch input: [census, aggregated_pop, coordinates]
        branch_input = torch.cat([census_features, aggregated_pop, coordinates], dim=1)
        
        # Branch net: encode spatial context
        branch_output = self.branch_net(branch_input)  # (batch_size, latent_dim)
        
        # Trunk net: encode query (ethnicity)
        ethnicity_idx = batch['ethnicity']
        trunk_output = self.trunk_net(ethnicity_idx)  # (batch_size, latent_dim)
        
        # DeepONet operation: inner product of branch and trunk
        # This is the key operation that learns the operator
        predictions = torch.sum(branch_output * trunk_output, dim=1)  # (batch_size,)
        
        # Add bias if used
        if self.use_bias:
            predictions = predictions + self.bias
        
        # Ensure non-negative predictions
        predictions = F.softplus(predictions)
        
        return predictions


class NeighborhoodAggregator(nn.Module):
    """
    Optional: Aggregate information from spatial neighbors before DeepONet.
    This can capture local spatial context more explicitly.
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        
        self.aggregator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, node_features, adjacency, node_indices):
        """
        Aggregate features from neighbors.
        
        Args:
            node_features: (n_nodes, feature_dim)
            adjacency: Sparse adjacency matrix
            node_indices: (batch_size,) - which nodes to aggregate for
        """
        # Get neighbors for each node in batch
        # Simple mean aggregation
        aggregated = []
        for idx in node_indices:
            neighbors = adjacency[idx].indices
            if len(neighbors) > 0:
                neighbor_features = node_features[neighbors].mean(dim=0)
            else:
                neighbor_features = node_features[idx]
            aggregated.append(neighbor_features)
        
        aggregated = torch.stack(aggregated)
        return self.aggregator(aggregated)


class DeepONetWithSpatialContext(nn.Module):
    """
    Enhanced DeepONet that explicitly incorporates spatial neighborhood context.
    Useful when spatial adjacency is important.
    """
    
    def __init__(
        self,
        n_census_features,
        n_ethnicities,
        adjacency,
        branch_hidden_dims=[256, 256, 256],
        trunk_hidden_dims=[128, 128, 128],
        latent_dim=128,
        ethnicity_embed_dim=32,
        dropout=0.1,
        use_neighborhood_aggregation=True
    ):
        super().__init__()
        
        self.n_census_features = n_census_features
        self.n_ethnicities = n_ethnicities
        self.use_neighborhood_aggregation = use_neighborhood_aggregation
        
        # Store adjacency as sparse tensor
        from torch_geometric.utils import from_scipy_sparse_matrix
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight.float())
        
        # Optional neighborhood aggregator
        if use_neighborhood_aggregation:
            feature_dim = n_census_features + 1  # census + aggregated pop
            self.neighborhood_aggregator = NeighborhoodAggregator(
                feature_dim=feature_dim,
                hidden_dim=128
            )
        
        # Core DeepONet
        self.deeponet = DeepONet(
            n_census_features=n_census_features,
            n_ethnicities=n_ethnicities,
            branch_hidden_dims=branch_hidden_dims,
            trunk_hidden_dims=trunk_hidden_dims,
            latent_dim=latent_dim,
            ethnicity_embed_dim=ethnicity_embed_dim,
            dropout=dropout
        )
    
    def forward(self, batch):
        """Forward pass with optional spatial context aggregation"""
        
        if self.use_neighborhood_aggregation:
            # Build full graph features
            full_census = batch['full_graph_census']
            full_pop = batch['full_graph_population_t']
            full_aggregated_pop = full_pop.sum(dim=1, keepdim=True)
            full_features = torch.cat([full_census, full_aggregated_pop], dim=1)
            
            # Aggregate neighborhood context
            node_idx = batch['node_idx']
            neighborhood_context = self.neighborhood_aggregator(
                full_features, 
                self.edge_index,
                node_idx
            )
            
            # Add neighborhood context to batch features
            # This enriches the spatial context for the branch network
            original_features = batch['features']
            enriched_features = torch.cat([original_features, neighborhood_context], dim=1)
            
            # Create enriched batch
            enriched_batch = batch.copy()
            enriched_batch['features'] = enriched_features
            
            return self.deeponet(enriched_batch)
        else:
            return self.deeponet(batch)


def create_deeponet_model(
    n_census_features,
    n_ethnicities,
    adjacency=None,
    branch_hidden_dims=[256, 256, 256],
    trunk_hidden_dims=[128, 128, 128],
    latent_dim=128,
    ethnicity_embed_dim=32,
    dropout=0.1,
    use_spatial_context=False
):
    """
    Factory function to create DeepONet model.
    
    Args:
        n_census_features: Number of census features (74)
        n_ethnicities: Number of ethnic groups (9)
        adjacency: Scipy sparse adjacency matrix (optional, for spatial version)
        branch_hidden_dims: Hidden dimensions for branch network
        trunk_hidden_dims: Hidden dimensions for trunk network
        latent_dim: Dimension of latent space (must match for branch & trunk)
        ethnicity_embed_dim: Ethnicity embedding dimension
        dropout: Dropout rate
        use_spatial_context: Whether to use spatial neighborhood aggregation
    
    Returns:
        DeepONet model
    """
    if use_spatial_context and adjacency is not None:
        model = DeepONetWithSpatialContext(
            n_census_features=n_census_features,
            n_ethnicities=n_ethnicities,
            adjacency=adjacency,
            branch_hidden_dims=branch_hidden_dims,
            trunk_hidden_dims=trunk_hidden_dims,
            latent_dim=latent_dim,
            ethnicity_embed_dim=ethnicity_embed_dim,
            dropout=dropout,
            use_neighborhood_aggregation=True
        )
    else:
        model = DeepONet(
            n_census_features=n_census_features,
            n_ethnicities=n_ethnicities,
            branch_hidden_dims=branch_hidden_dims,
            trunk_hidden_dims=trunk_hidden_dims,
            latent_dim=latent_dim,
            ethnicity_embed_dim=ethnicity_embed_dim,
            dropout=dropout
        )
    
    return model
