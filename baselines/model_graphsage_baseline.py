"""
model_graphsage_baseline.py

GraphSAGE baseline for multi-ethnic population prediction.
Uses GraphSAGE layers (Hamilton et al. 2017) with mean aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np


class EthnicityEmbedding(nn.Module):
    """Embed ethnicity as a learnable vector"""
    def __init__(self, n_ethnicities, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_ethnicities, embedding_dim)
    
    def forward(self, ethnicity_idx):
        return self.embedding(ethnicity_idx)


class SpatialFeatureEncoder(nn.Module):
    """Encode node features (census + aggregated population signal)"""
    def __init__(self, n_census_features, hidden_dim=128):
        super().__init__()
        
        # Input is census features + 1 (aggregated population signal)
        input_dim = n_census_features + 1
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, features):
        return self.encoder(features)


class GraphSAGELayers(nn.Module):
    """
    Stack of GraphSAGE layers (Hamilton et al. 2017).
    Uses mean aggregation by default.
    
    Key differences from GCN:
    - Samples and aggregates from neighborhoods
    - Concatenates self-features with aggregated neighbor features
    - More scalable for large graphs
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.1, aggr='mean'):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Last layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            # SAGEConv doesn't use edge_weight in standard implementation
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection (if dimensions match)
            if i > 0 and x.shape == x_in.shape:
                x = x + x_in
        
        return x


class PopulationPredictor(nn.Module):
    """Predict population for a specific ethnicity given spatial context"""
    def __init__(self, spatial_dim, ethnicity_dim, hidden_dim=128):
        super().__init__()
        
        input_dim = spatial_dim + ethnicity_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, spatial_features, ethnicity_embedding):
        combined = torch.cat([spatial_features, ethnicity_embedding], dim=-1)
        pred = self.predictor(combined)
        return pred.squeeze(-1)


class GraphSAGEBaseline(nn.Module):
    """
    Complete GraphSAGE baseline model for multi-ethnic population prediction.
    
    Architecture:
    1. Encode node features (census + aggregated population)
    2. Apply GraphSAGE layers to capture spatial dependencies via sampling and aggregation
    3. Embed ethnicity
    4. Predict population combining spatial context + ethnicity
    
    Difference from GCN:
    - Uses sample-and-aggregate approach (more scalable)
    - Concatenates self-features with aggregated neighbors
    - Better for inductive learning on unseen nodes
    
    Difference from GAT:
    - No attention mechanism
    - Fixed aggregation function (mean/max/LSTM)
    - More efficient but less adaptive
    """
    
    def __init__(
        self,
        n_census_features,
        n_ethnicities,
        adjacency,
        feature_encoder_dim=128,
        sage_hidden_dim=128,
        sage_num_layers=3,
        ethnicity_embed_dim=32,
        predictor_hidden_dim=128,
        dropout=0.1,
        aggr='mean'  # 'mean', 'max', or 'lstm'
    ):
        super().__init__()
        
        self.n_census_features = n_census_features
        self.n_ethnicities = n_ethnicities
        
        # Convert adjacency to edge_index for PyTorch Geometric
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight.float())
        
        # Components
        self.feature_encoder = SpatialFeatureEncoder(
            n_census_features, feature_encoder_dim
        )
        
        self.sage = GraphSAGELayers(
            in_channels=feature_encoder_dim,
            hidden_channels=sage_hidden_dim,
            num_layers=sage_num_layers,
            dropout=dropout,
            aggr=aggr
        )
        
        self.ethnicity_embedding = EthnicityEmbedding(
            n_ethnicities, ethnicity_embed_dim
        )
        
        self.predictor = PopulationPredictor(
            spatial_dim=sage_hidden_dim,
            ethnicity_dim=ethnicity_embed_dim,
            hidden_dim=predictor_hidden_dim
        )
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary with keys:
                - 'features': (batch_size, n_features) - census + other ethnicities
                - 'ethnicity': (batch_size,) - ethnicity index
                - 'node_idx': (batch_size,) - node indices in graph
                - 'full_graph_census': (n_nodes, n_census) - census for all nodes
                - 'full_graph_population_t': (n_nodes, n_ethnicities) - populations for all nodes
        
        Returns:
            predictions: (batch_size,) - predicted populations
        """
        # Build full graph features from census + population
        full_census = batch['full_graph_census']  # (n_nodes, n_census)
        full_pop = batch['full_graph_population_t']  # (n_nodes, n_ethnicities)
        
        # Aggregate population signal
        full_other_pop = full_pop.sum(dim=1, keepdim=True)  # (n_nodes, 1)
        full_features = torch.cat([full_census, full_other_pop], dim=1)  # (n_nodes, n_census + 1)
        
        # Encode all node features
        encoded_features = self.feature_encoder(full_features)  # (n_nodes, feature_dim)
        
        # Apply GraphSAGE to get spatial context for all nodes
        spatial_context = self.sage(encoded_features, self.edge_index, self.edge_weight)  # (n_nodes, sage_dim)
        
        # Get spatial context for batch nodes
        node_idx = batch['node_idx']
        batch_spatial = spatial_context[node_idx]  # (batch_size, sage_dim)
        
        # Get ethnicity embeddings
        ethnicity_idx = batch['ethnicity']
        ethnicity_embed = self.ethnicity_embedding(ethnicity_idx)  # (batch_size, eth_dim)
        
        # Predict population
        predictions = self.predictor(batch_spatial, ethnicity_embed)  # (batch_size,)
        
        # Ensure non-negative predictions
        predictions = F.softplus(predictions)
        
        return predictions
    
    def get_spatial_embeddings(self, full_census, full_populations):
        """
        Get spatial embeddings for all nodes (for analysis/visualization).
        
        Args:
            full_census: (n_nodes, n_census) - census features for all nodes
            full_populations: (n_nodes, n_ethnicities) - populations for all nodes
        """
        # Combine census + aggregated population signal
        full_other_pop = full_populations.sum(dim=1, keepdim=True)
        full_features = torch.cat([full_census, full_other_pop], dim=1)
        
        encoded_features = self.feature_encoder(full_features)
        spatial_context = self.sage(encoded_features, self.edge_index, self.edge_weight)
        return spatial_context


def create_graphsage_model(
    n_census_features,
    n_ethnicities,
    adjacency,
    feature_encoder_dim=128,
    sage_hidden_dim=128,
    sage_num_layers=3,
    ethnicity_embed_dim=32,
    predictor_hidden_dim=128,
    dropout=0.1,
    aggr='mean'
):
    """
    Factory function to create GraphSAGE baseline model.
    
    Args:
        n_census_features: Number of census features (74)
        n_ethnicities: Number of ethnic groups (9)
        adjacency: Scipy sparse adjacency matrix
        feature_encoder_dim: Hidden dimension for feature encoder
        sage_hidden_dim: Hidden dimension for GraphSAGE layers
        sage_num_layers: Number of GraphSAGE layers
        ethnicity_embed_dim: Ethnicity embedding dimension
        predictor_hidden_dim: Hidden dimension for predictor
        dropout: Dropout rate
        aggr: Aggregation function ('mean', 'max', or 'lstm')
    
    Returns:
        GraphSAGE model
    """
    model = GraphSAGEBaseline(
        n_census_features=n_census_features,
        n_ethnicities=n_ethnicities,
        adjacency=adjacency,
        feature_encoder_dim=feature_encoder_dim,
        sage_hidden_dim=sage_hidden_dim,
        sage_num_layers=sage_num_layers,
        ethnicity_embed_dim=ethnicity_embed_dim,
        predictor_hidden_dim=predictor_hidden_dim,
        dropout=dropout,
        aggr=aggr
    )
    
    return model
