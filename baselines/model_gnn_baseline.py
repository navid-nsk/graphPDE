"""
model_gnn_baseline.py

Graph Neural Network baseline for multi-ethnic population prediction.
Uses Graph Attention Networks (GAT) to incorporate spatial structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np


class EthnicityEmbedding(nn.Module):
    """
    Embed ethnicity as a learnable vector.
    """
    def __init__(self, n_ethnicities, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_ethnicities, embedding_dim)
    
    def forward(self, ethnicity_idx):
        return self.embedding(ethnicity_idx)


class SpatialFeatureEncoder(nn.Module):
    """
    Encode node features (census + aggregated population signal).
    """
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


class GraphAttentionLayers(nn.Module):
    """
    Stack of Graph Attention Network layers.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Last layer (average attention heads)
        self.convs.append(
            GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=False)
        )
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection (if dimensions match)
            if i > 0 and x.shape == x_in.shape:
                x = x + x_in
        
        return x


class PopulationPredictor(nn.Module):
    """
    Predict population for a specific ethnicity given spatial context.
    """
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


class GNNBaseline(nn.Module):
    """
    Complete GNN baseline model for multi-ethnic population prediction.
    
    Architecture:
    1. Encode node features (census + other ethnicities)
    2. Apply Graph Attention layers to capture spatial dependencies
    3. Embed ethnicity
    4. Predict population combining spatial context + ethnicity
    """
    
    def __init__(
        self,
        n_census_features,
        n_ethnicities,
        adjacency,
        feature_encoder_dim=128,
        gnn_hidden_dim=128,
        gnn_num_layers=3,
        gnn_heads=4,
        ethnicity_embed_dim=32,
        predictor_hidden_dim=128,
        dropout=0.1
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
        
        self.gnn = GraphAttentionLayers(
            in_channels=feature_encoder_dim,
            hidden_channels=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            heads=gnn_heads,
            dropout=dropout
        )
        
        self.ethnicity_embedding = EthnicityEmbedding(
            n_ethnicities, ethnicity_embed_dim
        )
        
        self.predictor = PopulationPredictor(
            spatial_dim=gnn_hidden_dim,
            ethnicity_dim=ethnicity_embed_dim,
            hidden_dim=predictor_hidden_dim
        )
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary with keys from your dataloader:
                - 'features': (batch_size, n_features) - census + other ethnicities for batch
                - 'ethnicity': (batch_size,) - ethnicity index
                - 'node_idx': (batch_size,) - node indices in graph
                - 'full_graph_census': (n_nodes, n_census) - census features for ALL nodes
                - 'full_graph_population_t': (n_nodes, n_ethnicities) - populations for ALL nodes
        
        Returns:
            predictions: (batch_size,) - predicted populations
        """
        # Build full graph features from census + population
        # full_graph_census: (n_nodes, n_census)
        # full_graph_population_t: (n_nodes, n_ethnicities)
        full_census = batch['full_graph_census']  # (n_nodes, n_census)
        full_pop = batch['full_graph_population_t']  # (n_nodes, n_ethnicities)
        
        # For each node, we need census + other ethnicities (excluding the target ethnicity)
        # But GNN operates on all nodes simultaneously, so we use all populations as features
        # This is an approximation - in reality each sample excludes one ethnicity
        # For simplicity, we'll use the batch features which are already correctly formatted
        
        # Use batch features directly for the sampled nodes
        batch_size = batch['features'].shape[0]
        n_nodes = full_census.shape[0]
        
        # Create full graph features by combining census + all ethnic populations
        # Sum across ethnicities to get total "other" population signal
        full_other_pop = full_pop.sum(dim=1, keepdim=True)  # (n_nodes, 1)
        full_features = torch.cat([full_census, full_other_pop], dim=1)  # (n_nodes, n_census + 1)
        
        # Encode all node features
        encoded_features = self.feature_encoder(full_features)  # (n_nodes, feature_dim)
        
        # Apply GNN to get spatial context for all nodes
        spatial_context = self.gnn(encoded_features, self.edge_index)  # (n_nodes, gnn_dim)
        
        # Get spatial context for batch nodes
        node_idx = batch['node_idx']
        batch_spatial = spatial_context[node_idx]  # (batch_size, gnn_dim)
        
        # Get ethnicity embeddings
        ethnicity_idx = batch['ethnicity']  # Note: your batch uses 'ethnicity' not 'ethnicity_idx'
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
        spatial_context = self.gnn(encoded_features, self.edge_index)
        return spatial_context


def create_gnn_model(
    n_census_features,
    n_ethnicities,
    adjacency,
    feature_encoder_dim=128,
    gnn_hidden_dim=128,
    gnn_num_layers=3,
    gnn_heads=4,
    ethnicity_embed_dim=32,
    predictor_hidden_dim=128,
    dropout=0.1
):
    """
    Factory function to create GNN baseline model.
    
    Args:
        n_census_features: Number of census features (74)
        n_ethnicities: Number of ethnic groups (9)
        adjacency: Scipy sparse adjacency matrix
        feature_encoder_dim: Hidden dimension for feature encoder
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_num_layers: Number of GNN layers
        gnn_heads: Number of attention heads
        ethnicity_embed_dim: Ethnicity embedding dimension
        predictor_hidden_dim: Hidden dimension for predictor
        dropout: Dropout rate
    
    Returns:
        GNN model
    """
    model = GNNBaseline(
        n_census_features=n_census_features,
        n_ethnicities=n_ethnicities,
        adjacency=adjacency,
        feature_encoder_dim=feature_encoder_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_heads=gnn_heads,
        ethnicity_embed_dim=ethnicity_embed_dim,
        predictor_hidden_dim=predictor_hidden_dim,
        dropout=dropout
    )
    
    return model