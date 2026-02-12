"""
model_tgcn_baseline.py

Temporal Graph Convolutional Network (TGCN) baseline for multi-ethnic population prediction.
Combines GCN spatial layers with GRU temporal layers.

Reference: Zhao et al. (2020) "T-GCN: A Temporal Graph Convolutional Network for 
Traffic Prediction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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


class TGCNCell(nn.Module):
    """
    TGCN Cell: Combines GCN for spatial dependencies with GRU for temporal dynamics.
    
    At each time step:
    1. Apply GCN to capture spatial dependencies
    2. Feed into GRU to capture temporal evolution
    3. Output updated hidden state
    """
    def __init__(self, in_channels, hidden_channels, edge_index, edge_weight):
        super().__init__()
        
        # GCN for spatial aggregation
        self.gcn = GCNConv(in_channels, hidden_channels)
        
        # GRU for temporal evolution
        # Input: spatial features from GCN
        # Hidden: previous temporal state
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        
        # Store graph structure
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)
    
    def forward(self, x, h):
        """
        Forward pass through TGCN cell.
        
        Args:
            x: (n_nodes, in_channels) - input features at current time
            h: (n_nodes, hidden_channels) - hidden state from previous time
        
        Returns:
            h_new: (n_nodes, hidden_channels) - updated hidden state
        """
        # Apply GCN for spatial aggregation
        spatial_features = self.gcn(x, self.edge_index, self.edge_weight)
        
        # Apply GRU for temporal update
        h_new = self.gru(spatial_features, h)
        
        return h_new


class TemporalGraphConvolutionalLayers(nn.Module):
    """
    Stack of TGCN cells for deep spatial-temporal modeling.
    
    Each layer processes the sequence with its own TGCN cell.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, 
                 edge_index, edge_weight, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(dropout)
        
        # First TGCN layer
        self.tgcn_cells = nn.ModuleList()
        self.tgcn_cells.append(
            TGCNCell(in_channels, hidden_channels, edge_index, edge_weight)
        )
        
        # Additional TGCN layers
        for _ in range(num_layers - 1):
            self.tgcn_cells.append(
                TGCNCell(hidden_channels, hidden_channels, edge_index, edge_weight)
            )
        
        # Layer normalization for each layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
    
    def forward(self, x_sequence, h_init=None):
        """
        Process a sequence of spatial features through TGCN layers.
        
        Args:
            x_sequence: List of (n_nodes, in_channels) tensors, one per time step
            h_init: Optional initial hidden states for each layer
        
        Returns:
            output: (n_nodes, hidden_channels) - final hidden state
            hidden_states: List of hidden states for each layer
        """
        n_nodes = x_sequence[0].shape[0]
        
        # Initialize hidden states if not provided
        if h_init is None:
            h_init = [
                torch.zeros(n_nodes, self.hidden_channels, 
                           device=x_sequence[0].device)
                for _ in range(self.num_layers)
            ]
        
        # Process sequence through each layer
        current_sequence = x_sequence
        final_hidden_states = []
        
        for layer_idx, (tgcn_cell, norm) in enumerate(zip(self.tgcn_cells, self.norms)):
            h = h_init[layer_idx]
            layer_outputs = []
            
            # Process each time step
            for x_t in current_sequence:
                h = tgcn_cell(x_t, h)
                h = norm(h)
                h = self.dropout(h)
                layer_outputs.append(h)
            
            final_hidden_states.append(h)
            current_sequence = layer_outputs
        
        # Return final output from last layer, last time step
        return current_sequence[-1], final_hidden_states


class PopulationPredictor(nn.Module):
    """Predict population for a specific ethnicity given spatial-temporal context"""
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


class TGCNBaseline(nn.Module):
    """
    Complete TGCN baseline model for multi-ethnic population prediction.
    
    Architecture:
    1. Encode node features (census + aggregated population)
    2. Create temporal sequence from current and historical states
    3. Apply TGCN layers to capture spatial-temporal dependencies
    4. Embed ethnicity
    5. Predict population combining spatial-temporal context + ethnicity
    
    Difference from GCN:
    - Explicitly models temporal evolution with GRU
    - Captures both spatial (GCN) and temporal (RNN) dependencies
    - More expressive but more parameters
    """
    
    def __init__(
        self,
        n_census_features,
        n_ethnicities,
        adjacency,
        feature_encoder_dim=128,
        tgcn_hidden_dim=128,
        tgcn_num_layers=2,  # Fewer layers than GCN due to GRU complexity
        ethnicity_embed_dim=32,
        predictor_hidden_dim=128,
        dropout=0.1,
        sequence_length=1  # Number of historical time steps to use
    ):
        super().__init__()
        
        self.n_census_features = n_census_features
        self.n_ethnicities = n_ethnicities
        self.sequence_length = sequence_length
        
        # Convert adjacency to edge_index for PyTorch Geometric
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight.float())
        
        # Components
        self.feature_encoder = SpatialFeatureEncoder(
            n_census_features, feature_encoder_dim
        )
        
        self.tgcn = TemporalGraphConvolutionalLayers(
            in_channels=feature_encoder_dim,
            hidden_channels=tgcn_hidden_dim,
            num_layers=tgcn_num_layers,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            dropout=dropout
        )
        
        self.ethnicity_embedding = EthnicityEmbedding(
            n_ethnicities, ethnicity_embed_dim
        )
        
        self.predictor = PopulationPredictor(
            spatial_dim=tgcn_hidden_dim,
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
                - 'full_graph_population_t': (n_nodes, n_ethnicities) - populations at time t
        
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
        
        # Create temporal sequence
        # For now, we use a simple approach: repeat current state for sequence_length steps
        # In a more advanced version, you could use actual historical data
        x_sequence = [encoded_features for _ in range(self.sequence_length)]
        
        # Apply TGCN to get spatial-temporal context for all nodes
        spatial_temporal_context, _ = self.tgcn(x_sequence)  # (n_nodes, tgcn_dim)
        
        # Get spatial-temporal context for batch nodes
        node_idx = batch['node_idx']
        batch_spatial_temporal = spatial_temporal_context[node_idx]  # (batch_size, tgcn_dim)
        
        # Get ethnicity embeddings
        ethnicity_idx = batch['ethnicity']
        ethnicity_embed = self.ethnicity_embedding(ethnicity_idx)  # (batch_size, eth_dim)
        
        # Predict population
        predictions = self.predictor(batch_spatial_temporal, ethnicity_embed)  # (batch_size,)
        
        # Ensure non-negative predictions
        predictions = F.softplus(predictions)
        
        return predictions
    
    def get_spatial_temporal_embeddings(self, full_census, full_populations):
        """
        Get spatial-temporal embeddings for all nodes (for analysis/visualization).
        
        Args:
            full_census: (n_nodes, n_census) - census features for all nodes
            full_populations: (n_nodes, n_ethnicities) - populations for all nodes
        """
        # Combine census + aggregated population signal
        full_other_pop = full_populations.sum(dim=1, keepdim=True)
        full_features = torch.cat([full_census, full_other_pop], dim=1)
        
        encoded_features = self.feature_encoder(full_features)
        
        # Create sequence
        x_sequence = [encoded_features for _ in range(self.sequence_length)]
        
        spatial_temporal_context, _ = self.tgcn(x_sequence)
        return spatial_temporal_context


def create_tgcn_model(
    n_census_features,
    n_ethnicities,
    adjacency,
    feature_encoder_dim=128,
    tgcn_hidden_dim=128,
    tgcn_num_layers=2,
    ethnicity_embed_dim=32,
    predictor_hidden_dim=128,
    dropout=0.1,
    sequence_length=1
):
    """
    Factory function to create TGCN baseline model.
    
    Args:
        n_census_features: Number of census features (74)
        n_ethnicities: Number of ethnic groups (9)
        adjacency: Scipy sparse adjacency matrix
        feature_encoder_dim: Hidden dimension for feature encoder
        tgcn_hidden_dim: Hidden dimension for TGCN layers
        tgcn_num_layers: Number of TGCN layers (default 2, less than GCN due to GRU)
        ethnicity_embed_dim: Ethnicity embedding dimension
        predictor_hidden_dim: Hidden dimension for predictor
        dropout: Dropout rate
        sequence_length: Number of time steps in sequence
    
    Returns:
        TGCN model
    """
    model = TGCNBaseline(
        n_census_features=n_census_features,
        n_ethnicities=n_ethnicities,
        adjacency=adjacency,
        feature_encoder_dim=feature_encoder_dim,
        tgcn_hidden_dim=tgcn_hidden_dim,
        tgcn_num_layers=tgcn_num_layers,
        ethnicity_embed_dim=ethnicity_embed_dim,
        predictor_hidden_dim=predictor_hidden_dim,
        dropout=dropout,
        sequence_length=sequence_length
    )
    
    return model
