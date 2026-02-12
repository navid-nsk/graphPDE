"""
model_dcrnn.py

Diffusion Convolutional Recurrent Neural Network (DCRNN) baseline
Paper: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"

Pure data-driven spatiotemporal model that learns diffusion on graphs
without physics priors - direct comparison to GraphPDE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionGraphConv(nn.Module):
    """
    Diffusion graph convolution - SPARSE VERSION
    Keeps adjacency matrix sparse throughout computation
    """
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.K = K
        
        # Learnable weights for each diffusion step
        self.weight_forward = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.weight_backward = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_forward)
        nn.init.xavier_uniform_(self.weight_backward)
        nn.init.zeros_(self.bias)
    
    def _normalize_adj_sparse(self, adj_sparse):
        """
        Compute D^{-1}A in sparse format
        
        Args:
            adj_sparse: (num_nodes, num_nodes) sparse tensor
        Returns:
            trans_forward: D^{-1}A (sparse)
            trans_backward: (D^{-1}A)^T (sparse)
        """
        # Get number of nodes
        n_nodes = adj_sparse.size(0)
        
        # Compute degree: sum of each row (sparse sum)
        # For sparse tensor, we sum the values for each row
        indices = adj_sparse.indices()
        values = adj_sparse.values()
        
        # Degree computation
        degree = torch.zeros(n_nodes, device=adj_sparse.device, dtype=adj_sparse.dtype)
        degree.scatter_add_(0, indices[0], values)
        degree = degree + 1e-8  # Avoid division by zero
        
        # Create D^{-1} and multiply: D^{-1} @ A
        # Since D is diagonal, we can just divide each row's values
        degree_inv = 1.0 / degree
        
        # Normalize: multiply values by 1/degree of source node
        normalized_values = values * degree_inv[indices[0]]
        
        # Create normalized adjacency (forward)
        trans_forward = torch.sparse_coo_tensor(
            indices, 
            normalized_values,
            size=adj_sparse.size(),
            device=adj_sparse.device
        ).coalesce()
        
        # Transpose for backward diffusion
        trans_backward = trans_forward.t().coalesce()
        
        return trans_forward, trans_backward
    
    def forward(self, x, adj_matrix):
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
            adj_matrix: (num_nodes, num_nodes) SPARSE tensor
        Returns:
            (batch_size, num_nodes, out_channels)
        """
        batch_size, num_nodes, in_channels = x.shape
        
        # Normalize adjacency matrices (stays sparse)
        trans_forward, trans_backward = self._normalize_adj_sparse(adj_matrix)
        
        outputs = []
        
        # Forward diffusion (K steps)
        x_forward = x  # (batch, nodes, features)
        for k in range(self.K):
            # Sparse matrix multiplication
            # We need to do: (nodes, nodes) @ (batch, nodes, features)
            # Reshape to (batch * nodes, features) for matmul
            x_forward_flat = x_forward.reshape(batch_size * num_nodes, in_channels)
            
            # Diffusion: A @ X for each batch element
            diffused = []
            for b in range(batch_size):
                x_b = x_forward[b]  # (nodes, features)
                # Sparse @ Dense = Dense
                x_diffused = torch.sparse.mm(trans_forward, x_b)  # (nodes, features)
                diffused.append(x_diffused)
            
            x_forward = torch.stack(diffused, dim=0)  # (batch, nodes, features)
            
            # Transform with learned weights
            out = torch.einsum('bni,io->bno', x_forward, self.weight_forward[k])
            outputs.append(out)
        
        # Backward diffusion (K steps)
        x_backward = x
        for k in range(self.K):
            diffused = []
            for b in range(batch_size):
                x_b = x_backward[b]
                x_diffused = torch.sparse.mm(trans_backward, x_b)
                diffused.append(x_diffused)
            
            x_backward = torch.stack(diffused, dim=0)
            
            out = torch.einsum('bni,io->bno', x_backward, self.weight_backward[k])
            outputs.append(out)
        
        # Sum all diffusion steps and add bias
        output = torch.stack(outputs, dim=0).sum(dim=0) + self.bias
        
        return output


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.
    
    Combines graph diffusion with GRU recurrence for spatiotemporal modeling.
    """
    def __init__(self, input_dim, hidden_dim, K=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Gates use diffusion convolution
        self.graph_conv_reset = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, K)
        self.graph_conv_update = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, K)
        self.graph_conv_candidate = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, K)
    
    def forward(self, x, h, adj_matrix):
        """
        Args:
            x: (batch_size, num_nodes, input_dim) - current input
            h: (batch_size, num_nodes, hidden_dim) - previous hidden state
            adj_matrix: (num_nodes, num_nodes) - graph adjacency
        Returns:
            h_new: (batch_size, num_nodes, hidden_dim)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)
        
        # Reset gate
        r = torch.sigmoid(self.graph_conv_reset(combined, adj_matrix))
        
        # Update gate
        u = torch.sigmoid(self.graph_conv_update(combined, adj_matrix))
        
        # Candidate hidden state (with reset gate applied)
        combined_r = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.graph_conv_candidate(combined_r, adj_matrix))
        
        # New hidden state
        h_new = u * h + (1 - u) * c
        
        return h_new


class DCRNNEncoder(nn.Module):
    """
    DCRNN Encoder: Processes historical sequence.
    
    In our case, we have single timestep input (current population),
    but architecture is designed for sequences.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, K=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Stack of DCGRU cells
        self.dcgru_cells = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim
            self.dcgru_cells.append(DCGRUCell(input_size, hidden_dim, K))
    
    def forward(self, x, adj_matrix, hidden_states=None):
        """
        Args:
            x: (batch_size, num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes)
            hidden_states: List of (batch_size, num_nodes, hidden_dim) or None
        Returns:
            output: (batch_size, num_nodes, hidden_dim)
            hidden_states: List of final hidden states
        """
        batch_size, num_nodes, _ = x.shape
        
        # Initialize hidden states if needed
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, num_nodes, self.hidden_dim, 
                          device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
        
        # Process through layers
        current_input = x
        new_hidden_states = []
        
        for layer_idx, dcgru_cell in enumerate(self.dcgru_cells):
            h = hidden_states[layer_idx]
            h_new = dcgru_cell(current_input, h, adj_matrix)
            new_hidden_states.append(h_new)
            current_input = h_new
        
        return current_input, new_hidden_states


class DCRNNDecoder(nn.Module):
    """
    DCRNN Decoder: Generates predictions.
    
    Uses encoder's final state to predict future populations.
    """
    def __init__(self, output_dim, hidden_dim, num_layers=2, K=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Stack of DCGRU cells
        self.dcgru_cells = nn.ModuleList()
        for i in range(num_layers):
            # Decoder takes its own previous output as input
            self.dcgru_cells.append(DCGRUCell(output_dim, hidden_dim, K))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, initial_input, adj_matrix, hidden_states, horizon=1):
        """
        Args:
            initial_input: (batch_size, num_nodes, output_dim) - start prediction
            adj_matrix: (num_nodes, num_nodes)
            hidden_states: List of hidden states from encoder
            horizon: Number of future steps to predict
        Returns:
            outputs: (batch_size, num_nodes, output_dim)
        """
        batch_size, num_nodes, output_dim = initial_input.shape
        
        # For our task, horizon=1 (predict single future snapshot)
        current_input = initial_input
        
        for step in range(horizon):
            # Process through DCGRU layers
            for layer_idx, dcgru_cell in enumerate(self.dcgru_cells):
                h = hidden_states[layer_idx]
                h_new = dcgru_cell(current_input, h, adj_matrix)
                hidden_states[layer_idx] = h_new
                current_input = h_new
            
            # Project to output space
            output = self.output_proj(current_input)
        
        return output


class DCRNN(nn.Module):
    """
    Complete DCRNN model for ethnic population prediction.
    
    Key differences from GraphPDE:
    - Pure data-driven (no physics equations)
    - Learns diffusion patterns from data
    - Recurrent architecture for temporal modeling
    - No interpretable parameters (black box)
    
    Comparison points:
    - Both use graph structure
    - Both model spatial diffusion
    - GraphPDE has physics constraints, DCRNN learns freely
    """
    def __init__(
        self,
        n_ethnicities,
        n_cities,
        n_dauids,
        n_census_features,
        adjacency,
        node_to_city,
        hidden_dim=64,
        num_layers=2,
        K=2
    ):
        super().__init__()
        self.n_ethnicities = n_ethnicities
        self.n_dauids = n_dauids
        self.hidden_dim = hidden_dim
        
        # Store adjacency matrix
        self.register_buffer('adjacency', self._prepare_adjacency(adjacency))
        
        # Input dimension: current population + census features
        # For each node, we have n_ethnicities populations + census features
        # But we predict per-node-ethnicity, so input is per-node
        self.input_dim = n_ethnicities + n_census_features
        
        # Output dimension: single ethnicity prediction per forward pass
        self.output_dim = 1
        
        # Feature encoder: Map input features to hidden dimension
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder: Process current state
        self.encoder = DCRNNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
        
        # Decoder: Generate predictions
        self.decoder = DCRNNDecoder(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _prepare_adjacency(self, adjacency):
        """Convert scipy sparse to torch sparse tensor"""
        if hasattr(adjacency, 'tocoo'):
            coo = adjacency.tocoo()
            indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            adj_sparse = torch.sparse_coo_tensor(
                indices, values, size=adjacency.shape
            )
            return adj_sparse.coalesce()
        return adjacency
    
    def forward(self, batch, node_to_city_batch, debug_batch=False):
        """
        Args:
            batch: Dictionary with:
                - full_graph_population_t: (n_dauids, n_ethnicities)
                - full_graph_census: (n_dauids, n_census_features)
                - node_idx: (batch_size,) - which nodes we're predicting
                - ethnicity: (batch_size,) - which ethnicity per node
        Returns:
            dict with 'pop_pred': (batch_size,) predictions
        """
        device = batch['pop_t'].device
        
        # Get full graph state
        full_population = batch['full_graph_population_t'].to(device)  # (n_dauids, n_eth)
        full_census = batch['full_graph_census'].to(device)  # (n_dauids, n_features)
        
        node_indices = batch['node_idx']
        ethnicity_indices = batch['ethnicity']
        
        n_nodes = full_population.shape[0]
        
        # Combine population and census features
        # (n_nodes, n_eth + n_features)
        full_features = torch.cat([full_population, full_census], dim=-1)
        
        # Add batch dimension: (1, n_nodes, features)
        full_features = full_features.unsqueeze(0)
        
        # Encode features
        encoded_features = self.feature_encoder(full_features)  # (1, n_nodes, hidden_dim)
        
        # Encode with DCGRU
        encoder_output, hidden_states = self.encoder(
            encoded_features, 
            self.adjacency
        )  # (1, n_nodes, hidden_dim)
        
        # Decode to generate predictions
        # Use encoder output as initial decoder input
        decoder_output = self.decoder(
            encoder_output,
            self.adjacency,
            hidden_states,
            horizon=1
        )  # (1, n_nodes, hidden_dim)
        
        # Remove batch dimension
        decoder_output = decoder_output.squeeze(0)  # (n_nodes, hidden_dim)
        
        # Generate predictions for all nodes and ethnicities
        node_predictions = self.prediction_head(decoder_output)  # (n_nodes, 1)
        
        # Extract predictions for the specific (node, ethnicity) pairs in batch
        # Since we predict per-node, we need to handle ethnicity-specific predictions
        # For simplicity, we'll use the same prediction for all ethnicities of a node
        # (In practice, you might want separate heads per ethnicity)
        predictions = node_predictions[node_indices].squeeze(-1)  # (batch_size,)
        
        # Ensure non-negative predictions
        predictions = F.relu(predictions)
        
        if debug_batch:
            print(f"\nDCRNN Forward Pass:")
            print(f"  Input features shape: {full_features.shape}")
            print(f"  Encoder output shape: {encoder_output.shape}")
            print(f"  Predictions: mean={predictions.mean():.2f}, max={predictions.max():.2f}")
        
        return {'pop_pred': predictions}


class DCRNNMultiEthnicity(nn.Module):
    """
    Enhanced DCRNN with separate prediction heads per ethnicity.
    
    This version is more comparable to GraphPDE's multi-ethnicity modeling.
    """
    def __init__(
        self,
        n_ethnicities,
        n_cities,
        n_dauids,
        n_census_features,
        adjacency,
        node_to_city,
        hidden_dim=64,
        num_layers=2,
        K=2
    ):
        super().__init__()
        self.n_ethnicities = n_ethnicities
        self.n_dauids = n_dauids
        self.hidden_dim = hidden_dim
        
        # Store adjacency matrix
        self.register_buffer('adjacency', self._prepare_adjacency(adjacency))
        
        # Input: all ethnicities + census features
        self.input_dim = n_ethnicities + n_census_features
        
        # Feature encoder - processes node features
        # Input: (batch=1, n_nodes, input_dim)
        # We need to encode each node's features
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm for graph data
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Encoder
        self.encoder = DCRNNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
        
        # Decoder
        self.decoder = DCRNNDecoder(
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
        
        # Separate prediction heads for each ethnicity
        self.ethnicity_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(n_ethnicities)
        ])
    
    def _prepare_adjacency(self, adjacency):
        """Convert scipy sparse to torch sparse tensor"""
        if hasattr(adjacency, 'tocoo'):
            coo = adjacency.tocoo()
            indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
            values = torch.tensor(coo.data, dtype=torch.float32)
            adj_sparse = torch.sparse_coo_tensor(
                indices, values, size=adjacency.shape
            )
            return adj_sparse.coalesce()
        return adjacency
    
    def forward(self, batch, node_to_city_batch, debug_batch=False):
        """
        Predict population for specific (node, ethnicity) pairs.
        """
        device = batch['pop_t'].device
        
        # Full graph state
        full_population = batch['full_graph_population_t'].to(device)
        full_census = batch['full_graph_census'].to(device)
        
        node_indices = batch['node_idx']
        ethnicity_indices = batch['ethnicity']
        
        n_nodes = full_population.shape[0]
        
        # Combine features
        full_features = torch.cat([full_population, full_census], dim=-1)  # (n_nodes, features)
        
        # Encode - add batch dimension, encode, keep batch dimension
        full_features_batched = full_features.unsqueeze(0)  # (1, n_nodes, features)
        encoded = self.feature_encoder(full_features_batched)  # (1, n_nodes, hidden_dim)
        
        # Encode with DCGRU
        encoder_output, hidden_states = self.encoder(encoded, self.adjacency)
        
        # Decode
        decoder_output = self.decoder(
            encoder_output, self.adjacency, hidden_states, horizon=1
        ).squeeze(0)  # (n_nodes, hidden_dim)
        
        # Generate predictions using ethnicity-specific heads
        batch_size = len(node_indices)
        predictions = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            node_idx = node_indices[i]
            eth_idx = ethnicity_indices[i]
            
            # Use corresponding ethnicity head
            node_hidden = decoder_output[node_idx:node_idx+1]  # (1, hidden_dim)
            pred = self.ethnicity_heads[eth_idx](node_hidden)
            predictions[i] = pred.squeeze()
        
        # Ensure non-negative
        predictions = F.relu(predictions)
        
        if debug_batch:
            print(f"\nDCRNN Multi-Ethnicity Forward Pass:")
            print(f"  Predictions: mean={predictions.mean():.2f}, max={predictions.max():.2f}")
        
        return {'pop_pred': predictions}


def create_dcrnn_model(
    n_ethnicities,
    n_cities,
    n_dauids,
    n_census_features,
    adjacency,
    node_to_city,
    hidden_dim=64,
    num_layers=2,
    K=2,
    multi_ethnicity=True
):
    """
    Factory function to create DCRNN model.
    
    Args:
        multi_ethnicity: If True, use separate heads per ethnicity (more comparable to GraphPDE)
    """
    if multi_ethnicity:
        model = DCRNNMultiEthnicity(
            n_ethnicities=n_ethnicities,
            n_cities=n_cities,
            n_dauids=n_dauids,
            n_census_features=n_census_features,
            adjacency=adjacency,
            node_to_city=node_to_city,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
    else:
        model = DCRNN(
            n_ethnicities=n_ethnicities,
            n_cities=n_cities,
            n_dauids=n_dauids,
            n_census_features=n_census_features,
            adjacency=adjacency,
            node_to_city=node_to_city,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
    
    return model


if __name__ == "__main__":
    print("DCRNN baseline model loaded successfully!")