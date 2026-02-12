"""
model_graphpde.py

GraphPDE Model for Population Dynamics Prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Try to import custom CUDA extensions
try:
    cuda_path = Path(__file__).parent / 'cuda_extensions'
    sys.path.insert(0, str(cuda_path / 'graph_laplacian'))
    
    import graph_laplacian_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class VectorizedMultiHeadAttention(nn.Module):
    """
    Batched multi-head attention that processes all queries simultaneously.
    
    Instead of:
        for i in range(n_queries):
            output[i] = attention(query[i], keys, values)
    
    We do:
        output = attention(all_queries, keys, values)  # Batched!
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, n_queries, d_model) - Multiple queries processed together
            key: (batch, n_keys, d_model)
            value: (batch, n_keys, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, n_queries, d_model)
            attn_weights: (batch, n_heads, n_queries, n_keys)
        """
        batch_size = query.shape[0]
        n_queries = query.shape[1]
        n_keys = key.shape[1]
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, n_queries, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, n_keys, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, n_keys, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, n_heads, n_queries, d_k) @ (batch, n_heads, d_k, n_keys)
        # -> (batch, n_heads, n_queries, n_keys)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, n_heads, n_queries, n_keys) @ (batch, n_heads, n_keys, d_k)
        # -> (batch, n_heads, n_queries, d_k)
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # (batch, n_heads, n_queries, d_k) -> (batch, n_queries, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, n_queries, self.d_model)
        
        # Output projection + residual
        output = self.W_o(context)
        output = self.layer_norm(output + query)
        
        return output, attn_weights


class EthnicityAttentionModule(nn.Module):
    """
    Attention-based ethnicity interaction (replaces simple linear W matrix)
    
    Learns which ethnicities influence each other and how.
    More expressive than W @ population!
    """
    def __init__(self, n_ethnicities, n_periods, d_model=64, n_heads=4):
        super().__init__()
        self.n_ethnicities = n_ethnicities
        self.d_model = d_model
        
        # Per-period ethnicity embeddings
        self.ethnicity_embed = nn.Parameter(
            torch.randn(n_periods, n_ethnicities, d_model) * 0.02
        )
        
        # Population encoder
        self.pop_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Vectorized multi-head attention (processes all queries at once)
        self.attention = VectorizedMultiHeadAttention(d_model, n_heads)
        
        # Output projection (can handle batched input)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, population, period_idx):
        """
        Args:
            population: (n_nodes, n_ethnicities) - current populations
            period_idx: (n_nodes,) - period index for each node
        Returns:
            (n_nodes, n_ethnicities) - interaction term
        """
        n_nodes, n_ethnicities = population.shape
        device = population.device
        
        # Get period-specific embeddings
        dominant_period = period_idx.mode().values.item() if len(period_idx.unique()) > 0 else 0
        eth_emb = self.ethnicity_embed[dominant_period]  # (n_eth, d_model)
        
        # Encode populations
        # (n_nodes, n_eth, 1) -> (n_nodes, n_eth, d_model)
        pop_encoded = self.pop_encoder(population.unsqueeze(-1))
        
        # Add ethnicity embeddings (broadcast)
        # (n_nodes, n_eth, d_model) + (n_eth, d_model) -> (n_nodes, n_eth, d_model)
        pop_with_emb = pop_encoded + eth_emb.unsqueeze(0)
        
        # ====================================================================
        # KEY OPTIMIZATION: Process all ethnicities at once!
        # ====================================================================
        # Instead of looping, we use pop_with_emb as both query and key/value
        # Each ethnicity attends to all others simultaneously
        
        # Attention: All ethnicities attend to all others in parallel
        # query: (n_nodes, n_eth, d_model) - all ethnicities are queries
        # key/value: (n_nodes, n_eth, d_model) - all ethnicities are keys/values
        attn_out, _ = self.attention(
            pop_with_emb,  # queries: all ethnicities
            pop_with_emb,  # keys: all ethnicities
            pop_with_emb   # values: all ethnicities
        )
        # Output: (n_nodes, n_eth, d_model)
        
        # Project to interaction terms
        # (n_nodes, n_eth, d_model) -> (n_nodes, n_eth, 1) -> (n_nodes, n_eth)
        interaction_repr = self.output_proj(attn_out).squeeze(-1)
        
        return interaction_repr


# ============================================================================
# GRAPH LAPLACIAN (Same as original, with CUDA support)
# ============================================================================

class GraphLaplacianFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, population, diffusion_coef, node_to_city, row_ptr, col_idx, laplacian_values):
        # Clamp for numerical stability
        population = torch.clamp(population, min=0.0, max=1e6)
        diffusion_coef = torch.clamp(diffusion_coef, min=0.0, max=10.0)
        
        ctx.save_for_backward(population, diffusion_coef, node_to_city, row_ptr, col_idx, laplacian_values)
        
        output = graph_laplacian_cuda.forward(
            population.contiguous(),
            diffusion_coef.contiguous(),
            node_to_city.contiguous(),
            row_ptr.contiguous(),
            col_idx.contiguous(),
            laplacian_values.contiguous()
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = torch.clamp(grad_output, min=-1e6, max=1e6)
        
        population, diffusion_coef, node_to_city, row_ptr, col_idx, laplacian_values = ctx.saved_tensors
        
        grad_population, grad_diffusion_coef = graph_laplacian_cuda.backward(
            grad_output.contiguous(),
            population.contiguous(),
            diffusion_coef.contiguous(),
            node_to_city.contiguous(),
            row_ptr.contiguous(),
            col_idx.contiguous(),
            laplacian_values.contiguous()
        )
        
        return grad_population, grad_diffusion_coef, None, None, None, None


class GraphLaplacianModule(nn.Module):
    def __init__(self, adjacency, node_to_city, use_cuda_kernel=True):
        super().__init__()
        
        self.use_cuda_kernel = use_cuda_kernel and CUDA_AVAILABLE
        
        self.register_buffer('laplacian_sparse', self._compute_laplacian(adjacency))
        self.register_buffer('node_to_city', torch.from_numpy(node_to_city.astype(np.int32)).to(torch.int32))
        
        if self.use_cuda_kernel:
            self._prepare_csr_format()
    
    def _compute_laplacian(self, adjacency):
        coo = adjacency.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        
        adj_sparse = torch.sparse_coo_tensor(indices, values, size=adjacency.shape)
        degree = torch.sparse.sum(adj_sparse, dim=1).to_dense()
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        
        row, col = indices[0], indices[1]
        norm_values = values * degree_inv_sqrt[row] * degree_inv_sqrt[col]
        laplacian_values = -norm_values
        
        laplacian_sparse = torch.sparse_coo_tensor(
            indices, laplacian_values, size=adjacency.shape
        )
        
        return laplacian_sparse
    
    def _prepare_csr_format(self):
        laplacian_coo = self.laplacian_sparse.coalesce()
        indices = laplacian_coo.indices()
        values = laplacian_coo.values()
        n_nodes = self.laplacian_sparse.size(0)
        
        row_idx = indices[0].cpu().numpy()
        col_idx = indices[1].cpu().numpy()
        data = values.cpu().numpy()
        
        row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
        for i in row_idx:
            row_ptr[i + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        
        self.register_buffer('row_ptr', torch.from_numpy(row_ptr).to(torch.int32))
        self.register_buffer('col_idx', torch.from_numpy(col_idx.astype(np.int32)).to(torch.int32))
        self.register_buffer('laplacian_values', torch.from_numpy(data.astype(np.float32)))
    
    def forward(self, population, diffusion_coef):
        if self.use_cuda_kernel:
            return GraphLaplacianFunction.apply(
                population, diffusion_coef, self.node_to_city,
                self.row_ptr, self.col_idx, self.laplacian_values
            )
        else:
            return self._pytorch_implementation(population, diffusion_coef)
    
    def _pytorch_implementation(self, population, diffusion_coef):
        n_nodes, n_ethnicities = population.shape
        laplacian_dense = self.laplacian_sparse.to_dense()
        diffusion_term = torch.zeros_like(population)
        
        for eth in range(n_ethnicities):
            laplacian_pop = torch.matmul(laplacian_dense, population[:, eth])
            city_diff_coef = diffusion_coef[self.node_to_city, eth]
            diffusion_term[:, eth] = city_diff_coef * laplacian_pop
        
        return diffusion_term

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GroupNorm(32, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.net(x)  # Skip connection
    
# ============================================================================
# ENHANCED REACTION MODULE (Physics + Neural) - NO DAUID EMBEDDINGS
# ============================================================================

class EnhancedReactionModule(nn.Module):
    """
    Physics-based reaction terms + Neural residual
    
    Components:
    1. Physics terms (logistic, immigration, emigration)
    2. Attention-based competition (replaces linear W)
    3. Deep embeddings (City, Ethnicity, Period) - NO DAUID!
    4. Neural residual (MLP learns what physics misses)
    
    """
    def __init__(self, n_ethnicities, n_cities, n_dauids, n_census_features):
        super().__init__()
        
        self.n_ethnicities = n_ethnicities
        self.n_cities = n_cities
        self.n_dauids = n_dauids
        self.n_periods = 4
        
        self.period_map = {2001: 0, 2006: 1, 2011: 2, 2016: 3}
        
        # ====================================================================
        # PHYSICS PARAMETERS (interpretable)
        # ====================================================================
        
        # Growth rates
        self.growth_rates = nn.Parameter(
            torch.randn(self.n_periods, n_cities, n_ethnicities) * 0.005 + 0.015
        )
        
        # Carrying capacity
        self.carrying_capacity = nn.Parameter(
            torch.randn(self.n_periods, n_cities) * 200.0 + 1000.0
        )
        
        # Immigration rates
        self.immigration_rates = nn.Parameter(
            torch.randn(self.n_periods, n_cities, n_ethnicities) * 0.002 + 0.003
        )
        
        # Individual capacity
        self.individual_capacity = nn.Parameter(
            torch.randn(self.n_periods, n_cities, n_ethnicities) * 50.0 + 200.0
        )
        
        # Emigration rates
        self.emigration_rates = nn.Parameter(
            torch.randn(self.n_periods, n_cities, n_ethnicities) * 0.0005 + 0.001
        )
        
        # ====================================================================
        # ATTENTION-BASED COMPETITION (replaces simple W matrix)
        # ====================================================================
        
        self.ethnicity_attention = EthnicityAttentionModule(
            n_ethnicities, self.n_periods, d_model=64, n_heads=4
        )
        
        # ====================================================================
        # DEEP EMBEDDINGS 
        # ====================================================================
        
        # City embeddings (city-specific dynamics)
        self.city_embeddings = nn.Embedding(n_cities, 512)
        nn.init.normal_(self.city_embeddings.weight, mean=0.0, std=0.05)
        
        # Ethnicity embeddings (ethnicity-specific patterns)
        self.ethnicity_embeddings = nn.Embedding(n_ethnicities, 512)
        nn.init.normal_(self.ethnicity_embeddings.weight, mean=0.0, std=0.05)
        
        # Period embeddings (temporal patterns)
        self.period_embeddings = nn.Embedding(self.n_periods, 64)
        nn.init.normal_(self.period_embeddings.weight, mean=0.0, std=0.05)
        
        # ====================================================================
        # CENSUS MODULATOR (deeper network)
        # ====================================================================
        
        self.census_modulator = nn.Sequential(
            nn.Linear(n_census_features, 256),
            nn.BatchNorm1d(256),  # ✓ Good - fixed input size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),  # ✓ Good
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # ✓ Good
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_ethnicities)
            # No norm on output layer
        )
        
        # ====================================================================
        # NEURAL RESIDUAL (learns what physics misses!)
        # ====================================================================
        
        # Feature fusion dimension
        # City(256) + Period(16) + census_mod(n_ethnicities) + population(n_ethnicities)
        fusion_dim = 512 + 64 + n_ethnicities + n_ethnicities
        
        self.residual_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            ResidualBlock(512, dropout=0.05),  # Add skip connection
            ResidualBlock(512, dropout=0.05),  # Add skip connection
            nn.Linear(512, 256),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            ResidualBlock(256, dropout=0.05),
            nn.Linear(256, n_ethnicities)
        )
        
        # Learnable weight for residual term
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights
        for m in list(self.census_modulator.modules()) + \
                  list(self.residual_mlp.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _count_physics_params(self):
        return (self.growth_rates.numel() + 
                self.carrying_capacity.numel() +
                self.immigration_rates.numel() +
                self.individual_capacity.numel() +
                self.emigration_rates.numel())
    
    def _count_neural_params(self):
        return (sum(p.numel() for p in self.ethnicity_attention.parameters()) +
                sum(p.numel() for p in self.census_modulator.parameters()) +
                sum(p.numel() for p in self.residual_mlp.parameters()) +
                self.city_embeddings.weight.numel() +
                self.ethnicity_embeddings.weight.numel() +
                self.period_embeddings.weight.numel())
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, population, node_to_city, census_features, node_idx,
                ethnicity_idx, period_idx):
        """
        Args:
            population: (n_nodes, n_ethnicities)
            node_to_city: (n_nodes,) - city index for each node
            census_features: (n_nodes, n_census_features)
            node_idx: (n_nodes,) - DAUID index (NOT USED)
            ethnicity_idx: (n_nodes,) - target ethnicity (for batch)
            period_idx: (n_nodes,) - period index
        Returns:
            (n_nodes, n_ethnicities) - reaction term
        """
        n_nodes, n_ethnicities = population.shape
        device = population.device
        
        # Convert period_idx to tensor if needed
        if isinstance(period_idx, int):
            period_idx = torch.full((n_nodes,), period_idx, dtype=torch.long, device=device)
        elif isinstance(period_idx, (list, np.ndarray)):
            period_idx = torch.tensor(period_idx, dtype=torch.long, device=device)
        
        unique_periods = period_idx.unique()
        
        # ====================================================================
        # GET PHYSICS PARAMETERS
        # ====================================================================
        
        growth_rates = torch.zeros(n_nodes, n_ethnicities, device=device)
        carrying_capacity_vals = torch.zeros(n_nodes, device=device)
        immigration_rates = torch.zeros(n_nodes, n_ethnicities, device=device)
        individual_capacity_vals = torch.zeros(n_nodes, n_ethnicities, device=device)
        emigration_rates = torch.zeros(n_nodes, n_ethnicities, device=device)
        
        for p in unique_periods:
            mask = (period_idx == p)
            cities_p = node_to_city[mask]
            
            growth_rates[mask] = self.growth_rates[p][cities_p]
            carrying_capacity_vals[mask] = self.carrying_capacity[p][cities_p]
            immigration_rates[mask] = self.immigration_rates[p][cities_p]
            individual_capacity_vals[mask] = self.individual_capacity[p][cities_p]
            emigration_rates[mask] = self.emigration_rates[p][cities_p]
        
        # ====================================================================
        # GET EMBEDDINGS 
        # ====================================================================
        
        city_emb = self.city_embeddings(node_to_city)  # (n_nodes, 64)
        period_emb = self.period_embeddings(period_idx)  # (n_nodes, 16)
        
        # ====================================================================
        # MODULATIONS
        # ====================================================================
        
        census_hidden = self.census_modulator(census_features)
        census_mod = torch.sigmoid(census_hidden)
        
        # Apply census modulation only (no spatial adjustment)
        r_i = growth_rates * census_mod
        F_i = immigration_rates * census_mod
        emigration_i = emigration_rates * census_mod
        
        K_c = carrying_capacity_vals
        K_i = individual_capacity_vals
        
        # ====================================================================
        # COMPUTE PHYSICS TERMS
        # ====================================================================
        
        N = population.sum(dim=1, keepdim=True) + 1e-6
        
        # 1. Logistic growth
        carrying_ratio = N / (K_c.unsqueeze(1) + 1e-6)
        logistic_term = r_i * population * (1.0 - carrying_ratio)
        
        # 2. Competition (attention-based!)
        competition_term = self.ethnicity_attention(population, period_idx)
        
        # 3. Immigration
        immigration_ratio = population / (K_i + 1e-6)
        immigration_term = F_i * (1.0 - immigration_ratio)
        
        # 4. Emigration
        emigration_term = -emigration_i * population
        
        # Combine physics terms
        physics_reaction = (logistic_term + competition_term + 
                          immigration_term + emigration_term)
        
        # ====================================================================
        # NEURAL RESIDUAL (learns what physics misses!)
        # ====================================================================
        
        # Normalize population for stability (avoid large values)
        pop_normalized = population / (population.sum(dim=1, keepdim=True) + 1e-6)
        
        # Fuse embeddings: City + Period + census_mod + population (NO DAUID!)
        fused_features = torch.cat([
            city_emb,           # (n_nodes, 64) City-specific dynamics
            period_emb,         # (n_nodes, 16) Temporal patterns
            census_mod,         # (n_nodes, n_ethnicities) Census-modulated features
            pop_normalized      # (n_nodes, n_ethnicities) Current population distribution
        ], dim=-1)  # Total: 64 + 16 + n_ethnicities + n_ethnicities
        
        # MLP learns residual
        neural_residual = self.residual_mlp(fused_features)
        
        # Combine: physics + neural residual
        total_reaction = physics_reaction + self.residual_weight * neural_residual
        
        return total_reaction


# ============================================================================
# ODE SOLVER 
# ============================================================================

class NeuralODESolver(nn.Module):
    def __init__(self, n_steps=1):
        super().__init__()
        self.n_steps = n_steps
    
    def forward(self, initial_state, diffusion_fn, reaction_fn, T=5.0):
        h = T / self.n_steps
        state = initial_state
        
        for step in range(self.n_steps):
            state = self._rk4_step(state, diffusion_fn, reaction_fn, h)
            state = torch.clamp(state, min=0.0)
        
        return state
    
    def _rk4_step(self, y_n, diffusion_fn, reaction_fn, h):
        k1 = diffusion_fn(y_n) + reaction_fn(y_n)
        
        y_temp = torch.clamp(y_n + h * k1 / 2.0, min=0.0)
        k2 = diffusion_fn(y_temp) + reaction_fn(y_temp)
        
        y_temp = torch.clamp(y_n + h * k2 / 2.0, min=0.0)
        k3 = diffusion_fn(y_temp) + reaction_fn(y_temp)
        
        y_temp = torch.clamp(y_n + h * k3, min=0.0)
        k4 = diffusion_fn(y_temp) + reaction_fn(y_temp)
        
        y_next = y_n + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        y_next = torch.clamp(y_next, min=0.0)
        
        return y_next


# ============================================================================
# MAIN MODEL
# ============================================================================

class EnhancedGraphPDE(nn.Module):
    """
    Enhanced GraphPDE: Physics + Neural Network Augmentation
    
    """
    def __init__(
        self,
        n_ethnicities,
        n_cities,
        n_dauids,
        n_census_features,
        adjacency,
        node_to_city,
        n_ode_steps=1,
        integration_time=5.0,
        use_cuda_kernels=True
    ):
        super().__init__()
        
        self.n_ethnicities = n_ethnicities
        self.n_cities = n_cities
        self.n_dauids = n_dauids
        self.integration_time = integration_time
        self.n_periods = 4
        
        self.period_map = {2001: 0, 2006: 1, 2011: 2, 2016: 3}
        
        # Diffusion coefficients
        self.diffusion_coef = nn.Parameter(
            torch.randn(self.n_periods, n_cities, n_ethnicities) * 0.0005 + 0.001
        )
        
        
        # Learnable weight for diffusion term
        self.diffusion_weight = nn.Parameter(torch.tensor(1.0))
        
        # Graph Laplacian
        self.graph_laplacian = GraphLaplacianModule(
            adjacency, node_to_city, use_cuda_kernel=use_cuda_kernels
        )
        
        # Enhanced Reaction Module (NO DAUID)
        self.reaction_module = EnhancedReactionModule(
            n_ethnicities, n_cities, n_dauids, n_census_features
        )
        
        # ODE Solver
        self.ode_solver = NeuralODESolver(n_steps=n_ode_steps)
        
    
    def forward(self, batch, node_to_city_batch):
        batch_size = batch['pop_t'].shape[0]
        device = batch['pop_t'].device
        
        node_indices = batch['node_idx']
        ethnicity_indices = batch['ethnicity']
        
        # Get period indices
        year_t = batch['year_t']
        if isinstance(year_t, list):
            period_indices = torch.tensor([self.period_map[int(y)] for y in year_t], 
                                        dtype=torch.long, device=device)
        else:
            period_idx = self.period_map[int(year_t[0])]
            period_indices = torch.full((batch_size,), period_idx, dtype=torch.long, device=device)
        
        # Full graph state
        full_population = batch['full_graph_population_t'].to(device)
        full_census = batch['full_graph_census'].to(device)
        
        n_nodes = full_population.shape[0]
        full_node_idx = torch.arange(n_nodes, dtype=torch.long, device=device)
        full_period_idx = torch.zeros(n_nodes, dtype=torch.long, device=device)
        full_period_idx[node_indices] = period_indices
        
        # Dominant period for diffusion
        dominant_period = period_indices.mode().values.item()
        diffusion_coef = self.diffusion_coef[dominant_period]
        
        # Define ODE functions
        def diffusion_fn(state):
            return self.diffusion_weight * self.graph_laplacian(state, diffusion_coef)

        def reaction_fn(state):
            return self.reaction_module(
                state,
                self.graph_laplacian.node_to_city,
                full_census,
                full_node_idx,
                ethnicity_indices,
                full_period_idx
            )

        # Integrate ODE
        final_population = self.ode_solver(
            full_population, diffusion_fn, reaction_fn, T=self.integration_time
        )
        
        # Extract predictions
        predictions = final_population[node_indices, ethnicity_indices]

        return {'pop_pred': predictions}
    
    def get_parameters_dict(self):
        """Return interpretable physics parameters"""
        params_dict = {}
        
        params_dict['diffusion_weight'] = self.diffusion_weight.item()
        params_dict['residual_weight'] = self.reaction_module.residual_weight.item()
        
        for p in range(self.n_periods):
            period_name = ['2001-2006', '2006-2011', '2011-2016', '2016-2021'][p]
            
            params_dict[f'diffusion_coef_{period_name}'] = self.diffusion_coef[p].detach().cpu().numpy()
            params_dict[f'growth_rates_{period_name}'] = self.reaction_module.growth_rates[p].detach().cpu().numpy()
            params_dict[f'carrying_capacity_{period_name}'] = self.reaction_module.carrying_capacity[p].detach().cpu().numpy()
            params_dict[f'immigration_rates_{period_name}'] = self.reaction_module.immigration_rates[p].detach().cpu().numpy()
            params_dict[f'emigration_rates_{period_name}'] = self.reaction_module.emigration_rates[p].detach().cpu().numpy()
        
        return params_dict
    
    def clip_parameters(self):
        """Loose clipping to prevent numerical overflow"""
        with torch.no_grad():
            self.diffusion_coef.clamp_(min=-10.0, max=10.0)
            self.reaction_module.growth_rates.clamp_(min=-5.0, max=5.0)
            self.reaction_module.carrying_capacity.clamp_(min=1.0, max=100000.0)


def create_graphpde_model(
    n_ethnicities,
    n_cities,
    n_dauids,
    n_census_features,
    adjacency,
    node_to_city,
    n_ode_steps=1,
    integration_time=5.0,
    use_cuda_kernels=True
):
    """Factory function to create Enhanced GraphPDE model"""
    model = EnhancedGraphPDE(
        n_ethnicities=n_ethnicities,
        n_cities=n_cities,
        n_dauids=n_dauids,
        n_census_features=n_census_features,
        adjacency=adjacency,
        node_to_city=node_to_city,
        n_ode_steps=n_ode_steps,
        integration_time=integration_time,
        use_cuda_kernels=use_cuda_kernels
    )
    
    return model


if __name__ == "__main__":
    print("Enhanced GraphPDE model loaded successfully!")

