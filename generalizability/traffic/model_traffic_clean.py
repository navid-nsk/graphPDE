"""
model_traffic_clean.py

Clean GraphPDE Traffic Speed Prediction Model.

Simplified version extracted from DO_NOT_USE model with:
- NO CMS (Continuum Memory System)
- NO SMM (Self-Modifying Memory)
- NO M3 (Deep Momentum)
- NO nested learning

Core components retained:
- Physics-based reaction (Greenshields speed-density model)
- Graph Laplacian diffusion
- Sensor attention for inter-sensor interactions
- Multi-horizon ODE with checkpoints (H3, H6, H12)
- Residual decoder with interpolation
- Softmax term combination
- Proper normalization/denormalization
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from scipy import sparse


# ============================================================================
# TRAJECTORY ODE SOLVER
# ============================================================================

class TrajectoryODESolver(nn.Module):
    """
    ODE solver that captures states at multiple checkpoints in a SINGLE integration.

    For H3, H6, H12 predictions, we run ONE ODE and capture intermediate states
    at t=0.25T, t=0.5T, t=1.0T instead of running 3 separate ODEs.
    """

    def __init__(
        self,
        n_steps: int = 4,
        method: str = 'euler',
        max_value: float = 100.0,
        initial_max_deriv: float = 100.0,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.method = method
        self.max_value = max_value

        # Learnable max_deriv (log-scale for better optimization)
        self.log_max_deriv = nn.Parameter(
            torch.log(torch.tensor(initial_max_deriv))
        )

    @property
    def max_deriv(self) -> Tensor:
        """Get current max_deriv value (clamped to valid range)."""
        return torch.clamp(torch.exp(self.log_max_deriv), 10.0, 1000.0)

    def forward(
        self,
        initial_state: Tensor,
        diffusion_fn,
        reaction_fn,
        T: float = 1.0,
        checkpoints: Optional[List[float]] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Integrate ODE and optionally capture states at checkpoints.

        Args:
            initial_state: Starting state [B, n_nodes, 1]
            diffusion_fn: Function computing diffusion term
            reaction_fn: Function computing reaction term
            T: Total integration time
            checkpoints: List of times (as fractions of T) to capture states

        Returns:
            If checkpoints is None: Final state tensor
            If checkpoints provided: List of state tensors at each checkpoint
        """
        h = T / self.n_steps
        state = initial_state.clone()

        if checkpoints is not None:
            checkpoint_steps = [int(round(cp * self.n_steps)) for cp in checkpoints]
            checkpoint_steps = [min(max(1, s), self.n_steps) for s in checkpoint_steps]
            captured_states = []
            next_checkpoint_idx = 0

        for step in range(1, self.n_steps + 1):
            if self.method == 'rk4':
                state = self._rk4_step(state, diffusion_fn, reaction_fn, h)
            else:
                state = self._euler_step(state, diffusion_fn, reaction_fn, h)

            # Non-negativity constraint
            state = torch.clamp(state, min=0.0)

            # Emergency hard clamp
            if state.max() > self.max_value * 2:
                state = torch.clamp(state, max=self.max_value)

            # Capture state at checkpoints
            if checkpoints is not None and next_checkpoint_idx < len(checkpoint_steps):
                if step == checkpoint_steps[next_checkpoint_idx]:
                    captured_states.append(state.clone())
                    next_checkpoint_idx += 1
                    while (next_checkpoint_idx < len(checkpoint_steps) and
                           checkpoint_steps[next_checkpoint_idx] == step):
                        captured_states.append(state.clone())
                        next_checkpoint_idx += 1

        if checkpoints is not None:
            while len(captured_states) < len(checkpoints):
                captured_states.append(state.clone())
            return captured_states

        return state

    def _euler_step(self, y_n: Tensor, diffusion_fn, reaction_fn, h: float) -> Tensor:
        """Euler integration step with soft derivative clamping."""
        diffusion = diffusion_fn(y_n)
        reaction = reaction_fn(y_n)
        k = diffusion + reaction

        # Soft clamp derivative
        max_deriv = self.max_deriv
        k = max_deriv * torch.tanh(k / max_deriv)

        y_next = y_n + h * k
        return torch.clamp(y_next, min=0.0)

    def _rk4_step(self, y_n: Tensor, diffusion_fn, reaction_fn, h: float) -> Tensor:
        """RK4 integration step with soft derivative clamping."""
        max_deriv = self.max_deriv

        k1 = diffusion_fn(y_n) + reaction_fn(y_n)
        k1 = max_deriv * torch.tanh(k1 / max_deriv)

        y_temp = torch.clamp(y_n + h * k1 / 2.0, min=0.0)
        k2 = diffusion_fn(y_temp) + reaction_fn(y_temp)
        k2 = max_deriv * torch.tanh(k2 / max_deriv)

        y_temp = torch.clamp(y_n + h * k2 / 2.0, min=0.0)
        k3 = diffusion_fn(y_temp) + reaction_fn(y_temp)
        k3 = max_deriv * torch.tanh(k3 / max_deriv)

        y_temp = torch.clamp(y_n + h * k3, min=0.0)
        k4 = diffusion_fn(y_temp) + reaction_fn(y_temp)
        k4 = max_deriv * torch.tanh(k4 / max_deriv)

        y_next = y_n + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return torch.clamp(y_next, min=0.0)


# ============================================================================
# GRAPH LAPLACIAN
# ============================================================================

class GraphLaplacian(nn.Module):
    """
    Normalized graph Laplacian for traffic diffusion.
    Uses symmetric normalization: L = I - D^{-1/2} A D^{-1/2}
    """

    def __init__(self, adjacency: Union[sparse.spmatrix, np.ndarray, Tensor]):
        super().__init__()

        if isinstance(adjacency, Tensor):
            adj_np = adjacency.cpu().numpy()
        elif isinstance(adjacency, sparse.spmatrix):
            adj_np = adjacency.toarray()
        else:
            adj_np = adjacency

        laplacian = self._compute_normalized_laplacian(adj_np)
        self.register_buffer('laplacian', laplacian)
        self.n_nodes = adj_np.shape[0]

    def _compute_normalized_laplacian(self, adjacency: np.ndarray) -> Tensor:
        """Compute normalized graph Laplacian."""
        degree = adjacency.sum(axis=1)
        degree_inv_sqrt = np.power(degree + 1e-8, -0.5)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt
        # Negative for diffusion
        laplacian = -torch.tensor(adj_normalized, dtype=torch.float32)
        return laplacian

    def forward(self, state: Tensor, diffusion_coef: Tensor) -> Tensor:
        """Compute diffusion: coef * L @ state"""
        laplacian = self.laplacian.to(state.device)
        laplacian_state = torch.matmul(laplacian, state)

        if diffusion_coef.dim() == 0:
            return diffusion_coef * laplacian_state
        else:
            return diffusion_coef.unsqueeze(-1) * laplacian_state


# ============================================================================
# MULTI-HEAD ATTENTION (Simple implementation without external dependencies)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Simple multi-head attention for sensor interactions."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
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

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        B, N, _ = query.shape

        Q = self.W_q(query).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)
        output = self.W_o(context)

        return output, attn_weights


# ============================================================================
# SENSOR ATTENTION MODULE
# ============================================================================

class SensorAttention(nn.Module):
    """
    Attention-based sensor interaction module.
    Models how traffic at one sensor affects nearby sensors.
    """

    def __init__(self, n_nodes: int, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.n_nodes = n_nodes
        self.d_model = d_model

        # Learnable sensor embeddings
        self.sensor_embed = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)

        # Speed encoder
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, speed_state: Tensor) -> Tensor:
        """
        Args:
            speed_state: [B, n_nodes, 1] current speed
        Returns:
            interaction: [B, n_nodes, 1] sensor interactions
        """
        B, N, _ = speed_state.shape

        speed_encoded = self.speed_encoder(speed_state)
        speed_with_emb = speed_encoded + self.sensor_embed.unsqueeze(0)

        attn_out, _ = self.attention(speed_with_emb, speed_with_emb, speed_with_emb)
        interaction = self.output_proj(attn_out)

        return interaction


# ============================================================================
# REACTION MODULE (Physics + Neural, NO CMS/SMM)
# ============================================================================

class ReactionModule(nn.Module):
    """
    Reaction module combining physics and neural terms.

    NO CMS, NO SMM - just clean physics + attention + neural residual
    with softmax combination weights.
    """

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_dim: int = 64,
        use_sensor_attention: bool = True,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.use_attention = use_sensor_attention

        # ================================================================
        # PHYSICS PARAMETERS (simple scalars, not per-node)
        # ================================================================
        self.free_flow_speed = nn.Parameter(torch.tensor(65.0))
        self.jam_density = nn.Parameter(torch.tensor(0.1))
        self.relaxation_rate = nn.Parameter(torch.tensor(0.1))

        # ================================================================
        # COMBINATION WEIGHTS (softmax)
        # ================================================================
        self.physics_weight = nn.Parameter(torch.tensor(0.4))
        self.neural_weight = nn.Parameter(torch.tensor(0.3))
        self.attention_weight = nn.Parameter(torch.tensor(0.3))

        # Scaling
        self.physics_scale = nn.Parameter(torch.tensor(1.0))
        self.attention_scale = nn.Parameter(torch.tensor(0.01))

        # ================================================================
        # NEURAL ENCODER
        # ================================================================
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ================================================================
        # NEURAL RESIDUAL MLP
        # ================================================================
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ================================================================
        # SENSOR ATTENTION
        # ================================================================
        if use_sensor_attention:
            self.sensor_attention = SensorAttention(
                n_nodes=n_nodes,
                d_model=hidden_dim,
                n_heads=4,
                dropout=0.1,
            )

        # Cache
        self._cached_encoded: Optional[Tensor] = None
        self._cached_batch_shape: Optional[Tuple[int, int]] = None

    def precompute_features(self, features: Tensor) -> Tensor:
        """Pre-compute encoder output ONCE per batch."""
        B, N, F = features.shape
        features_flat = features.reshape(B * N, F)
        encoded_flat = self.encoder(features_flat)
        self._cached_encoded = encoded_flat
        self._cached_batch_shape = (B, N)
        return encoded_flat

    def forward_fast(self, state: Tensor) -> Tensor:
        """Fast reaction computation using cached encoder output."""
        assert self._cached_encoded is not None, "Must call precompute_features() first!"

        B, N = self._cached_batch_shape
        device = state.device

        if state.dim() == 2:
            state = state.unsqueeze(-1)

        # ================================================================
        # PHYSICS TERM: Greenshields model
        # ================================================================
        density_proxy = 1.0 / (state.abs() + 1e-6)
        density_proxy = torch.clamp(density_proxy, 0.0, 1.0)

        free_flow = self.free_flow_speed.abs()
        jam_dens = self.jam_density.abs()

        density_ratio = density_proxy / (jam_dens + 1e-6)
        flow_factor = 1 - density_ratio
        v_eq = free_flow * torch.clamp(flow_factor, min=0.0, max=1.0)
        v_eq = torch.minimum(v_eq, free_flow)

        physics_term = self.relaxation_rate.abs() * (v_eq - state)

        # ================================================================
        # SENSOR ATTENTION
        # ================================================================
        attention_term = torch.zeros(B, N, 1, device=device)
        if self.use_attention and hasattr(self, 'sensor_attention'):
            attention_raw = self.sensor_attention(state)
            attn_scale = torch.clamp(self.attention_scale, 1e-4, 1.0)
            attention_term = attention_raw * attn_scale

        # ================================================================
        # NEURAL RESIDUAL
        # ================================================================
        state_flat = state.reshape(B * N, 1)
        combined = torch.cat([self._cached_encoded, state_flat], dim=-1)
        neural_residual_flat = self.residual_mlp(combined)
        neural_residual = neural_residual_flat.reshape(B, N, 1)

        # ================================================================
        # COMBINE WITH SOFTMAX WEIGHTS
        # ================================================================
        weights = torch.softmax(torch.stack([
            self.physics_weight, self.attention_weight, self.neural_weight
        ]), dim=0)

        phys_scale = torch.clamp(self.physics_scale, 0.01, 10.0)

        reaction = (
            weights[0] * phys_scale * physics_term +
            weights[1] * attention_term +
            weights[2] * neural_residual
        )

        return reaction

    def clear_cache(self):
        """Clear cached encoder output."""
        self._cached_encoded = None
        self._cached_batch_shape = None

    def get_physics_params(self) -> Dict[str, float]:
        """Get current physics parameters."""
        weights = torch.softmax(torch.stack([
            self.physics_weight, self.attention_weight, self.neural_weight
        ]), dim=0)

        return {
            'free_flow_speed': self.free_flow_speed.item(),
            'jam_density': self.jam_density.item(),
            'relaxation_rate': self.relaxation_rate.item(),
            'physics_weight': weights[0].item(),
            'attention_weight': weights[1].item(),
            'neural_weight': weights[2].item(),
        }


# ============================================================================
# MULTI-HORIZON DECODER
# ============================================================================

class HorizonDecoder(nn.Module):
    """
    Residual decoder for multi-horizon prediction.

    Predicts H3, H6, H12 from ODE states, then interpolates to all 12 horizons.
    Uses residual learning: pred = alpha * ODE_state + beta * delta
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        input_dim = hidden_dim + 1  # encoded_features + ODE_state

        # H3 decoder
        self.h3_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # H6 decoder
        self.h6_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # H12 decoder (deeper)
        self.h12_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Residual weights (sigmoid initialization)
        self.h3_alpha = nn.Parameter(torch.tensor(1.4))   # ~0.8
        self.h3_beta = nn.Parameter(torch.tensor(-1.4))   # ~0.2

        self.h6_alpha = nn.Parameter(torch.tensor(1.4))
        self.h6_beta = nn.Parameter(torch.tensor(-1.4))

        self.h12_alpha = nn.Parameter(torch.tensor(1.0))  # ~0.73
        self.h12_beta = nn.Parameter(torch.tensor(-0.5))  # ~0.38

    def forward(
        self,
        encoded_features: Tensor,
        ode_state_h3: Tensor,
        ode_state_h6: Tensor,
        ode_state_h12: Tensor,
    ) -> Tensor:
        """
        Decode ODE states to 12 horizon predictions.

        Args:
            encoded_features: [B, N, hidden_dim]
            ode_state_h3/h6/h12: [B, N, 1] normalized ODE states

        Returns:
            predictions: [B, N, 12] normalized predictions
        """
        # Combine features with ODE states
        h3_input = torch.cat([encoded_features, ode_state_h3], dim=-1)
        h6_input = torch.cat([encoded_features, ode_state_h6], dim=-1)
        h12_input = torch.cat([encoded_features, ode_state_h12], dim=-1)

        # Decode corrections
        delta_h3 = self.h3_decoder(h3_input)
        delta_h6 = self.h6_decoder(h6_input)
        delta_h12 = self.h12_decoder(h12_input)

        # Apply residual: pred = alpha * ODE_state + beta * delta
        pred_h3 = torch.sigmoid(self.h3_alpha) * ode_state_h3 + torch.sigmoid(self.h3_beta) * delta_h3
        pred_h6 = torch.sigmoid(self.h6_alpha) * ode_state_h6 + torch.sigmoid(self.h6_beta) * delta_h6
        pred_h12 = torch.sigmoid(self.h12_alpha) * ode_state_h12 + torch.sigmoid(self.h12_beta) * delta_h12

        # Interpolate to 12 horizons
        predictions = self._interpolate_horizons(ode_state_h3, pred_h3, pred_h6, pred_h12)

        return predictions

    def _interpolate_horizons(
        self,
        h0: Tensor,
        pred_h3: Tensor,
        pred_h6: Tensor,
        pred_h12: Tensor,
    ) -> Tensor:
        """Interpolate 3 predictions to 12 horizons."""
        horizons = []

        # H1-H3
        horizons.append(h0 + (pred_h3 - h0) * (1/3))
        horizons.append(h0 + (pred_h3 - h0) * (2/3))
        horizons.append(pred_h3)

        # H4-H6
        horizons.append(pred_h3 + (pred_h6 - pred_h3) * (1/3))
        horizons.append(pred_h3 + (pred_h6 - pred_h3) * (2/3))
        horizons.append(pred_h6)

        # H7-H12
        for i in range(1, 6):
            horizons.append(pred_h6 + (pred_h12 - pred_h6) * (i / 6))
        horizons.append(pred_h12)

        return torch.cat(horizons, dim=-1)

    def get_residual_weights(self) -> Dict[str, float]:
        """Get residual weights for logging."""
        return {
            'h3_alpha': torch.sigmoid(self.h3_alpha).item(),
            'h3_beta': torch.sigmoid(self.h3_beta).item(),
            'h6_alpha': torch.sigmoid(self.h6_alpha).item(),
            'h6_beta': torch.sigmoid(self.h6_beta).item(),
            'h12_alpha': torch.sigmoid(self.h12_alpha).item(),
            'h12_beta': torch.sigmoid(self.h12_beta).item(),
        }


# ============================================================================
# MAIN MODEL: GraphPDE Traffic (Clean)
# ============================================================================

class GraphPDETrafficClean(nn.Module):
    """
    Clean GraphPDE model for traffic speed prediction.

    NO CMS, NO SMM, NO nested learning - just core GraphPDE components:
    - Graph Laplacian diffusion
    - Physics-based reaction (Greenshields)
    - Sensor attention
    - Neural residual
    - Multi-horizon ODE with interpolation
    """

    def __init__(
        self,
        n_nodes: int,
        adjacency: Union[sparse.spmatrix, np.ndarray, Tensor],
        input_window: int = 12,
        pred_horizon: int = 12,
        hidden_dim: int = 128,
        n_ode_steps: int = 4,
        ode_method: str = 'euler',
        integration_time: float = 1.0,
        data_mean: float = 58.47,
        data_std: float = 12.62,
        use_sensor_attention: bool = True,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_features = input_window
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.integration_time = integration_time

        # Normalization parameters
        self.register_buffer('data_mean', torch.tensor(data_mean))
        self.register_buffer('data_std', torch.tensor(data_std))

        # ================================================================
        # DIFFUSION
        # ================================================================
        self.diffusion_coef = nn.Parameter(torch.tensor(0.1))
        self.diffusion_weight = nn.Parameter(torch.tensor(1.0))
        self.graph_laplacian = GraphLaplacian(adjacency)

        # ================================================================
        # REACTION (No CMS/SMM)
        # ================================================================
        self.reaction_module = ReactionModule(
            n_nodes=n_nodes,
            n_features=input_window,
            hidden_dim=hidden_dim,
            use_sensor_attention=use_sensor_attention,
        )

        # ================================================================
        # ODE SOLVER
        # ================================================================
        self.ode_solver = TrajectoryODESolver(
            n_steps=n_ode_steps,
            method=ode_method,
            max_value=100.0,
        )
        self.ode_checkpoints = [0.25, 0.5, 1.0]  # H3, H6, H12

        # ================================================================
        # FEATURE ENCODER
        # ================================================================
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ================================================================
        # HORIZON DECODER
        # ================================================================
        self.horizon_decoder = HorizonDecoder(hidden_dim=hidden_dim)

    def _normalize(self, x: Tensor) -> Tensor:
        """Normalize from mph to standard scale."""
        return (x - self.data_mean) / self.data_std

    def _denormalize(self, x: Tensor) -> Tensor:
        """Denormalize from standard scale to mph."""
        return x * self.data_std + self.data_mean

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            batch: Dict with 'input' [B, n_nodes, input_window] in mph

        Returns:
            Dict with 'predictions' [B, n_nodes, pred_horizon] in mph
        """
        device = next(self.parameters()).device

        # Handle both 'input' and 'features' keys for compatibility
        if 'input' in batch:
            features = batch['input'].to(device)
        else:
            features = batch['features'].to(device)

        # Initial state: last timestep
        initial_state = features[:, :, -1:]  # [B, n_nodes, 1] in mph

        # Encode features (normalized for encoder)
        features_norm = self._normalize(features)
        encoded_features = self.feature_encoder(features_norm)  # [B, n_nodes, hidden_dim]

        # Pre-compute reaction encoder
        self.reaction_module.precompute_features(features)

        # Diffusion function
        def diffusion_fn(state: Tensor) -> Tensor:
            coef = self.diffusion_weight * self.diffusion_coef.abs()
            return self.graph_laplacian(state, coef)

        # Reaction function
        def reaction_fn(state: Tensor) -> Tensor:
            return self.reaction_module.forward_fast(state)

        # Integrate ODE with checkpoints
        checkpoint_states = self.ode_solver(
            initial_state,
            diffusion_fn,
            reaction_fn,
            T=self.integration_time,
            checkpoints=self.ode_checkpoints,
        )

        state_h3, state_h6, state_h12 = checkpoint_states

        # Clear cache
        self.reaction_module.clear_cache()

        # Normalize ODE states for decoder
        state_h3_norm = self._normalize(state_h3)
        state_h6_norm = self._normalize(state_h6)
        state_h12_norm = self._normalize(state_h12)

        # Decode
        predictions_norm = self.horizon_decoder(
            encoded_features,
            state_h3_norm,
            state_h6_norm,
            state_h12_norm,
        )

        # Denormalize
        predictions = self._denormalize(predictions_norm)

        return {'predictions': predictions}

    def get_parameters_dict(self) -> Dict[str, Any]:
        """Get interpretable parameters for logging."""
        params = self.reaction_module.get_physics_params()
        params['diffusion_coef'] = self.diffusion_coef.item()
        params['diffusion_weight'] = self.diffusion_weight.item()
        params.update(self.horizon_decoder.get_residual_weights())
        return params

    def clip_parameters(self):
        """Clip parameters for stability."""
        with torch.no_grad():
            self.diffusion_coef.data.clamp_(min=1e-6, max=1.0)
            self.diffusion_weight.data.clamp_(min=0.0, max=10.0)
            self.reaction_module.free_flow_speed.data.clamp_(min=10.0, max=100.0)
            self.reaction_module.jam_density.data.clamp_(min=0.01, max=1.0)
            self.reaction_module.relaxation_rate.data.clamp_(min=0.01, max=1.0)


def create_model(
    n_nodes: int,
    adjacency: Union[sparse.spmatrix, np.ndarray, Tensor],
    input_window: int = 12,
    pred_horizon: int = 12,
    hidden_dim: int = 128,
    n_ode_steps: int = 4,
    ode_method: str = 'euler',
    data_mean: float = 58.47,
    data_std: float = 12.62,
    use_sensor_attention: bool = True,
) -> GraphPDETrafficClean:
    """Factory function to create the clean traffic model."""
    return GraphPDETrafficClean(
        n_nodes=n_nodes,
        adjacency=adjacency,
        input_window=input_window,
        pred_horizon=pred_horizon,
        hidden_dim=hidden_dim,
        n_ode_steps=n_ode_steps,
        ode_method=ode_method,
        data_mean=data_mean,
        data_std=data_std,
        use_sensor_attention=use_sensor_attention,
    )


if __name__ == "__main__":
    # Test
    n_nodes = 207
    batch_size = 4

    adj = np.random.rand(n_nodes, n_nodes).astype(np.float32)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 1.0)

    model = create_model(
        n_nodes=n_nodes,
        adjacency=adj,
        hidden_dim=128,
        n_ode_steps=4,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    x = torch.randn(batch_size, n_nodes, 12) * 10 + 50  # mph
    batch = {'input': x}

    output = model(batch)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output['predictions'].shape}")
    print(f"Physics params: {model.get_parameters_dict()}")
    print("Test passed!")
