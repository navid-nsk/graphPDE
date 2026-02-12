"""
model_epidemiology_clean.py

Clean GraphPDE model for epidemiology prediction.
NO CMS, NO SMM, NO M3 - but keeps all the core architecture from DO_NOT_USE.

Key features:
- SIR-inspired physics (growth, decay, competition)
- Graph Laplacian diffusion
- Neural encoder and residual MLP
- Softmax term combination (physics, neural)
- Multi-step horizon decoder
- Proper ODE integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from typing import Dict, Any, List, Optional, Tuple, Union
from torch import Tensor


# =============================================================================
# GRAPH LAPLACIAN
# =============================================================================

class GraphLaplacian(nn.Module):
    """
    Normalized graph Laplacian for diffusion.
    Uses symmetric normalization: L = I - D^{-1/2} A D^{-1/2}
    """

    def __init__(self, adjacency: Union[sparse.spmatrix, np.ndarray, Tensor]):
        super().__init__()

        # Convert to numpy
        if isinstance(adjacency, Tensor):
            adj_np = adjacency.cpu().numpy()
        elif isinstance(adjacency, sparse.spmatrix):
            adj_np = adjacency.toarray()
        else:
            adj_np = adjacency.copy()

        # Compute normalized Laplacian
        laplacian = self._compute_normalized_laplacian(adj_np)
        self.register_buffer('laplacian', laplacian)
        self.n_nodes = adj_np.shape[0]

    def _compute_normalized_laplacian(self, adjacency: np.ndarray) -> Tensor:
        """Compute normalized graph Laplacian."""
        degree = adjacency.sum(axis=1)
        degree_inv_sqrt = np.power(degree + 1e-8, -0.5)
        D_inv_sqrt = np.diag(degree_inv_sqrt)

        # Normalized adjacency: D^{-1/2} A D^{-1/2}
        adj_normalized = D_inv_sqrt @ adjacency @ D_inv_sqrt

        # Negative for Laplacian diffusion
        laplacian = -torch.tensor(adj_normalized, dtype=torch.float32)
        return laplacian

    def forward(self, state: Tensor, diffusion_coef: Tensor) -> Tensor:
        """
        Compute diffusion term.

        Args:
            state: [n_nodes, 1] or [n_nodes]
            diffusion_coef: Scalar diffusion coefficient

        Returns:
            Diffusion term [n_nodes, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(-1)

        laplacian = self.laplacian.to(state.device)
        laplacian_state = torch.matmul(laplacian, state)

        if diffusion_coef.dim() == 0:
            diffusion = diffusion_coef * laplacian_state
        else:
            diffusion = diffusion_coef.unsqueeze(-1) * laplacian_state

        return diffusion


# =============================================================================
# REACTION MODULE (SIR-inspired, no CMS/SMM)
# =============================================================================

class ReactionModule(nn.Module):
    """
    Reaction module for epidemiological dynamics.
    SIR-inspired physics + neural residual, NO CMS/SMM.
    """

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # ================================================================
        # PHYSICS PARAMETERS (SIR-inspired)
        # ================================================================
        self.growth_rate = nn.Parameter(torch.tensor(0.1))  # Infection rate
        self.decay_rate = nn.Parameter(torch.tensor(0.05))  # Recovery rate
        self.log_capacity = nn.Parameter(torch.tensor(5.0))  # Susceptible population
        self.competition_scale = nn.Parameter(torch.tensor(0.01))

        # Physics scaling
        self.physics_scale = nn.Parameter(torch.tensor(1.0))

        # ================================================================
        # BALANCE WEIGHTS (softmax combination)
        # ================================================================
        self.physics_weight = nn.Parameter(torch.tensor(0.6))
        self.neural_weight = nn.Parameter(torch.tensor(0.4))

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

        # Neural residual MLP
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for current state
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

        # Cache for encoder output
        self._cached_encoded: Optional[Tensor] = None

    def precompute_features(self, features: Tensor) -> Tensor:
        """Pre-compute encoder output once per batch."""
        encoded = self.encoder(features)
        self._cached_encoded = encoded
        return encoded

    def forward_fast(self, state: Tensor) -> Tensor:
        """Fast reaction computation using cached encoder output."""
        assert self._cached_encoded is not None, "Must call precompute_features() first!"

        if state.dim() == 1:
            state = state.unsqueeze(-1)

        # ================================================================
        # PHYSICS TERM: SIR-inspired logistic growth
        # ================================================================
        capacity = self.log_capacity.exp()
        growth = self.growth_rate.abs() * state * (1 - state / capacity)
        decay = self.decay_rate.abs() * state
        competition = self.competition_scale.abs() * state ** 2

        phys_scale = torch.clamp(self.physics_scale, 0.01, 10.0)
        physics_term = phys_scale * (growth - decay - competition)

        # ================================================================
        # NEURAL RESIDUAL
        # ================================================================
        combined = torch.cat([self._cached_encoded, state], dim=-1)
        neural_term = self.residual_mlp(combined)

        # ================================================================
        # COMBINE WITH SOFTMAX WEIGHTS
        # ================================================================
        weights = torch.softmax(torch.stack([
            self.physics_weight, self.neural_weight
        ]), dim=0)

        reaction = weights[0] * physics_term + weights[1] * neural_term

        return reaction

    def clear_cache(self):
        """Clear cached encoder output."""
        self._cached_encoded = None

    def forward(self, state: Tensor, features: Tensor) -> Tuple[Tensor, Dict]:
        """Full forward pass."""
        if state.dim() == 1:
            state = state.unsqueeze(-1)

        self.precompute_features(features)
        reaction = self.forward_fast(state)
        self.clear_cache()

        debug_info = self._get_debug_info(state, reaction)
        return reaction, debug_info

    def _get_debug_info(self, state: Tensor, reaction: Tensor) -> Dict:
        """Get debug information."""
        weights = torch.softmax(torch.stack([
            self.physics_weight, self.neural_weight
        ]), dim=0)

        return {
            'reaction_mean': reaction.mean().item(),
            'physics_weight': weights[0].item(),
            'neural_weight': weights[1].item(),
            'growth_rate': self.growth_rate.item(),
            'decay_rate': self.decay_rate.item(),
        }

    def get_physics_params(self) -> Dict[str, float]:
        """Get current physics parameters."""
        weights = torch.softmax(torch.stack([
            self.physics_weight, self.neural_weight
        ]), dim=0)

        return {
            'growth_rate': self.growth_rate.item(),
            'decay_rate': self.decay_rate.item(),
            'capacity': self.log_capacity.exp().item(),
            'competition_scale': self.competition_scale.item(),
            'physics_weight': weights[0].item(),
            'neural_weight': weights[1].item(),
        }


# =============================================================================
# ODE SOLVER
# =============================================================================

class ODESolver(nn.Module):
    """Simple ODE solver with Euler or RK4 integration."""

    def __init__(self, n_steps: int = 4, method: str = 'euler'):
        super().__init__()
        self.n_steps = n_steps
        self.method = method

    def forward(
        self,
        initial_state: Tensor,
        diffusion_fn,
        reaction_fn,
        T: float = 1.0,
    ) -> Tensor:
        """
        Solve ODE: dstate/dt = diffusion + reaction

        Args:
            initial_state: [n_nodes, 1]
            diffusion_fn: function(state) -> diffusion term
            reaction_fn: function(state) -> reaction term
            T: Integration time

        Returns:
            Final state [n_nodes, 1]
        """
        dt = T / self.n_steps
        state = initial_state.clone()

        for _ in range(self.n_steps):
            if self.method == 'euler':
                state = self._euler_step(state, diffusion_fn, reaction_fn, dt)
            else:
                state = self._rk4_step(state, diffusion_fn, reaction_fn, dt)

        return state

    def _euler_step(self, state, diffusion_fn, reaction_fn, dt):
        dstate = diffusion_fn(state) + reaction_fn(state)
        return state + dt * dstate

    def _rk4_step(self, state, diffusion_fn, reaction_fn, dt):
        def f(s):
            return diffusion_fn(s) + reaction_fn(s)

        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)

        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# =============================================================================
# HORIZON DECODER
# =============================================================================

class HorizonDecoder(nn.Module):
    """Decoder for multi-step predictions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pred_horizon: int,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, pred_horizon),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


# =============================================================================
# MAIN MODEL
# =============================================================================

class GraphPDEEpidemiologyClean(nn.Module):
    """
    Clean GraphPDE model for epidemiology.
    NO CMS, NO SMM, NO M3.
    """

    def __init__(
        self,
        n_nodes: int,
        adjacency: Union[np.ndarray, Tensor],
        input_window: int = 4,
        pred_horizon: int = 15,
        hidden_dim: int = 64,
        n_ode_steps: int = 4,
        ode_method: str = 'euler',
        integration_time: float = 1.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.integration_time = integration_time

        # Store normalization stats
        self.register_buffer('data_mean', torch.tensor(data_mean))
        self.register_buffer('data_std', torch.tensor(data_std))

        # ================================================================
        # DIFFUSION
        # ================================================================
        self.diffusion_coef = nn.Parameter(torch.tensor(0.01))
        self.diffusion_weight = nn.Parameter(torch.tensor(1.0))
        self.graph_laplacian = GraphLaplacian(adjacency)

        # ================================================================
        # REACTION MODULE
        # ================================================================
        self.reaction_module = ReactionModule(
            n_nodes=n_nodes,
            n_features=input_window,
            hidden_dim=hidden_dim,
        )

        # ================================================================
        # ODE SOLVER
        # ================================================================
        self.ode_solver = ODESolver(n_steps=n_ode_steps, method=ode_method)

        # ================================================================
        # HORIZON DECODER
        # ================================================================
        # Input: hidden_dim + 1 (ODE state) + input_window
        decoder_input_dim = hidden_dim + 1 + input_window

        self.decoder_encoder = nn.Sequential(
            nn.Linear(input_window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.horizon_decoder = HorizonDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=hidden_dim,
            pred_horizon=pred_horizon,
        )

        self.global_step = 0

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary with 'input' key [B, n_nodes, input_window]

        Returns:
            Dictionary with 'predictions' [B, n_nodes, pred_horizon]
        """
        device = next(self.parameters()).device

        # Handle both 'input' and 'features' keys
        if 'input' in batch:
            features = batch['input'].to(device)
        else:
            features = batch['features'].to(device)

        batch_size = features.shape[0]

        # Use last time step as initial state
        initial_state = features[:, :, -1:]  # [B, n_nodes, 1]

        predictions = []

        for b in range(batch_size):
            feat_b = features[b]  # [n_nodes, input_window]
            state_b = initial_state[b]  # [n_nodes, 1]

            # Precompute encoder
            self.reaction_module.precompute_features(feat_b)

            # Define reaction function
            def reaction_fn(state: Tensor) -> Tensor:
                return self.reaction_module.forward_fast(state)

            # Define diffusion function
            def diffusion_fn(state: Tensor) -> Tensor:
                D = self.diffusion_coef.abs()
                return self.diffusion_weight * self.graph_laplacian(state, D)

            # Solve ODE
            final_state = self.ode_solver(
                state_b,
                diffusion_fn,
                reaction_fn,
                T=self.integration_time,
            )

            # Clear cache
            self.reaction_module.clear_cache()

            # ================================================================
            # DECODE TO MULTI-STEP PREDICTIONS
            # ================================================================
            encoded_features = self.decoder_encoder(feat_b)

            decoder_input = torch.cat([
                encoded_features,  # [n_nodes, hidden_dim]
                final_state,       # [n_nodes, 1]
                feat_b,            # [n_nodes, input_window]
            ], dim=-1)

            pred = self.horizon_decoder(decoder_input)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [B, n_nodes, pred_horizon]

        self.global_step += 1

        return {'predictions': predictions}

    def clip_parameters(self):
        """Clip parameters for stability."""
        with torch.no_grad():
            self.diffusion_coef.data.clamp_(min=1e-6, max=1.0)
            self.diffusion_weight.data.clamp_(min=0.0, max=10.0)
            self.reaction_module.growth_rate.data.clamp_(min=1e-6, max=1.0)
            self.reaction_module.decay_rate.data.clamp_(min=1e-6, max=1.0)
            self.reaction_module.competition_scale.data.clamp_(min=1e-6, max=0.1)
            self.reaction_module.physics_scale.data.clamp_(min=0.1, max=10.0)

    def get_parameters_dict(self) -> Dict[str, Any]:
        """Get interpretable parameters."""
        params = self.reaction_module.get_physics_params()
        params['diffusion_coef'] = self.diffusion_coef.item()
        params['diffusion_weight'] = self.diffusion_weight.item()
        return params


def create_model(
    n_nodes: int,
    adjacency: np.ndarray,
    input_window: int = 4,
    pred_horizon: int = 15,
    hidden_dim: int = 64,
    n_ode_steps: int = 4,
    ode_method: str = 'euler',
    data_mean: float = 0.0,
    data_std: float = 1.0,
) -> GraphPDEEpidemiologyClean:
    """Factory function to create the model."""
    return GraphPDEEpidemiologyClean(
        n_nodes=n_nodes,
        adjacency=adjacency,
        input_window=input_window,
        pred_horizon=pred_horizon,
        hidden_dim=hidden_dim,
        n_ode_steps=n_ode_steps,
        ode_method=ode_method,
        data_mean=data_mean,
        data_std=data_std,
    )
