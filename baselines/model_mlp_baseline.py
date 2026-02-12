"""
model_mlp_baseline.py

Multi-Layer Perceptron baseline for multi-ethnic population prediction.
Similar to XGBoost but with neural network - no spatial structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EthnicityEmbeddingMLP(nn.Module):
    """Embed ethnicity as a learnable vector"""
    def __init__(self, n_ethnicities, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_ethnicities, embedding_dim)
    
    def forward(self, ethnicity_idx):
        return self.embedding(ethnicity_idx)


class FeatureEncoder(nn.Module):
    """Encode input features (census + other ethnicities + period)"""
    def __init__(self, n_input_features, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_input_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, features):
        return self.encoder(features)


class PopulationPredictorMLP(nn.Module):
    """Predict population combining encoded features + ethnicity embedding"""
    def __init__(self, feature_dim, ethnicity_dim, hidden_dim=128):
        super().__init__()
        
        input_dim = feature_dim + ethnicity_dim
        
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
    
    def forward(self, features, ethnicity_embedding):
        combined = torch.cat([features, ethnicity_embedding], dim=-1)
        pred = self.predictor(combined)
        return pred.squeeze(-1)


class MLPBaseline(nn.Module):
    """
    Complete MLP baseline for multi-ethnic population prediction.
    
    Architecture:
    1. Encode features (census + other ethnicities + period)
    2. Embed ethnicity
    3. Predict population
    
    No spatial structure - treats each sample independently like XGBoost.
    """
    
    def __init__(
        self,
        n_census_features,
        n_ethnicities,
        feature_encoder_dim=256,
        ethnicity_embed_dim=32,
        predictor_hidden_dim=128,
        dropout=0.2
    ):
        super().__init__()
        
        self.n_census_features = n_census_features
        self.n_ethnicities = n_ethnicities
        
        # Input features: census + other ethnicities
        # Note: The dataloader provides ALL other ethnicities (including the target one)
        # So we have n_census_features + n_ethnicities features
        n_input_features = n_census_features + n_ethnicities
        
        print(f"\nMLP Model Input Dimension Calculation:")
        print(f"  Census features: {n_census_features}")
        print(f"  Other ethnic features: {n_ethnicities}")
        print(f"  Total input features: {n_input_features}")
        
        # Components
        self.feature_encoder = FeatureEncoder(
            n_input_features, feature_encoder_dim
        )
        
        self.ethnicity_embedding = EthnicityEmbeddingMLP(
            n_ethnicities, ethnicity_embed_dim
        )
        
        self.predictor = PopulationPredictorMLP(
            feature_dim=feature_encoder_dim // 2,
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
        
        Returns:
            predictions: (batch_size,) - predicted populations
        """
        # Get features (already includes census + other ethnicities from dataloader)
        features = batch['features']
        
        # Encode features
        encoded_features = self.feature_encoder(features)
        
        # Get ethnicity embeddings
        ethnicity_idx = batch['ethnicity']
        ethnicity_embed = self.ethnicity_embedding(ethnicity_idx)
        
        # Predict population
        predictions = self.predictor(encoded_features, ethnicity_embed)
        
        # Ensure non-negative predictions
        predictions = F.softplus(predictions)
        
        return predictions


def create_mlp_model(
    n_census_features,
    n_ethnicities,
    feature_encoder_dim=256,
    ethnicity_embed_dim=32,
    predictor_hidden_dim=128,
    dropout=0.2
):
    """
    Factory function to create MLP baseline model.
    
    Args:
        n_census_features: Number of census features (74)
        n_ethnicities: Number of ethnic groups (9)
        feature_encoder_dim: Hidden dimension for feature encoder
        ethnicity_embed_dim: Ethnicity embedding dimension
        predictor_hidden_dim: Hidden dimension for predictor
        dropout: Dropout rate
    
    Returns:
        MLP model
    """
    model = MLPBaseline(
        n_census_features=n_census_features,
        n_ethnicities=n_ethnicities,
        feature_encoder_dim=feature_encoder_dim,
        ethnicity_embed_dim=ethnicity_embed_dim,
        predictor_hidden_dim=predictor_hidden_dim,
        dropout=dropout
    )
    
    return model