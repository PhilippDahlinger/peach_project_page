"""
Gated pooling aggregation module for meta-learning.
"""

import torch
import torch.nn as nn
from torch import Tensor

from pc_mango.network.meta_aggregation.base import BaseMetaAggregation


class GatedAggregation(BaseMetaAggregation):
    """
    Gated pooling aggregation over context trajectories.
    
    Uses a gating mechanism (similar to GRU gates) to learn which
    features from each context vector should be retained. The gate
    values are then used to weight the sum.
    
    This allows the model to learn feature-wise importance rather
    than just sample-wise importance (like attention).
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        latent_dim = config.latent_dim
        
        # Gate network: maps latent -> gate values per feature
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, latent_vectors: Tensor) -> Tensor:
        """
        Aggregate latent vectors using gated pooling.
        
        Args:
            latent_vectors: [context_size, latent_dim] Latent vectors to aggregate
            
        Returns:
            aggregated: [1, latent_dim] Gated representation
        """
        # Compute gate values: [context_size, latent_dim]
        gate_values = self.gate_net(latent_vectors)
        
        # Apply gates and sum
        gated = gate_values * latent_vectors
        
        # Sum over context dimension and normalize
        aggregated = torch.sum(gated, dim=0) / latent_vectors.shape[0]
        
        # Add batch dimension back
        return aggregated[None, :]
