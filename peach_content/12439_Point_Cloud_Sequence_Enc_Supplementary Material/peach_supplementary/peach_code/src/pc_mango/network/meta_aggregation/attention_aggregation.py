"""
Attention-based aggregation module for meta-learning.
"""

import torch
import torch.nn as nn
from torch import Tensor

from pc_mango.network.meta_aggregation.base import BaseMetaAggregation


class AttentionAggregation(BaseMetaAggregation):
    """
    Attention-based aggregation over context trajectories.

    Uses a learned attention mechanism to weight context vectors
    before aggregating. This is similar to the attention mechanism
    used in Attentive Neural Processes.

    The attention scores are computed using a simple MLP that
    maps each latent vector to a scalar attention weight.
    """

    def __init__(self, config):
        super().__init__(config)

        latent_dim = config.latent_dim
        hidden_dim = getattr(config, "hidden_dim", latent_dim)

        # Attention network: maps latent -> scalar attention score
        self.attention_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent_vectors: Tensor) -> Tensor:
        """
        Aggregate latent vectors using learned attention weights.

        Args:
            latent_vectors: [context_size, latent_dim] Latent vectors to aggregate

        Returns:
            aggregated: [1, latent_dim] Attention-weighted representation
        """
        # Compute attention scores: [context_size, 1]
        attention_scores = self.attention_net(latent_vectors)

        # Softmax over context dimension
        attention_weights = torch.softmax(attention_scores, dim=0)

        # Weighted sum: [latent_dim]
        aggregated = torch.sum(attention_weights * latent_vectors, dim=0)

        # Add batch dimension back
        return aggregated[None, :]

