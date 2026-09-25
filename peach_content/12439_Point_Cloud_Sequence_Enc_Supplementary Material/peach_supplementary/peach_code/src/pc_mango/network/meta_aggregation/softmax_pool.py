"""
Softmax pool aggregation module for meta-learning.
"""

import torch
import torch.nn as nn
from torch import Tensor

from pc_mango.network.meta_aggregation.base import BaseMetaAggregation


class SoftmaxPoolAggregation(BaseMetaAggregation):
    """
    Softmax pooling aggregation over context trajectories.

    Uses softmax-weighted sum where the temperature (beta) controls
    the smoothness of the aggregation:
    - Higher beta -> more max-like behavior
    - Lower beta -> more mean-like behavior

    The beta parameter can optionally be learned during training.
    """

    def __init__(self, config):
        super().__init__(config)

        # Initialize beta as a parameter (learnable) or buffer (fixed)
        initial_beta = config.beta_init
        if config.beta_trainable:
            self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(initial_beta, dtype=torch.float32))

    def forward(self, latent_vectors: Tensor) -> Tensor:
        """
        Aggregate latent vectors using softmax pooling.

        Args:
            latent_vectors: [context_size, latent_dim] Latent vectors to aggregate

        Returns:
            aggregated: [1, latent_dim] Softmax-pooled representation
        """
        # Compute softmax weights over context dimension
        # weights shape: [context_size, latent_dim]
        weights = torch.softmax(self.beta * latent_vectors, dim=0)

        # Weighted sum over context dimension
        aggregated = torch.sum(weights * latent_vectors, dim=0)

        # Add batch dimension back
        return aggregated[None, :]
