"""
Abstract base class for meta-aggregation modules.

These modules aggregate latent vectors from multiple context trajectories
into a single representation, following the Conditional Neural Process paradigm.
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor


class BaseMetaAggregation(nn.Module, ABC):
    """
    Abstract base class for meta-aggregation modules.

    Takes a set of latent vectors and aggregates them into a single vector.

    Input shape: (context_size, latent_dim)
    Output shape: (1, latent_dim)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, latent_vectors: Tensor) -> Tensor:
        """
        Aggregate latent vectors into a single representation.

        Args:
            latent_vectors: [context_size, latent_dim] Latent vectors to aggregate

        Returns:
            aggregated: [1, latent_dim] Aggregated representation
        """
        pass
