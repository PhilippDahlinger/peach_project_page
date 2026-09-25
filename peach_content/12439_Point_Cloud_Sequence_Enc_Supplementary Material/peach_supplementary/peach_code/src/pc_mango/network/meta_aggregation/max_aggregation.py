"""
Max aggregation module for meta-learning.
"""

import torch
from torch import Tensor

from pc_mango.network.meta_aggregation.base import BaseMetaAggregation


class MaxAggregation(BaseMetaAggregation):
    """
    Max pooling aggregation over context trajectories.
    
    Takes element-wise maximum over the context dimension.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, latent_vectors: Tensor) -> Tensor:
        """
        Aggregate latent vectors using max pooling.
        
        Args:
            latent_vectors: [context_size, latent_dim] Latent vectors to aggregate
            
        Returns:
            aggregated: [1, latent_dim] Max-pooled representation
        """
        # Take max over context dimension (dim=0)
        aggregated = torch.max(latent_vectors, dim=0)[0]
        # Add batch dimension back
        return aggregated[None, :]

