"""
Set Transformer aggregation module for meta-learning.

Based on "Set Transformer: A Framework for Attention-based 
Permutation-Invariant Neural Networks" (Lee et al., 2019)
"""

import torch
import torch.nn as nn
from torch import Tensor

from pc_mango.network.meta_aggregation.base import BaseMetaAggregation


class MultiheadAttentionBlock(nn.Module):
    """Multihead attention block with residual connection and layer norm."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # Multihead attention
        attn_out, _ = self.attention(query, key, value)
        # Residual + norm
        return self.norm(query + self.dropout(attn_out))


class SetAttentionBlock(nn.Module):
    """Self-attention block for sets (SAB)."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: Tensor) -> Tensor:
        # Self-attention
        h = self.mab(x, x, x)
        # Feed-forward with residual
        return self.norm(h + self.ff(h))


class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA) - pools set to fixed size."""
    
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        # Learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, set_size, dim] or [set_size, dim]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim
        
        batch_size = x.shape[0]
        seeds = self.seeds.expand(batch_size, -1, -1)
        
        # Attend from seeds to set
        return self.mab(seeds, x, x)


class SetTransformerAggregation(BaseMetaAggregation):
    """
    Set Transformer aggregation over context trajectories.
    
    Uses Induced Set Attention Blocks (ISAB) for efficient 
    processing and Pooling by Multihead Attention (PMA) for
    aggregation. This provides a more expressive aggregation
    than simple pooling while remaining permutation invariant.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        latent_dim = config.latent_dim
        num_heads = getattr(config, "num_heads", 4)
        num_isab_layers = getattr(config, "num_isab_layers", 2)
        dropout = getattr(config, "dropout", 0.0)

        # Stack of Set Attention Blocks
        self.encoder = nn.ModuleList([
            SetAttentionBlock(latent_dim, num_heads, dropout)
            for _ in range(num_isab_layers)
        ])
        
        # Pooling by multihead attention to single vector
        self.pooling = PoolingByMultiheadAttention(
            latent_dim, num_heads, num_seeds=1, dropout=dropout
        )
    
    def forward(self, latent_vectors: Tensor) -> Tensor:
        """
        Aggregate latent vectors using Set Transformer.
        
        Args:
            latent_vectors: [context_size, latent_dim] Latent vectors to aggregate
            
        Returns:
            aggregated: [1, latent_dim] Set Transformer representation
        """
        # Add batch dimension: [1, context_size, latent_dim]
        x = latent_vectors.unsqueeze(0)
        
        # Process through SAB layers
        for sab in self.encoder:
            x = sab(x)
        
        # Pool to single vector: [1, 1, latent_dim]
        aggregated = self.pooling(x)
        
        # Remove extra dimension: [1, latent_dim]
        return aggregated.squeeze(1)
