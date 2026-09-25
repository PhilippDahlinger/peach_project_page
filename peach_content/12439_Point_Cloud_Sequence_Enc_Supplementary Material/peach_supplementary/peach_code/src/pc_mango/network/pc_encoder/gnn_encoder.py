"""
GNN Encoder for point cloud trajectory encoding.

This module provides a graph neural network encoder that processes
streams of point clouds over time and produces trajectory-level embeddings.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from pc_mango.network.pc_encoder.tokenizer import get_tokenizer
from pc_mango.network.pc_encoder.tokenizer.pointnet_tokenizer import PointnetTokenizer
from pc_mango.network.util.gnn_encoder.token_graph_builder import TokenGraphBuilder
from pc_mango.network.util.gnn_encoder.mpn import (
    BatchedMPN,
    collate_graphs,
    MeanReadout,
    MaxReadout,
    AttentionReadout,
    GatedPoolingReadout,
)


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for point cloud trajectories.
    
    Takes multiple streams of point clouds over time and produces
    one embedding vector per trajectory.
    
    Pipeline:
    1. Tokenize each timestep's point cloud using PPRL tokenizer
    2. Build a spatio-temporal graph from tokens
    3. Batch all trajectory graphs together
    4. Process through message passing network
    5. Readout to get trajectory-level embeddings
    
    Input shape: (num_trajectories, num_timesteps, num_points, 3)
    Output shape: (num_trajectories, output_dim)
    """

    def __init__(self, config, example_batch=None):
        """
        Initialize the GNN Encoder.
        
        Args:
            config: Configuration object with tokenizer, graph, mpn, and readout settings
            example_batch: Optional example batch for shape inference (unused, for API compatibility)
        """
        super().__init__()
        self.config = config

        # Build tokenizer
        self.tokenizer = self._build_tokenizer(config.tokenizer, example_batch)

        # Build graph builder
        self.graph_builder = self._build_graph_builder(config.graph)

        # Compute dimensions
        token_dim = config.tokenizer.embed_dim
        time_encoding_dim = 2 * config.graph.time_encoding.num_frequencies
        node_dim = token_dim + time_encoding_dim
        edge_dim = 8  # 5 edge types + 3 relative position

        # Build MPN
        self.mpn = self._build_mpn(config.mpn, node_dim, edge_dim)

        # Build readout
        latent_dim = config.mpn.latent_dim
        self.readout = self._build_readout(config.readout, latent_dim)

        # Output projection if needed
        output_dim = config.output_dim
        if output_dim != latent_dim:
            self.output_proj = nn.Linear(latent_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def _build_tokenizer(self, tokenizer_config, example_batch) -> PointnetTokenizer:
        return get_tokenizer(tokenizer_config, example_batch)

    def _build_graph_builder(self, graph_config) -> TokenGraphBuilder:
        """Build the token graph builder from config."""
        return TokenGraphBuilder(
            k_same_time=graph_config.k_same_time,
            k_one_step=graph_config.k_one_step,
            k_two_step=graph_config.k_two_step,
            num_frequencies=graph_config.time_encoding.num_frequencies,
        )

    def _build_mpn(self, mpn_config, node_dim: int, edge_dim: int) -> BatchedMPN:
        """Build the message passing network from config."""
        return BatchedMPN(
            n_layers=mpn_config.n_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=mpn_config.latent_dim,
            activation=mpn_config.activation,
            aggr=mpn_config.aggr,
            use_hidden_layers=mpn_config.use_hidden_layers,
            dropout_embedding=mpn_config.dropout_embedding,
            dropout_node=mpn_config.dropout_node,
            dropout_edge=mpn_config.dropout_edge,
            drop_edge_p=mpn_config.drop_edge_p,
        )

    def _build_readout(self, readout_config, latent_dim: int) -> nn.Module:
        """Build the readout module from config."""
        readout_type = readout_config.type.lower()

        if readout_type == "mean":
            return MeanReadout(latent_dim)
        elif readout_type == "max":
            return MaxReadout(latent_dim)
        elif readout_type == "attention":
            hidden_dim = readout_config.attention_hidden_dim
            return AttentionReadout(latent_dim, hidden_dim=hidden_dim, dropout=readout_config.dropout)
        elif readout_type == "gated":
            return GatedPoolingReadout(latent_dim, dropout=readout_config.dropout)
        else:
            raise ValueError(f"Unknown readout type: {readout_type}")

    def tokenize_trajectory(
            self,
            trajectory: Tensor,
            trajectory_color: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Tokenize a single trajectory of point clouds.
        
        Args:
            trajectory: [num_timesteps, num_points, 3] Point cloud trajectory
            trajectory_color: [num_timesteps, num_points, 1] Color information
            
        Returns:
            tokens: [num_timesteps, num_tokens, embed_dim] Token features
            center_points: [num_timesteps, num_tokens, 3] Token center positions
        """
        num_timesteps, num_points, _ = trajectory.shape
        device = trajectory.device
        # build batch indices tensor
        batch_idx = torch.arange(num_timesteps, device=device)
        batch_idx = batch_idx.view(-1, 1).repeat(1, num_points).view(-1)
        pc = trajectory.view(-1, 3)  # shape [T*N, 3]
        color = trajectory_color.view(-1, trajectory_color.shape[-1])  # shape [T*N, 3 or 1]
        # Tokenize: returns (tokens, neighborhoods, center_points)
        tokens, center_points, _ = self.tokenizer(pc, batch_idx, color=color)  # shape [T, N, D], [T, N, 3], _
        return tokens, center_points

    def forward(self, xyzs: Tensor, colors) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the GNN encoder.
        
        Args:
            xyzs: [num_trajectories, num_timesteps, num_points, 3] 
                  Multiple point cloud trajectories
            colors: [num_trajectories, num_timesteps, num_points, 1]
                  
        Returns:
            embeddings: [num_trajectories, output_dim] 
                       One embedding per trajectory
        """
        num_trajectories = xyzs.shape[0]
        device = xyzs.device

        # Process each trajectory and build graphs
        graphs = []

        for traj_idx in range(num_trajectories):
            # Get trajectory: [num_timesteps, num_points, 3]
            trajectory = xyzs[traj_idx]
            trajectory_color = colors[traj_idx]

            # Tokenize: get tokens and center points
            tokens, center_points = self.tokenize_trajectory(trajectory, trajectory_color)

            # Build graph from tokens
            h, edge_index, edge_attr = self.graph_builder(tokens, center_points)

            graphs.append((h, edge_index, edge_attr))

        # Batch all graphs together
        batch = collate_graphs(graphs)

        # Process through MPN
        h_out, batch_idx = self.mpn(batch)

        # Readout to get trajectory-level embeddings
        trajectory_embeddings = self.readout(h_out, batch_idx)

        # Project to output dimension
        output = self.output_proj(trajectory_embeddings)

        return output, tokens, center_points

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.config.output_dim


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(config_path="../../../../config/network/pc_encoder", version_base="1.3.2"):
        config = compose(config_name="gnn_encoder")

    # Create example input
    # 2 trajectories, 10 timesteps, 512 points per timestep
    xyzs = torch.randn(2, 10, 512, 3)

    # Create model
    model = GNNEncoder(config, None)

    # Forward pass
    output = model(xyzs)
    print(f"Input shape: {xyzs.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({xyzs.shape[0]}, {config.output_dim})")
