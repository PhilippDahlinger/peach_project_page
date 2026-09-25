"""
Graph construction from tokenized point cloud streams.

Constructs a spatio-temporal graph from tokens where:
- Nodes are tokens with Fourier-encoded time features
- Edges connect spatially close nodes within and across timesteps
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
from dataclasses import dataclass
from torch_geometric.nn import knn
from torch_geometric.utils import to_undirected


@dataclass
class TokenGraphConfig:
    """Configuration for token graph construction."""
    k_same_time: int = 4      # Neighbors within same timestep
    k_one_step: int = 2       # Neighbors in adjacent timesteps (t±1)
    k_two_step: int = 1       # Neighbors in timesteps t±2
    num_frequencies: int = 8  # Fourier encoding frequencies


def fourier_encode_time(
    time_indices: Tensor,
    num_frequencies: int,
    max_time: int,
) -> Tensor:
    """
    Apply Fourier positional encoding to time indices.
    
    Args:
        time_indices: [N] tensor of time indices
        num_frequencies: Number of frequency bands
        max_time: Maximum time value for normalization
        
    Returns:
        [N, 2 * num_frequencies] tensor of Fourier features
    """
    # Normalize time to [0, 1]
    t_normalized = time_indices.float() / max_time
    
    # Create frequency bands: 2^0, 2^1, ..., 2^(num_frequencies-1)
    freq_bands = 2.0 ** torch.arange(num_frequencies, device=time_indices.device, dtype=torch.float32)
    
    # Compute angles: [N, num_frequencies]
    angles = t_normalized[:, None] * freq_bands[None, :] * 2.0 * torch.pi
    
    # Concatenate sin and cos: [N, 2 * num_frequencies]
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    return encoding


def build_edges_within_timestep(
    center_points: Tensor,
    time_idx: int,
    node_offset: int,
    num_nodes_per_time: int,
    k: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build edges connecting nodes within the same timestep.
    
    Args:
        center_points: [T, N, 3] center point coordinates
        time_idx: Current timestep index
        node_offset: Global node index offset for this timestep
        num_nodes_per_time: Number of nodes per timestep
        k: Number of nearest neighbors
        
    Returns:
        edge_index: [2, num_edges]
        edge_attr_type: [num_edges, 5] one-hot edge type
        edge_attr_pos: [num_edges, 3] relative positions
    """
    pos = center_points[time_idx]  # [N, 3]
    n_nodes = pos.shape[0]
    
    if n_nodes <= 1 or k <= 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=pos.device),
            torch.zeros(0, 5, device=pos.device),
            torch.zeros(0, 3, device=pos.device),
        )
    
    # Use k+1 because knn includes self-loops, then filter them out
    k_actual = min(k + 1, n_nodes)
    

    # Find k nearest neighbors
    # knn returns (row, col) where row indexes into y (query), col indexes into x (source)
    row, col = knn(pos, pos, k_actual)
    
    # Filter out self-loops
    mask = row != col
    row = row[mask]
    col = col[mask]
    
    # Add node offset for global indexing
    row = row + node_offset
    col = col + node_offset

    # Create bidirectional edges
    edge_index = torch.stack([
        row, col,
    ], dim=0)

    edge_index = to_undirected(edge_index)

    num_edges = edge_index.shape[1]
    
    # Edge type: same_time (index 0)
    edge_type = torch.zeros(num_edges, 5, device=pos.device)
    edge_type[:, 0] = 1.0
    
    # Relative positions
    src_pos = center_points[time_idx][edge_index[0] - node_offset]
    dst_pos = center_points[time_idx][edge_index[1] - node_offset]
    rel_pos = dst_pos - src_pos
    
    return edge_index, edge_type, rel_pos


def build_edges_across_timesteps(
    center_points: Tensor,
    time_idx_from: int,
    time_idx_to: int,
    node_offset_from: int,
    node_offset_to: int,
    k: int,
    time_diff: int,  # Positive = future, negative = past
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build edges connecting nodes across different timesteps.
    
    Args:
        center_points: [T, N, 3] center point coordinates
        time_idx_from: Source timestep index
        time_idx_to: Target timestep index
        node_offset_from: Global node offset for source timestep
        node_offset_to: Global node offset for target timestep
        k: Number of nearest neighbors
        time_diff: Time difference (positive = future, negative = past)
        
    Returns:
        edge_index: [2, num_edges]
        edge_attr_type: [num_edges, 5] one-hot edge type
        edge_attr_pos: [num_edges, 3] relative positions
    """
    pos_from = center_points[time_idx_from]  # [N, 3]
    pos_to = center_points[time_idx_to]      # [N, 3]
    
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]
    
    if n_from == 0 or n_to == 0 or k <= 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=pos_from.device),
            torch.zeros(0, 5, device=pos_from.device),
            torch.zeros(0, 3, device=pos_from.device),
        )
    
    k_actual = min(k, n_to)

    
    # Find k nearest neighbors in target timestep for each node in source
    # row indexes into pos_from (query), col indexes into pos_to (source)
    row, col = knn(pos_to, pos_from, k_actual)
    
    # Add offsets for global indexing
    row = row + node_offset_from  # Source nodes (from)
    col = col + node_offset_to    # Target nodes (to)
    
    # Forward edges: from -> to
    edge_index_fwd = torch.stack([row, col], dim=0)
    
    # Backward edges: to -> from
    edge_index_bwd = torch.stack([col, row], dim=0)
    
    # Combine and make unique
    edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
    
    num_fwd = edge_index_fwd.shape[1]
    num_bwd = edge_index_bwd.shape[1]
    
    # Determine edge type indices:
    # 0: same_time, 1: future_1, 2: future_2, 3: past_1, 4: past_2
    if time_diff == 1:
        type_fwd, type_bwd = 1, 3  # future_1, past_1
    elif time_diff == 2:
        type_fwd, type_bwd = 2, 4  # future_2, past_2
    elif time_diff == -1:
        type_fwd, type_bwd = 3, 1  # past_1, future_1
    elif time_diff == -2:
        type_fwd, type_bwd = 4, 2  # past_2, future_2
    else:
        raise ValueError(f"Unsupported time_diff: {time_diff}")
    
    # Create edge type one-hot
    edge_type = torch.zeros(num_fwd + num_bwd, 5, device=pos_from.device)
    edge_type[:num_fwd, type_fwd] = 1.0
    edge_type[num_fwd:, type_bwd] = 1.0
    
    # Compute relative positions
    # For forward edges
    src_pos_fwd = center_points[time_idx_from][edge_index_fwd[0] - node_offset_from]
    dst_pos_fwd = center_points[time_idx_to][edge_index_fwd[1] - node_offset_to]
    rel_pos_fwd = dst_pos_fwd - src_pos_fwd
    
    # For backward edges
    src_pos_bwd = center_points[time_idx_to][edge_index_bwd[0] - node_offset_to]
    dst_pos_bwd = center_points[time_idx_from][edge_index_bwd[1] - node_offset_from]
    rel_pos_bwd = dst_pos_bwd - src_pos_bwd
    
    rel_pos = torch.cat([rel_pos_fwd, rel_pos_bwd], dim=0)
    
    return edge_index, edge_type, rel_pos


class TokenGraphBuilder(nn.Module):
    """
    Builds a spatio-temporal graph from tokenized point cloud streams.
    
    The graph connects tokens:
    - Within the same timestep (k_same_time neighbors)
    - Across adjacent timesteps t±1 (k_one_step neighbors)
    - Across timesteps t±2 (k_two_step neighbors)
    
    Node features: [token_features, fourier_time_encoding]
    Edge features: [edge_type_one_hot (5), relative_position (3)]
    """
    
    def __init__(
        self,
        k_same_time: int = 4,
        k_one_step: int = 2,
        k_two_step: int = 1,
        num_frequencies: int = 8,
    ):
        """
        Args:
            k_same_time: Number of neighbors within same timestep
            k_one_step: Number of neighbors in adjacent timesteps (t±1)
            k_two_step: Number of neighbors in timesteps t±2
            num_frequencies: Number of Fourier encoding frequencies
            max_time: Maximum time for normalization
        """
        super().__init__()
        self.k_same_time = k_same_time
        self.k_one_step = k_one_step
        self.k_two_step = k_two_step
        self.num_frequencies = num_frequencies

    @property
    def time_encoding_dim(self) -> int:
        """Dimension of Fourier time encoding."""
        return 2 * self.num_frequencies
    
    @property
    def edge_feature_dim(self) -> int:
        """Dimension of edge features (5 one-hot + 3 relative position)."""
        return 8
    
    def forward(
        self,
        tokens: Tensor,
        center_points: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build a spatio-temporal graph from tokens.
        
        Args:
            tokens: [T, N, D] Token features from tokenizer
            center_points: [T, N, 3] Center point coordinates
            
        Returns:
            h: [T*N, D + time_encoding_dim] Node features
            edge_index: [2, num_edges] Edge indices
            edge_attr: [num_edges, 8] Edge features (5 type + 3 position)
        """
        T, N, D = tokens.shape
        device = tokens.device
        
        # === Build node features ===
        # Flatten tokens: [T*N, D]
        h_tokens = tokens.reshape(T * N, D)
        
        # Create time indices for each node
        time_indices = torch.arange(T, device=device).repeat_interleave(N)
        
        # Fourier encode time: [T*N, 2*num_frequencies]
        time_encoding = fourier_encode_time(
            time_indices, self.num_frequencies, torch.max(time_indices).item()
        )
        
        # Concatenate: [T*N, D + 2*num_frequencies]
        h = torch.cat([h_tokens, time_encoding], dim=-1)
        
        # === Build edges ===
        edge_index_list = []
        edge_type_list = []
        edge_pos_list = []

        # TODO: This should be batchable. The same timestep definitely, and the between timesteps probably as well.
        for t in range(T):
            node_offset = t * N
            
            # Edges within same timestep
            ei, et, ep = build_edges_within_timestep(
                center_points, t, node_offset, N, self.k_same_time
            )
            edge_index_list.append(ei)
            edge_type_list.append(et)
            edge_pos_list.append(ep)
            
            # Edges to t+1
            if t + 1 < T:
                ei, et, ep = build_edges_across_timesteps(
                    center_points, t, t + 1, 
                    node_offset, (t + 1) * N,
                    self.k_one_step, time_diff=1
                )
                edge_index_list.append(ei)
                edge_type_list.append(et)
                edge_pos_list.append(ep)
            
            # Edges to t+2
            if t + 2 < T:
                ei, et, ep = build_edges_across_timesteps(
                    center_points, t, t + 2,
                    node_offset, (t + 2) * N,
                    self.k_two_step, time_diff=2
                )
                edge_index_list.append(ei)
                edge_type_list.append(et)
                edge_pos_list.append(ep)
            # TODO: Right now we are only going into the future (although bidirectional edges are created). We could also add knn edges to past timesteps (t-1, t-2). Then we have to remove duplicate edges though.
            # TODO: For now, we skip edges to past timesteps to reduce the total number of edges. But we should try that out later.
        
        # Concatenate all edges
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list, dim=0)
        edge_pos = torch.cat(edge_pos_list, dim=0)
        
        # Combine edge features: [num_edges, 8]
        edge_attr = torch.cat([edge_type, edge_pos], dim=-1)
        
        return h, edge_index, edge_attr
    
    @classmethod
    def from_config(cls, config: TokenGraphConfig) -> "TokenGraphBuilder":
        """Create from config dataclass."""
        return cls(
            k_same_time=config.k_same_time,
            k_one_step=config.k_one_step,
            k_two_step=config.k_two_step,
            num_frequencies=config.num_frequencies,
        )

