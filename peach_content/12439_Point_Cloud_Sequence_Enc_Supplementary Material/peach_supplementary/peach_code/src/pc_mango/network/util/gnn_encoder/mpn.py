import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from pc_mango.network.simulator.egno.basic import BaseMLP


@dataclass
class BatchedGraph:
    """
    Container for a batched graph following PyG's batching principle:
    Multiple graphs are represented as one big disconnected graph.

    Attributes:
        h: Node features [total_num_nodes, node_dim]
        edge_index: Edge indices [2, total_num_edges]
        edge_attr: Edge features [total_num_edges, edge_dim]
        batch_idx: Node-to-graph assignment [total_num_nodes]
        num_graphs: Number of graphs in the batch
    """
    h: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch_idx: torch.Tensor
    num_graphs: int


def collate_graphs(
    graphs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> BatchedGraph:
    """
    Convert a list of (h, edge_index, edge_attr) tuples to a BatchedGraph.

    Args:
        graphs: List of tuples, each containing:
            - h: [num_nodes_i, node_dim]
            - edge_index: [2, num_edges_i]
            - edge_attr: [num_edges_i, edge_dim]

    Returns:
        BatchedGraph with all graphs concatenated
    """
    h_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_idx_list = []

    node_offset = 0

    for graph_idx, (h, edge_index, edge_attr) in enumerate(graphs):
        num_nodes = h.shape[0]

        h_list.append(h)
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        batch_idx_list.append(torch.full((num_nodes,), graph_idx, dtype=torch.long, device=h.device))

        node_offset += num_nodes

    return BatchedGraph(
        h=torch.cat(h_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_attr=torch.cat(edge_attr_list, dim=0),
        batch_idx=torch.cat(batch_idx_list, dim=0),
        num_graphs=len(graphs),
    )


# @torch.compile
def mpn_aggregate(message, row, num_nodes, aggr="sum"):
    """
    message: [num_edges, latent_dim]
    row:     [num_edges]  (destination node indices)
    """
    latent_dim = message.shape[-1]
    out = torch.zeros(num_nodes, latent_dim, device=message.device)

    out.scatter_add_(0, row[:, None].expand(-1, latent_dim), message)

    if aggr == "mean":
        count = torch.zeros(num_nodes, latent_dim, device=message.device)
        ones = torch.ones_like(message)
        count.scatter_add_(0, row[:, None].expand(-1, latent_dim), ones)
        out = out / count.clamp(min=1)

    elif aggr != "sum":
        raise ValueError(f"Unknown aggregation method: {aggr}")

    return out


def drop_edge(edge_index: torch.Tensor, edge_attr: torch.Tensor, p: float):
    """Randomly drop edges with probability p (independent per edge).

    Returns new (edge_index, edge_attr). Works with edge_attr=None as well.
    """
    if p <= 0.0:
        return edge_index, edge_attr
    if p >= 1.0:
        device = edge_index.device
        empty_idx = torch.empty((2, 0), dtype=edge_index.dtype, device=device)
        empty_attr = None if edge_attr is None else edge_attr.new_empty((0, edge_attr.shape[1]))
        return empty_idx, empty_attr

    num_edges = edge_index.shape[1]
    mask = torch.rand(num_edges, device=edge_index.device) > p

    # If all kept or all dropped, handle explicitly
    if mask.all():
        return edge_index, edge_attr
    if (~mask).all():
        device = edge_index.device
        empty_idx = torch.empty((2, 0), dtype=edge_index.dtype, device=device)
        empty_attr = None if edge_attr is None else edge_attr.new_empty((0, edge_attr.shape[1]))
        return empty_idx, empty_attr

    new_edge_index = edge_index[:, mask]
    new_edge_attr = None if edge_attr is None else edge_attr[mask]
    return new_edge_index, new_edge_attr


class BatchedMPN(nn.Module):
    """
    Batched Message Passing Network that operates on BatchedGraph inputs.
    Uses PyG-style batching: multiple graphs as one big disconnected graph.
    """

    def __init__(
        self,
        n_layers: int,
        node_dim: int,
        edge_dim: int,
        latent_dim: int,
        activation="leakyrelu",
        aggr="mean",
        use_hidden_layers=False,
        dropout: float = 0.0,
        dropout_embedding: float = None,
        dropout_node: float = None,
        dropout_edge: float = None,
        drop_edge_p: float = 0.0,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.aggr = aggr
        self.latent_dim = latent_dim
        self.drop_edge_p = float(drop_edge_p)

        # Backwards compatibility
        if dropout_embedding is None:
            dropout_embedding = dropout
        if dropout_node is None:
            dropout_node = dropout
        if dropout_edge is None:
            dropout_edge = dropout

        # Activation
        if activation == "leakyrelu":
            act = nn.LeakyReLU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        hidden_dim = latent_dim if use_hidden_layers else -1

        # Input embeddings
        self.node_embedding = nn.Linear(node_dim, latent_dim)
        self.edge_embedding = nn.Linear(edge_dim, latent_dim)

        # Embedding dropout
        self.dropout_embedding = float(dropout_embedding)
        if self.dropout_embedding > 0:
            self.node_embedding = nn.Sequential(
                self.node_embedding, nn.Dropout(self.dropout_embedding)
            )
            self.edge_embedding = nn.Sequential(
                self.edge_embedding, nn.Dropout(self.dropout_embedding)
            )

        # Node / Edge dropouts
        self.node_dropout = nn.Dropout(float(dropout_node)) if float(dropout_node) > 0 else nn.Identity()
        self.edge_dropout = nn.Dropout(float(dropout_edge)) if float(dropout_edge) > 0 else nn.Identity()

        # Per-layer modules
        self.edge_update_modules = nn.ModuleList()
        self.edge_layer_norms = nn.ModuleList()
        self.node_update_modules = nn.ModuleList()
        self.node_layer_norms = nn.ModuleList()

        for _ in range(n_layers):
            self.edge_update_modules.append(
                BaseMLP(
                    input_dim=3 * latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=latent_dim,
                    activation=act,
                    residual=False,
                    last_act=False,
                )
            )
            self.edge_layer_norms.append(nn.LayerNorm(latent_dim))

            self.node_update_modules.append(
                BaseMLP(
                    input_dim=2 * latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=latent_dim,
                    activation=act,
                    residual=False,
                    last_act=False,
                )
            )
            self.node_layer_norms.append(nn.LayerNorm(latent_dim))

    def forward(self, batch: BatchedGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on a batched graph.

        Args:
            batch: BatchedGraph containing the batched input

        Returns:
            h: Processed node features [total_num_nodes, latent_dim]
            batch_idx: Node-to-graph assignment [total_num_nodes]
        """
        h = batch.h
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        # Optionally drop edges during training
        if self.drop_edge_p > 0.0 and self.training and edge_index is not None:
            edge_index, edge_attr = drop_edge(edge_index, edge_attr, p=self.drop_edge_p)

        # If there are no edges, handle gracefully
        if edge_index.numel() == 0:
            # still run embeddings
            h = self.node_embedding(h)
            # create empty message tensor with correct latent dim
            if isinstance(self.edge_embedding, nn.Sequential):
                emb_layer = self.edge_embedding[0]
            else:
                emb_layer = self.edge_embedding
            out_msg = emb_layer.weight.new_empty((0, self.latent_dim))
            return h, out_msg

        row, col = edge_index
        num_nodes = h.shape[0]

        # Embedding
        h = self.node_embedding(h)
        message = self.edge_embedding(edge_attr)

        for i in range(self.n_layers):
            # -------- Edge update --------
            edge_update = self.edge_update_modules[i]
            edge_ln = self.edge_layer_norms[i]

            message_normed = edge_ln(message)

            edge_input = torch.cat(
                [message_normed, h[row], h[col]], dim=-1
            )

            # Apply edge update MLP, then dropout, then layer-norm, then residual
            edge_updated = edge_update(edge_input)
            edge_dropped = self.edge_dropout(edge_updated)
            message = edge_dropped + message

            # -------- Node aggregation --------
            aggr_message = mpn_aggregate(
                message, row, num_nodes, aggr=self.aggr
            )

            # -------- Node update --------
            node_update = self.node_update_modules[i]
            node_ln = self.node_layer_norms[i]

            h_normed = node_ln(h)

            node_input = torch.cat([h_normed, aggr_message], dim=-1)
            # Apply node update MLP, then dropout, then layer-norm, then residual
            node_updated = node_update(node_input)
            node_dropped = self.node_dropout(node_updated)
            h = node_dropped + h

        return h, batch.batch_idx


# ============== Graph Readout Classes ==============

class GraphReadout(ABC, nn.Module):
    """Abstract base class for graph-level readout operations."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node features to graph-level features.

        Args:
            h: Node features [total_num_nodes, latent_dim]
            batch_idx: Node-to-graph assignment [total_num_nodes]

        Returns:
            Graph-level features [num_graphs, latent_dim]
        """
        pass


class MeanReadout(GraphReadout):
    """Mean pooling readout: average node features per graph."""

    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        num_graphs = batch_idx.max().item() + 1

        # Sum aggregation
        out = torch.zeros(num_graphs, self.latent_dim, device=h.device, dtype=h.dtype)
        out.scatter_add_(0, batch_idx[:, None].expand(-1, self.latent_dim), h)

        # Count nodes per graph for mean
        count = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        count.scatter_add_(0, batch_idx, torch.ones(batch_idx.shape[0], device=h.device, dtype=h.dtype))

        return out / count[:, None].clamp(min=1)


class MaxReadout(GraphReadout):
    """Max pooling readout: max over node features per graph (per feature dimension)."""

    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        num_graphs = batch_idx.max().item() + 1

        # Initialize with very negative values
        out = torch.full(
            (num_graphs, self.latent_dim),
            float('-inf'),
            device=h.device,
            dtype=h.dtype
        )

        # Scatter max
        out.scatter_reduce_(
            0,
            batch_idx[:, None].expand(-1, self.latent_dim),
            h,
            reduce='amax',
            include_self=False
        )

        return out


class AttentionReadout(GraphReadout):
    """
    Attention Rollout (Global Attention Pooling) readout.

    Computes attention scores per node and performs weighted sum per graph.
    s_i = g(h_i), alpha_i = softmax(s_i) within each graph,
    h_G = sum_i alpha_i * h_i
    """

    def __init__(self, latent_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__(latent_dim)
        hidden_dim = hidden_dim or latent_dim

        # Scoring function g: R^d -> R
        self.score_fn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Dropout applied to weighted features before aggregation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        num_graphs = batch_idx.max().item() + 1

        # Compute scores: [total_num_nodes, 1]
        scores = self.score_fn(h).squeeze(-1)  # [total_num_nodes]

        # Compute softmax per graph using the log-sum-exp trick for numerical stability
        # First, compute max score per graph
        max_scores = torch.full((num_graphs,), float('-inf'), device=h.device, dtype=h.dtype)
        max_scores.scatter_reduce_(0, batch_idx, scores, reduce='amax', include_self=False)

        # Subtract max for numerical stability
        scores_normalized = scores - max_scores[batch_idx]

        # Compute exp(scores)
        exp_scores = torch.exp(scores_normalized)

        # Sum exp(scores) per graph
        sum_exp = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        sum_exp.scatter_add_(0, batch_idx, exp_scores)

        # Compute attention weights
        alpha = exp_scores / sum_exp[batch_idx].clamp(min=1e-8)  # [total_num_nodes]

        # Weighted sum of node features
        weighted_h = alpha[:, None] * h  # [total_num_nodes, latent_dim]

        # Apply dropout before aggregation
        weighted_h = self.dropout(weighted_h)

        out = torch.zeros(num_graphs, self.latent_dim, device=h.device, dtype=h.dtype)
        out.scatter_add_(0, batch_idx[:, None].expand(-1, self.latent_dim), weighted_h)

        return out


class GatedPoolingReadout(GraphReadout):
    """
    Gated Pooling readout.

    Applies learned gating: h_tilde = W @ h, g = sigmoid(W_g @ h),
    m = g * h_tilde, h_G = sum_i m_i
    """

    def __init__(self, latent_dim: int, dropout: float = 0.0):
        super().__init__(latent_dim)

        # Linear projection W
        self.proj = nn.Linear(latent_dim, latent_dim)
        # Gating projection W_g
        self.gate = nn.Linear(latent_dim, latent_dim)

        # Dropout applied to gated features before aggregation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        num_graphs = batch_idx.max().item() + 1

        # Compute projections
        h_tilde = self.proj(h)  # [total_num_nodes, latent_dim]
        g = torch.sigmoid(self.gate(h))  # [total_num_nodes, latent_dim]

        # Element-wise gating
        m = g * h_tilde  # [total_num_nodes, latent_dim]

        # Apply dropout before aggregation
        m = self.dropout(m)

        # Sum aggregation per graph
        out = torch.zeros(num_graphs, self.latent_dim, device=h.device, dtype=h.dtype)
        out.scatter_add_(0, batch_idx[:, None].expand(-1, self.latent_dim), m)

        return out
