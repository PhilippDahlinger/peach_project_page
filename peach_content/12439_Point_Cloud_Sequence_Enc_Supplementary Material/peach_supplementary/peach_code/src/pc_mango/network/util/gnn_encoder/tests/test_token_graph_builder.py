"""
Test cases for TokenGraphBuilder and related functions.
"""

import pytest
import torch
from torch import Tensor

from pc_mango.network.util.gnn_encoder.token_graph_builder import (
    TokenGraphBuilder,
    TokenGraphConfig,
    fourier_encode_time,
    build_edges_within_timestep,
    build_edges_across_timesteps,
)


# ============== Fixtures ==============

@pytest.fixture
def simple_tokens():
    """Create simple tokens: 3 timesteps, 5 nodes each, 8-dim features."""
    T, N, D = 3, 5, 8
    tokens = torch.randn(T, N, D)
    # Create center points in a grid-like pattern for predictable neighbors
    center_points = torch.zeros(T, N, 3)
    for t in range(T):
        for n in range(N):
            center_points[t, n] = torch.tensor([n * 1.0, t * 0.1, 0.0])
    return tokens, center_points


@pytest.fixture
def larger_tokens():
    """Create larger tokens: 5 timesteps, 10 nodes each, 16-dim features."""
    T, N, D = 5, 10, 16
    tokens = torch.randn(T, N, D)
    center_points = torch.randn(T, N, 3)
    return tokens, center_points


@pytest.fixture
def graph_builder():
    """Create a default TokenGraphBuilder."""
    return TokenGraphBuilder(
        k_same_time=4,
        k_one_step=2,
        k_two_step=1,
        num_frequencies=8,
        max_time=100,
    )


# ============== Tests for Fourier Encoding ==============

class TestFourierEncoding:

    def test_output_shape(self):
        """Test that Fourier encoding produces correct output shape."""
        time_indices = torch.tensor([0, 1, 2, 3, 4])
        encoding = fourier_encode_time(time_indices, num_frequencies=8, max_time=100)
        assert encoding.shape == (5, 16)  # 2 * 8 = 16

    def test_different_times_different_encodings(self):
        """Test that different times produce different encodings."""
        time_indices = torch.tensor([0, 1, 2])
        encoding = fourier_encode_time(time_indices, num_frequencies=8, max_time=100)

        # Each row should be different
        assert not torch.allclose(encoding[0], encoding[1])
        assert not torch.allclose(encoding[1], encoding[2])
        assert not torch.allclose(encoding[0], encoding[2])

    def test_same_time_same_encoding(self):
        """Test that same time produces same encoding."""
        time_indices = torch.tensor([5, 5, 5])
        encoding = fourier_encode_time(time_indices, num_frequencies=8, max_time=100)

        assert torch.allclose(encoding[0], encoding[1])
        assert torch.allclose(encoding[1], encoding[2])

    def test_encoding_bounded(self):
        """Test that encoding values are bounded by [-1, 1] (sin/cos range)."""
        time_indices = torch.arange(100)
        encoding = fourier_encode_time(time_indices, num_frequencies=8, max_time=100)

        assert encoding.min() >= -1.0
        assert encoding.max() <= 1.0

    def test_encoding_deterministic(self):
        """Test that encoding is deterministic."""
        time_indices = torch.tensor([0, 5, 10, 50, 99])
        enc1 = fourier_encode_time(time_indices, num_frequencies=8, max_time=100)
        enc2 = fourier_encode_time(time_indices, num_frequencies=8, max_time=100)

        assert torch.allclose(enc1, enc2)


# ============== Tests for Edge Building Within Timestep ==============

class TestBuildEdgesWithinTimestep:

    def test_output_shapes(self):
        """Test that within-timestep edges have correct shapes."""
        center_points = torch.randn(3, 5, 3)  # 3 timesteps, 5 nodes

        edge_index, edge_type, rel_pos = build_edges_within_timestep(
            center_points, time_idx=1, node_offset=5, num_nodes_per_time=5, k=3
        )

        assert edge_index.shape[0] == 2
        assert edge_type.shape[1] == 5
        assert rel_pos.shape[1] == 3
        assert edge_index.shape[1] == edge_type.shape[0] == rel_pos.shape[0]

    def test_edges_are_bidirectional(self):
        """Test that all edges are bidirectional."""
        center_points = torch.randn(3, 5, 3)

        edge_index, _, _ = build_edges_within_timestep(
            center_points, time_idx=0, node_offset=0, num_nodes_per_time=5, k=3
        )

        # For each edge (a, b), there should be an edge (b, a)
        edge_set = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))

        for src, dst in list(edge_set):
            assert (dst, src) in edge_set, f"Edge ({src}, {dst}) has no reverse"

    def test_no_self_loops(self):
        """Test that there are no self-loops."""
        center_points = torch.randn(3, 5, 3)

        edge_index, _, _ = build_edges_within_timestep(
            center_points, time_idx=0, node_offset=0, num_nodes_per_time=5, k=4
        )

        # No edge should have same source and destination
        assert not (edge_index[0] == edge_index[1]).any()

    def test_edge_type_is_same_time(self):
        """Test that edge type is correctly set to same_time (index 0)."""
        center_points = torch.randn(3, 5, 3)

        _, edge_type, _ = build_edges_within_timestep(
            center_points, time_idx=0, node_offset=0, num_nodes_per_time=5, k=3
        )

        # All edges should have type 0 (same_time)
        assert (edge_type[:, 0] == 1.0).all()
        assert (edge_type[:, 1:] == 0.0).all()

    def test_node_indices_have_correct_offset(self):
        """Test that node indices are correctly offset."""
        center_points = torch.randn(3, 5, 3)
        node_offset = 10

        edge_index, _, _ = build_edges_within_timestep(
            center_points, time_idx=1, node_offset=node_offset, num_nodes_per_time=5, k=3
        )

        # All indices should be >= offset and < offset + num_nodes
        assert (edge_index >= node_offset).all()
        assert (edge_index < node_offset + 5).all()

    def test_relative_positions_correct(self):
        """Test that relative positions are computed correctly."""
        # Create simple points where we know the expected relative positions
        center_points = torch.zeros(1, 3, 3)
        center_points[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        center_points[0, 1] = torch.tensor([1.0, 0.0, 0.0])
        center_points[0, 2] = torch.tensor([0.0, 1.0, 0.0])

        edge_index, _, rel_pos = build_edges_within_timestep(
            center_points, time_idx=0, node_offset=0, num_nodes_per_time=3, k=2
        )

        # Check that relative positions match dst - src
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()
            expected_rel = center_points[0, dst_idx] - center_points[0, src_idx]
            assert torch.allclose(rel_pos[i], expected_rel), f"Edge {i}: {rel_pos[i]} != {expected_rel}"

    def test_k_larger_than_nodes(self):
        """Test handling when k is larger than available nodes."""
        center_points = torch.randn(1, 3, 3)  # Only 3 nodes

        edge_index, _, _ = build_edges_within_timestep(
            center_points, time_idx=0, node_offset=0, num_nodes_per_time=3, k=10
        )

        # Should still work, connecting all possible pairs
        assert edge_index.shape[1] > 0

    def test_single_node(self):
        """Test handling of single node (no edges possible)."""
        center_points = torch.randn(1, 1, 3)

        edge_index, edge_type, rel_pos = build_edges_within_timestep(
            center_points, time_idx=0, node_offset=0, num_nodes_per_time=1, k=4
        )

        assert edge_index.shape[1] == 0
        assert edge_type.shape[0] == 0
        assert rel_pos.shape[0] == 0


# ============== Tests for Edge Building Across Timesteps ==============

class TestBuildEdgesAcrossTimesteps:

    def test_output_shapes(self):
        """Test that cross-timestep edges have correct shapes."""
        center_points = torch.randn(5, 8, 3)

        edge_index, edge_type, rel_pos = build_edges_across_timesteps(
            center_points,
            time_idx_from=1, time_idx_to=2,
            node_offset_from=8, node_offset_to=16,
            k=3, time_diff=1
        )

        assert edge_index.shape[0] == 2
        assert edge_type.shape[1] == 5
        assert rel_pos.shape[1] == 3
        assert edge_index.shape[1] == edge_type.shape[0] == rel_pos.shape[0]

    def test_edges_are_bidirectional(self):
        """Test that cross-timestep edges are bidirectional."""
        center_points = torch.randn(3, 5, 3)

        edge_index, _, _ = build_edges_across_timesteps(
            center_points,
            time_idx_from=0, time_idx_to=1,
            node_offset_from=0, node_offset_to=5,
            k=2, time_diff=1
        )

        edge_set = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))

        for src, dst in list(edge_set):
            assert (dst, src) in edge_set, f"Edge ({src}, {dst}) has no reverse"

    def test_edge_types_future_one_step(self):
        """Test edge types for one step into the future."""
        center_points = torch.randn(3, 5, 3)

        edge_index, edge_type, _ = build_edges_across_timesteps(
            center_points,
            time_idx_from=0, time_idx_to=1,
            node_offset_from=0, node_offset_to=5,
            k=2, time_diff=1
        )

        # Forward edges (t -> t+1) should have type 1 (future_1)
        # Backward edges (t+1 -> t) should have type 3 (past_1)
        num_edges = edge_index.shape[1]

        # Check that we have both future and past types
        has_future_1 = (edge_type[:, 1] == 1.0).any()
        has_past_1 = (edge_type[:, 3] == 1.0).any()
        assert has_future_1, "Should have future_1 edges"
        assert has_past_1, "Should have past_1 edges"

    def test_edge_types_future_two_steps(self):
        """Test edge types for two steps into the future."""
        center_points = torch.randn(5, 5, 3)

        edge_index, edge_type, _ = build_edges_across_timesteps(
            center_points,
            time_idx_from=0, time_idx_to=2,
            node_offset_from=0, node_offset_to=10,
            k=2, time_diff=2
        )

        # Should have type 2 (future_2) and type 4 (past_2)
        has_future_2 = (edge_type[:, 2] == 1.0).any()
        has_past_2 = (edge_type[:, 4] == 1.0).any()
        assert has_future_2, "Should have future_2 edges"
        assert has_past_2, "Should have past_2 edges"

    def test_correct_node_offsets(self):
        """Test that node indices respect their timestep offsets."""
        center_points = torch.randn(3, 5, 3)

        edge_index, _, _ = build_edges_across_timesteps(
            center_points,
            time_idx_from=0, time_idx_to=1,
            node_offset_from=0, node_offset_to=5,
            k=2, time_diff=1
        )

        # Source nodes from timestep 0 should be in [0, 5)
        # Source nodes from timestep 1 should be in [5, 10)
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()

            # Either (src in t=0, dst in t=1) or (src in t=1, dst in t=0)
            valid = ((0 <= src < 5 and 5 <= dst < 10) or
                     (5 <= src < 10 and 0 <= dst < 5))
            assert valid, f"Invalid edge: ({src}, {dst})"

    def test_relative_positions_correct(self):
        """Test that relative positions are computed correctly across timesteps."""
        center_points = torch.zeros(2, 2, 3)
        center_points[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        center_points[0, 1] = torch.tensor([1.0, 0.0, 0.0])
        center_points[1, 0] = torch.tensor([0.5, 1.0, 0.0])
        center_points[1, 1] = torch.tensor([1.5, 1.0, 0.0])

        edge_index, _, rel_pos = build_edges_across_timesteps(
            center_points,
            time_idx_from=0, time_idx_to=1,
            node_offset_from=0, node_offset_to=2,
            k=1, time_diff=1
        )

        # Verify relative positions
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()

            src_time = 0 if src_idx < 2 else 1
            dst_time = 0 if dst_idx < 2 else 1
            src_local = src_idx if src_time == 0 else src_idx - 2
            dst_local = dst_idx if dst_time == 0 else dst_idx - 2

            expected_rel = center_points[dst_time, dst_local] - center_points[src_time, src_local]
            assert torch.allclose(rel_pos[i], expected_rel, atol=1e-5)


# ============== Tests for TokenGraphBuilder ==============

class TestTokenGraphBuilder:

    def test_output_shapes(self, simple_tokens, graph_builder):
        """Test that graph builder produces correct output shapes."""
        tokens, center_points = simple_tokens
        T, N, D = tokens.shape

        h, edge_index, edge_attr = graph_builder(tokens, center_points)

        # Node features: [T*N, D + 2*num_frequencies]
        expected_h_dim = D + 2 * graph_builder.num_frequencies
        assert h.shape == (T * N, expected_h_dim)

        # Edge index: [2, num_edges]
        assert edge_index.shape[0] == 2

        # Edge attr: [num_edges, 8]
        assert edge_attr.shape == (edge_index.shape[1], 8)

    def test_node_features_contain_tokens(self, simple_tokens, graph_builder):
        """Test that node features contain original token features."""
        tokens, center_points = simple_tokens
        T, N, D = tokens.shape

        h, _, _ = graph_builder(tokens, center_points)

        # First D dimensions should be flattened tokens
        h_tokens = h[:, :D]
        expected_tokens = tokens.reshape(T * N, D)
        assert torch.allclose(h_tokens, expected_tokens)

    def test_node_features_contain_time_encoding(self, simple_tokens, graph_builder):
        """Test that node features contain time encoding."""
        tokens, center_points = simple_tokens
        T, N, D = tokens.shape

        h, _, _ = graph_builder(tokens, center_points)

        # Last 2*num_frequencies dimensions should be time encoding
        time_dim = 2 * graph_builder.num_frequencies
        h_time = h[:, D:]

        assert h_time.shape == (T * N, time_dim)

        # Nodes from same timestep should have same time encoding
        for t in range(T):
            time_encodings = h_time[t * N:(t + 1) * N]
            # All rows should be identical for same timestep
            assert torch.allclose(time_encodings, time_encodings[0].unsqueeze(0).expand_as(time_encodings))

    def test_all_edge_types_present(self, larger_tokens, graph_builder):
        """Test that all 5 edge types are present in a larger graph."""
        tokens, center_points = larger_tokens

        h, edge_index, edge_attr = graph_builder(tokens, center_points)

        edge_types = edge_attr[:, :5]

        # Check each edge type is present
        for i, type_name in enumerate(['same_time', 'future_1', 'future_2', 'past_1', 'past_2']):
            has_type = (edge_types[:, i] == 1.0).any()
            assert has_type, f"Missing edge type: {type_name}"

    def test_edge_indices_valid(self, simple_tokens, graph_builder):
        """Test that all edge indices are valid node indices."""
        tokens, center_points = simple_tokens
        T, N, D = tokens.shape
        total_nodes = T * N

        h, edge_index, _ = graph_builder(tokens, center_points)

        assert (edge_index >= 0).all()
        assert (edge_index < total_nodes).all()

    def test_edges_are_bidirectional(self, simple_tokens, graph_builder):
        """Test that all edges in the graph are bidirectional."""
        tokens, center_points = simple_tokens

        _, edge_index, _ = graph_builder(tokens, center_points)

        edge_set = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))

        for src, dst in list(edge_set):
            assert (dst, src) in edge_set, f"Edge ({src}, {dst}) has no reverse"

    def test_no_self_loops(self, simple_tokens, graph_builder):
        """Test that there are no self-loops in the graph."""
        tokens, center_points = simple_tokens

        _, edge_index, _ = graph_builder(tokens, center_points)

        assert not (edge_index[0] == edge_index[1]).any()

    def test_edge_type_one_hot(self, simple_tokens, graph_builder):
        """Test that edge types are valid one-hot encodings."""
        tokens, center_points = simple_tokens

        _, _, edge_attr = graph_builder(tokens, center_points)

        edge_types = edge_attr[:, :5]

        # Each row should sum to 1 (one-hot)
        row_sums = edge_types.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

        # Each value should be 0 or 1
        assert ((edge_types == 0) | (edge_types == 1)).all()

    def test_spatial_edge_features(self, simple_tokens, graph_builder):
        """Test that spatial edge features (last 3 dims) are reasonable."""
        tokens, center_points = simple_tokens

        _, _, edge_attr = graph_builder(tokens, center_points)

        rel_pos = edge_attr[:, 5:]  # Last 3 dimensions

        assert rel_pos.shape[1] == 3
        # Relative positions should be finite
        assert torch.isfinite(rel_pos).all()

    def test_from_config(self):
        """Test creating builder from config."""
        config = TokenGraphConfig(
            k_same_time=3,
            k_one_step=1,
            k_two_step=1,
            num_frequencies=4,
            max_time=50,
        )

        builder = TokenGraphBuilder.from_config(config)

        assert builder.k_same_time == 3
        assert builder.k_one_step == 1
        assert builder.k_two_step == 1
        assert builder.num_frequencies == 4
        assert builder.max_time == 50

    def test_properties(self, graph_builder):
        """Test property methods."""
        assert graph_builder.time_encoding_dim == 16  # 2 * 8
        assert graph_builder.edge_feature_dim == 8    # 5 + 3

    def test_deterministic(self, simple_tokens, graph_builder):
        """Test that graph building is deterministic."""
        tokens, center_points = simple_tokens

        h1, ei1, ea1 = graph_builder(tokens, center_points)
        h2, ei2, ea2 = graph_builder(tokens, center_points)

        assert torch.allclose(h1, h2)
        assert torch.equal(ei1, ei2)
        assert torch.allclose(ea1, ea2)

    def test_gradients_flow(self, simple_tokens, graph_builder):
        """Test that gradients flow through the graph building."""
        tokens, center_points = simple_tokens
        tokens = tokens.clone().requires_grad_(True)

        h, _, _ = graph_builder(tokens, center_points)

        loss = h.sum()
        loss.backward()

        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()


# ============== Tests for Graph Connectivity ==============

class TestGraphConnectivity:

    def test_same_timestep_connectivity(self):
        """Test that nodes within same timestep are connected correctly."""
        builder = TokenGraphBuilder(
            k_same_time=2, k_one_step=0, k_two_step=0,
            num_frequencies=4, max_time=10
        )

        # Create 2 timesteps, 5 nodes each, in a line
        tokens = torch.randn(2, 5, 4)
        center_points = torch.zeros(2, 5, 3)
        for t in range(2):
            for n in range(5):
                center_points[t, n, 0] = n  # x = 0, 1, 2, 3, 4

        _, edge_index, edge_attr = builder(tokens, center_points)

        # All edges should be same_time type
        edge_types = edge_attr[:, :5]
        assert (edge_types[:, 0] == 1.0).all(), "All edges should be same_time"

        # Check connectivity within timestep 0 (nodes 0-4)
        edges_t0 = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < 5 and dst < 5:
                edges_t0.add((src, dst))

        # Node 0 should connect to nodes 1, 2 (2 nearest neighbors)
        assert (0, 1) in edges_t0 or (1, 0) in edges_t0

    def test_adjacent_timestep_connectivity(self):
        """Test that nodes across adjacent timesteps are connected."""
        builder = TokenGraphBuilder(
            k_same_time=0, k_one_step=2, k_two_step=0,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(3, 4, 4)
        center_points = torch.randn(3, 4, 3)

        _, edge_index, edge_attr = builder(tokens, center_points)

        # Should have future_1 and past_1 edges
        edge_types = edge_attr[:, :5]
        assert (edge_types[:, 1] == 1.0).any(), "Should have future_1 edges"
        assert (edge_types[:, 3] == 1.0).any(), "Should have past_1 edges"

        # Should NOT have same_time, future_2, or past_2
        assert not (edge_types[:, 0] == 1.0).any()
        assert not (edge_types[:, 2] == 1.0).any()
        assert not (edge_types[:, 4] == 1.0).any()

    def test_two_step_connectivity(self):
        """Test that nodes two timesteps apart are connected."""
        builder = TokenGraphBuilder(
            k_same_time=0, k_one_step=0, k_two_step=1,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(4, 4, 4)  # 4 timesteps
        center_points = torch.randn(4, 4, 3)

        _, edge_index, edge_attr = builder(tokens, center_points)

        # Should have future_2 and past_2 edges
        edge_types = edge_attr[:, :5]
        assert (edge_types[:, 2] == 1.0).any(), "Should have future_2 edges"
        assert (edge_types[:, 4] == 1.0).any(), "Should have past_2 edges"

    def test_first_timestep_no_past_edges(self):
        """Test that first timestep has no past edges."""
        builder = TokenGraphBuilder(
            k_same_time=1, k_one_step=2, k_two_step=1,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(5, 6, 4)
        center_points = torch.randn(5, 6, 3)
        N = 6  # nodes per timestep

        _, edge_index, edge_attr = builder(tokens, center_points)

        # Find edges where source is in first timestep
        first_timestep_nodes = set(range(N))

        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            if src in first_timestep_nodes:
                edge_type = edge_attr[i, :5]
                # Should not be past_1 or past_2
                assert edge_type[3].item() == 0, f"First timestep node {src} has past_1 edge"
                assert edge_type[4].item() == 0, f"First timestep node {src} has past_2 edge"

    def test_last_timestep_no_future_edges(self):
        """Test that last timestep has no future edges."""
        builder = TokenGraphBuilder(
            k_same_time=1, k_one_step=2, k_two_step=1,
            num_frequencies=4, max_time=10
        )

        T, N = 5, 6
        tokens = torch.randn(T, N, 4)
        center_points = torch.randn(T, N, 3)

        _, edge_index, edge_attr = builder(tokens, center_points)

        # Find edges where source is in last timestep
        last_timestep_nodes = set(range((T - 1) * N, T * N))

        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            if src in last_timestep_nodes:
                edge_type = edge_attr[i, :5]
                # Should not be future_1 or future_2
                assert edge_type[1].item() == 0, f"Last timestep node {src} has future_1 edge"
                assert edge_type[2].item() == 0, f"Last timestep node {src} has future_2 edge"

    def test_second_timestep_no_past_2_edges(self):
        """Test that second timestep (t=1) has no past_2 edges."""
        builder = TokenGraphBuilder(
            k_same_time=0, k_one_step=0, k_two_step=2,
            num_frequencies=4, max_time=10
        )

        T, N = 5, 4
        tokens = torch.randn(T, N, 4)
        center_points = torch.randn(T, N, 3)

        _, edge_index, edge_attr = builder(tokens, center_points)

        # Nodes in second timestep (indices N to 2*N-1)
        second_timestep_nodes = set(range(N, 2 * N))

        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            if src in second_timestep_nodes:
                edge_type = edge_attr[i, :5]
                # Should not be past_2
                assert edge_type[4].item() == 0, f"Second timestep node {src} has past_2 edge"


# ============== Tests for Edge Cases ==============

class TestEdgeCases:

    def test_single_timestep(self):
        """Test graph with only one timestep."""
        builder = TokenGraphBuilder(
            k_same_time=2, k_one_step=1, k_two_step=1,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(1, 5, 8)
        center_points = torch.randn(1, 5, 3)

        h, edge_index, edge_attr = builder(tokens, center_points)

        assert h.shape == (5, 8 + 8)  # 8 features + 8 time encoding

        # Only same_time edges
        if edge_index.shape[1] > 0:
            edge_types = edge_attr[:, :5]
            assert (edge_types[:, 0] == 1.0).all()

    def test_two_timesteps(self):
        """Test graph with only two timesteps."""
        builder = TokenGraphBuilder(
            k_same_time=1, k_one_step=1, k_two_step=1,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(2, 4, 8)
        center_points = torch.randn(2, 4, 3)

        h, edge_index, edge_attr = builder(tokens, center_points)

        # Should have same_time and one_step edges, but no two_step
        edge_types = edge_attr[:, :5]
        assert not (edge_types[:, 2] == 1.0).any(), "Should have no future_2 edges"
        assert not (edge_types[:, 4] == 1.0).any(), "Should have no past_2 edges"

    def test_k_zero_same_time(self):
        """Test with k_same_time=0 (no intra-timestep edges)."""
        builder = TokenGraphBuilder(
            k_same_time=0, k_one_step=2, k_two_step=0,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(3, 5, 8)
        center_points = torch.randn(3, 5, 3)

        _, _, edge_attr = builder(tokens, center_points)

        edge_types = edge_attr[:, :5]
        assert not (edge_types[:, 0] == 1.0).any(), "Should have no same_time edges"

    def test_single_node_per_timestep(self):
        """Test with only one node per timestep."""
        builder = TokenGraphBuilder(
            k_same_time=2, k_one_step=1, k_two_step=1,
            num_frequencies=4, max_time=10
        )

        tokens = torch.randn(3, 1, 8)
        center_points = torch.randn(3, 1, 3)

        h, edge_index, edge_attr = builder(tokens, center_points)

        assert h.shape == (3, 8 + 8)

        # No same_time edges possible
        if edge_index.shape[1] > 0:
            edge_types = edge_attr[:, :5]
            assert not (edge_types[:, 0] == 1.0).any()

    def test_large_graph(self):
        """Test with a large graph."""
        builder = TokenGraphBuilder(
            k_same_time=8, k_one_step=4, k_two_step=2,
            num_frequencies=16, max_time=200
        )

        T, N, D = 20, 50, 32
        tokens = torch.randn(T, N, D)
        center_points = torch.randn(T, N, 3)

        h, edge_index, edge_attr = builder(tokens, center_points)

        assert h.shape == (T * N, D + 32)  # 32 = 2 * 16
        assert edge_index.shape[0] == 2
        assert edge_attr.shape[1] == 8


# ============== Integration with BatchedMPN ==============

class TestIntegrationWithBatchedMPN:

    def test_output_compatible_with_collate(self):
        """Test that graph output can be used with collate_graphs."""
        from pc_mango.network.util.gnn_encoder.mpn import collate_graphs, BatchedMPN

        builder = TokenGraphBuilder(
            k_same_time=2, k_one_step=1, k_two_step=1,
            num_frequencies=8, max_time=50
        )

        # Create multiple token streams
        graphs = []
        for _ in range(3):
            T = torch.randint(3, 6, (1,)).item()
            N = torch.randint(4, 8, (1,)).item()

            tokens = torch.randn(T, N, 16)
            center_points = torch.randn(T, N, 3)

            h, edge_index, edge_attr = builder(tokens, center_points)
            graphs.append((h, edge_index, edge_attr))

        # Should be able to collate
        batch = collate_graphs(graphs)

        assert batch.num_graphs == 3
        assert batch.h.shape[1] == 16 + 16  # token_dim + time_encoding
        assert batch.edge_attr.shape[1] == 8

    def test_full_pipeline(self):
        """Test full pipeline: tokens -> graph -> batch -> MPN -> readout."""
        from pc_mango.network.util.gnn_encoder.mpn import (
            collate_graphs, BatchedMPN, MeanReadout
        )

        builder = TokenGraphBuilder(
            k_same_time=2, k_one_step=1, k_two_step=1,
            num_frequencies=8, max_time=50
        )

        # Create token streams
        graphs = []
        for _ in range(4):
            tokens = torch.randn(4, 6, 16)
            center_points = torch.randn(4, 6, 3)

            h, edge_index, edge_attr = builder(tokens, center_points)
            graphs.append((h, edge_index, edge_attr))

        # Batch graphs
        batch = collate_graphs(graphs)

        # Process through MPN
        node_dim = 16 + 16  # token_dim + time_encoding
        edge_dim = 8
        latent_dim = 32

        mpn = BatchedMPN(
            n_layers=2,
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
        )

        h_out, batch_idx = mpn(batch)

        # Readout
        readout = MeanReadout(latent_dim=latent_dim)
        graph_embeddings = readout(h_out, batch_idx)

        assert graph_embeddings.shape == (4, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
