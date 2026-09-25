"""
Test cases for BatchedMPN, collate_graphs, and GraphReadout classes.
"""
import pytest
import torch
import torch.nn as nn

from pc_mango.network.util.gnn_encoder.mpn import (
    BatchedGraph,
    BatchedMPN,
    collate_graphs,
    MeanReadout,
    MaxReadout,
    AttentionReadout,
    GatedPoolingReadout,
    GraphReadout,
)


# ============== Fixtures ==============

@pytest.fixture
def simple_graph():
    """Create a simple graph with 4 nodes and 4 edges (a square)."""
    h = torch.randn(4, 8)  # 4 nodes, 8-dim features
    edge_index = torch.tensor([
        [0, 1, 2, 3],  # source
        [1, 2, 3, 0],  # target
    ])
    edge_attr = torch.randn(4, 4)  # 4 edges, 4-dim features
    return h, edge_index, edge_attr


@pytest.fixture
def graph_list():
    """Create a list of 3 graphs with varying sizes."""
    graphs = []
    
    # Graph 0: 3 nodes, 2 edges
    h0 = torch.randn(3, 8)
    edge_index0 = torch.tensor([[0, 1], [1, 2]])
    edge_attr0 = torch.randn(2, 4)
    graphs.append((h0, edge_index0, edge_attr0))
    
    # Graph 1: 5 nodes, 4 edges
    h1 = torch.randn(5, 8)
    edge_index1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr1 = torch.randn(4, 4)
    graphs.append((h1, edge_index1, edge_attr1))
    
    # Graph 2: 2 nodes, 1 edge
    h2 = torch.randn(2, 8)
    edge_index2 = torch.tensor([[0], [1]])
    edge_attr2 = torch.randn(1, 4)
    graphs.append((h2, edge_index2, edge_attr2))
    
    return graphs


@pytest.fixture
def batched_graph(graph_list):
    """Create a BatchedGraph from graph_list."""
    return collate_graphs(graph_list)


# ============== Tests for collate_graphs ==============

class TestCollateGraphs:
    
    def test_basic_collation(self, graph_list):
        """Test that collate_graphs produces correct output shapes."""
        batch = collate_graphs(graph_list)
        
        # Total nodes: 3 + 5 + 2 = 10
        assert batch.h.shape == (10, 8)
        # Total edges: 2 + 4 + 1 = 7
        assert batch.edge_index.shape == (2, 7)
        assert batch.edge_attr.shape == (7, 4)
        assert batch.batch_idx.shape == (10,)
        assert batch.num_graphs == 3
    
    def test_batch_idx_assignment(self, graph_list):
        """Test that batch_idx correctly assigns nodes to graphs."""
        batch = collate_graphs(graph_list)
        
        # First 3 nodes belong to graph 0
        assert (batch.batch_idx[:3] == 0).all()
        # Next 5 nodes belong to graph 1
        assert (batch.batch_idx[3:8] == 1).all()
        # Last 2 nodes belong to graph 2
        assert (batch.batch_idx[8:] == 2).all()
    
    def test_edge_index_offset(self, graph_list):
        """Test that edge indices are correctly offset."""
        batch = collate_graphs(graph_list)
        
        # Graph 0 edges: original [0,1]->[1,2], still [0,1]->[1,2]
        assert batch.edge_index[0, 0].item() == 0
        assert batch.edge_index[1, 0].item() == 1
        
        # Graph 1 edges: original starts at 0, should start at 3 (offset)
        assert batch.edge_index[0, 2].item() == 3  # First edge of graph 1
        assert batch.edge_index[1, 2].item() == 4
        
        # Graph 2 edges: original [0]->[1], should be [8]->[9]
        assert batch.edge_index[0, 6].item() == 8
        assert batch.edge_index[1, 6].item() == 9
    
    def test_single_graph(self, simple_graph):
        """Test collating a single graph."""
        batch = collate_graphs([simple_graph])
        
        h, edge_index, edge_attr = simple_graph
        assert torch.allclose(batch.h, h)
        assert torch.equal(batch.edge_index, edge_index)
        assert torch.allclose(batch.edge_attr, edge_attr)
        assert (batch.batch_idx == 0).all()
        assert batch.num_graphs == 1
    
    def test_preserves_node_features(self, graph_list):
        """Test that node features are preserved in order."""
        batch = collate_graphs(graph_list)
        
        h0, _, _ = graph_list[0]
        h1, _, _ = graph_list[1]
        h2, _, _ = graph_list[2]
        
        assert torch.allclose(batch.h[:3], h0)
        assert torch.allclose(batch.h[3:8], h1)
        assert torch.allclose(batch.h[8:], h2)


# ============== Tests for BatchedMPN ==============

class TestBatchedMPN:
    
    def test_forward_shape(self, batched_graph):
        """Test that BatchedMPN produces correct output shapes."""
        model = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        h_out, batch_idx = model(batched_graph)
        
        assert h_out.shape == (10, 16)  # 10 nodes, 16 latent dim
        assert batch_idx.shape == (10,)
        assert torch.equal(batch_idx, batched_graph.batch_idx)
    
    def test_forward_different_configs(self, batched_graph):
        """Test BatchedMPN with different configurations."""
        configs = [
            {"n_layers": 1, "activation": "relu"},
            {"n_layers": 3, "activation": "silu", "use_hidden_layers": True},
            {"n_layers": 2, "aggr": "sum"},
            {"n_layers": 2, "dropout": 0.1},
        ]
        
        for config in configs:
            model = BatchedMPN(
                node_dim=8,
                edge_dim=4,
                latent_dim=16,
                **config,
            )
            model.eval()  # Disable dropout for consistent testing
            
            h_out, batch_idx = model(batched_graph)
            assert h_out.shape == (10, 16)
    
    def test_gradients_flow(self, batched_graph):
        """Test that gradients flow through the network."""
        model = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        h_out, _ = model(batched_graph)
        loss = h_out.sum()
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_deterministic_output(self, batched_graph):
        """Test that same input produces same output (in eval mode)."""
        model = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        model.eval()
        
        with torch.no_grad():
            h_out1, _ = model(batched_graph)
            h_out2, _ = model(batched_graph)
        
        assert torch.allclose(h_out1, h_out2)


# ============== Tests for Readout Classes ==============

class TestMeanReadout:
    
    def test_output_shape(self):
        """Test that MeanReadout produces correct output shape."""
        readout = MeanReadout(latent_dim=16)
        
        h = torch.randn(10, 16)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        assert out.shape == (3, 16)  # 3 graphs
    
    def test_mean_computation(self):
        """Test that MeanReadout correctly computes the mean."""
        readout = MeanReadout(latent_dim=2)
        
        # Simple test case: 2 graphs, 2 and 3 nodes each
        h = torch.tensor([
            [1.0, 2.0],  # Graph 0, node 0
            [3.0, 4.0],  # Graph 0, node 1
            [5.0, 6.0],  # Graph 1, node 0
            [7.0, 8.0],  # Graph 1, node 1
            [9.0, 10.0], # Graph 1, node 2
        ])
        batch_idx = torch.tensor([0, 0, 1, 1, 1])
        
        out = readout(h, batch_idx)
        
        # Graph 0 mean: (1+3)/2=2, (2+4)/2=3
        # Graph 1 mean: (5+7+9)/3=7, (6+8+10)/3=8
        expected = torch.tensor([[2.0, 3.0], [7.0, 8.0]])
        assert torch.allclose(out, expected)
    
    def test_single_node_graphs(self):
        """Test with single-node graphs."""
        readout = MeanReadout(latent_dim=4)
        
        h = torch.randn(3, 4)
        batch_idx = torch.tensor([0, 1, 2])
        
        out = readout(h, batch_idx)
        assert torch.allclose(out, h)  # Mean of single node is the node itself


class TestMaxReadout:
    
    def test_output_shape(self):
        """Test that MaxReadout produces correct output shape."""
        readout = MaxReadout(latent_dim=16)
        
        h = torch.randn(10, 16)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        assert out.shape == (3, 16)
    
    def test_max_computation(self):
        """Test that MaxReadout correctly computes the max per feature."""
        readout = MaxReadout(latent_dim=2)
        
        h = torch.tensor([
            [1.0, 4.0],  # Graph 0, node 0
            [3.0, 2.0],  # Graph 0, node 1
            [5.0, 6.0],  # Graph 1, node 0
            [7.0, 4.0],  # Graph 1, node 1
        ])
        batch_idx = torch.tensor([0, 0, 1, 1])
        
        out = readout(h, batch_idx)
        
        # Graph 0 max: max(1,3)=3, max(4,2)=4
        # Graph 1 max: max(5,7)=7, max(6,4)=6
        expected = torch.tensor([[3.0, 4.0], [7.0, 6.0]])
        assert torch.allclose(out, expected)
    
    def test_single_node_graphs(self):
        """Test with single-node graphs."""
        readout = MaxReadout(latent_dim=4)
        
        h = torch.randn(3, 4)
        batch_idx = torch.tensor([0, 1, 2])
        
        out = readout(h, batch_idx)
        assert torch.allclose(out, h)


class TestAttentionReadout:
    
    def test_output_shape(self):
        """Test that AttentionReadout produces correct output shape."""
        readout = AttentionReadout(latent_dim=16)
        
        h = torch.randn(10, 16)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        assert out.shape == (3, 16)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 within each graph."""
        readout = AttentionReadout(latent_dim=8)
        
        h = torch.randn(10, 8)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        # Manually compute attention weights
        scores = readout.score_fn(h).squeeze(-1)
        num_graphs = 3
        
        max_scores = torch.full((num_graphs,), float('-inf'))
        max_scores.scatter_reduce_(0, batch_idx, scores, reduce='amax', include_self=False)
        
        scores_normalized = scores - max_scores[batch_idx]
        exp_scores = torch.exp(scores_normalized)
        
        sum_exp = torch.zeros(num_graphs)
        sum_exp.scatter_add_(0, batch_idx, exp_scores)
        
        alpha = exp_scores / sum_exp[batch_idx]
        
        # Check that alphas sum to 1 per graph
        alpha_sum = torch.zeros(num_graphs)
        alpha_sum.scatter_add_(0, batch_idx, alpha)
        
        assert torch.allclose(alpha_sum, torch.ones(num_graphs), atol=1e-5)
    
    def test_gradients_flow(self):
        """Test that gradients flow through attention."""
        readout = AttentionReadout(latent_dim=16)
        
        h = torch.randn(10, 16, requires_grad=True)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        loss = out.sum()
        loss.backward()
        
        assert h.grad is not None
        assert not torch.isnan(h.grad).any()
    
    def test_with_custom_hidden_dim(self):
        """Test AttentionReadout with custom hidden dimension."""
        readout = AttentionReadout(latent_dim=16, hidden_dim=32)
        
        h = torch.randn(10, 16)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        assert out.shape == (3, 16)


class TestGatedPoolingReadout:
    
    def test_output_shape(self):
        """Test that GatedPoolingReadout produces correct output shape."""
        readout = GatedPoolingReadout(latent_dim=16)
        
        h = torch.randn(10, 16)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        assert out.shape == (3, 16)
    
    def test_gating_mechanism(self):
        """Test that gating mechanism is applied correctly."""
        readout = GatedPoolingReadout(latent_dim=4)
        
        # Set weights to identity for easier testing
        with torch.no_grad():
            readout.proj.weight.copy_(torch.eye(4))
            readout.proj.bias.zero_()
            readout.gate.weight.copy_(torch.zeros(4, 4))
            readout.gate.bias.copy_(torch.zeros(4))  # sigmoid(0) = 0.5
        
        h = torch.ones(4, 4)
        batch_idx = torch.tensor([0, 0, 1, 1])
        
        out = readout(h, batch_idx)
        
        # h_tilde = h (identity projection)
        # g = sigmoid(0) = 0.5
        # m = 0.5 * 1 = 0.5 per node
        # sum of 2 nodes per graph = 1.0
        expected = torch.ones(2, 4)
        assert torch.allclose(out, expected)
    
    def test_gradients_flow(self):
        """Test that gradients flow through gating."""
        readout = GatedPoolingReadout(latent_dim=16)
        
        h = torch.randn(10, 16, requires_grad=True)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        
        out = readout(h, batch_idx)
        loss = out.sum()
        loss.backward()
        
        assert h.grad is not None
        assert not torch.isnan(h.grad).any()
        
        # Check that model parameters have gradients
        assert readout.proj.weight.grad is not None
        assert readout.gate.weight.grad is not None


class TestReadoutAbstractClass:
    
    def test_is_abstract(self):
        """Test that GraphReadout cannot be instantiated."""
        with pytest.raises(TypeError):
            GraphReadout(latent_dim=16)
    
    def test_all_readouts_inherit_from_base(self):
        """Test that all readout classes inherit from GraphReadout."""
        readouts = [MeanReadout, MaxReadout, AttentionReadout, GatedPoolingReadout]
        
        for cls in readouts:
            assert issubclass(cls, GraphReadout)
            assert issubclass(cls, nn.Module)


# ============== Integration Tests ==============

class TestIntegration:
    
    def test_full_pipeline(self, graph_list):
        """Test full pipeline: collate -> MPN -> readout."""
        batch = collate_graphs(graph_list)
        
        mpn = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        readout = MeanReadout(latent_dim=16)
        
        h_out, batch_idx = mpn(batch)
        graph_embeddings = readout(h_out, batch_idx)
        
        assert graph_embeddings.shape == (3, 16)
    
    def test_all_readouts_with_mpn(self, graph_list):
        """Test all readout types with MPN output."""
        batch = collate_graphs(graph_list)
        
        mpn = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        readouts = [
            MeanReadout(16),
            MaxReadout(16),
            AttentionReadout(16),
            GatedPoolingReadout(16),
        ]
        
        h_out, batch_idx = mpn(batch)
        
        for readout in readouts:
            graph_embeddings = readout(h_out, batch_idx)
            assert graph_embeddings.shape == (3, 16)
    
    def test_backward_through_full_pipeline(self, graph_list):
        """Test that gradients flow through the full pipeline."""
        batch = collate_graphs(graph_list)
        
        mpn = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        readout = AttentionReadout(latent_dim=16)
        
        h_out, batch_idx = mpn(batch)
        graph_embeddings = readout(h_out, batch_idx)
        
        loss = graph_embeddings.sum()
        loss.backward()
        
        # Check MPN gradients
        for param in mpn.parameters():
            assert param.grad is not None
        
        # Check readout gradients
        for param in readout.parameters():
            assert param.grad is not None
    
    def test_batch_independence(self, graph_list):
        """Test that processing as batch gives same result as individual graphs."""
        torch.manual_seed(42)
        
        mpn_batched = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        mpn_batched.eval()
        
        # Process as batch
        batch = collate_graphs(graph_list)
        with torch.no_grad():
            h_batch, batch_idx = mpn_batched(batch)
        
        # Process individually
        individual_results = []
        for h, edge_index, edge_attr in graph_list:
            single_batch = collate_graphs([(h, edge_index, edge_attr)])
            with torch.no_grad():
                h_single, _ = mpn_batched(single_batch)
            individual_results.append(h_single)
        
        # Compare results
        h_individual = torch.cat(individual_results, dim=0)
        assert torch.allclose(h_batch, h_individual, atol=1e-5)


# ============== Edge Cases ==============

class TestEdgeCases:
    
    def test_empty_batch(self):
        """Test handling of empty graph list."""
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            collate_graphs([])
    
    def test_large_batch(self):
        """Test with a larger number of graphs."""
        graphs = []
        for i in range(100):
            n_nodes = torch.randint(2, 10, (1,)).item()
            n_edges = torch.randint(1, n_nodes * 2, (1,)).item()
            
            h = torch.randn(n_nodes, 8)
            edge_index = torch.randint(0, n_nodes, (2, n_edges))
            edge_attr = torch.randn(n_edges, 4)
            
            graphs.append((h, edge_index, edge_attr))
        
        batch = collate_graphs(graphs)
        
        mpn = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        h_out, batch_idx = mpn(batch)
        
        assert batch.num_graphs == 100
        assert h_out.shape[1] == 16
    
    def test_single_node_graph(self):
        """Test with a graph containing a single node and no edges."""
        h = torch.randn(1, 8)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 4)
        
        batch = collate_graphs([(h, edge_index, edge_attr)])
        
        mpn = BatchedMPN(
            n_layers=2,
            node_dim=8,
            edge_dim=4,
            latent_dim=16,
        )
        
        h_out, batch_idx = mpn(batch)
        
        assert h_out.shape == (1, 16)
        assert batch_idx.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
