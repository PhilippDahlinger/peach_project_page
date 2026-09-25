import torch
import torch.nn as nn
from pc_mango.network.util.gnn_encoder.mpn import (
    drop_edge, BatchedMPN, BatchedGraph,
    AttentionReadout, GatedPoolingReadout, MeanReadout, MaxReadout
)


def test_drop_edge_basic():
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    # p = 0 -> unchanged
    ei0, ea0 = drop_edge(edge_index, edge_attr, p=0.0)
    assert torch.equal(ei0, edge_index)
    assert torch.equal(ea0, edge_attr)

    # p = 1 -> all dropped
    ei1, ea1 = drop_edge(edge_index, edge_attr, p=1.0)
    assert ei1.shape == (2, 0)
    assert ea1.shape[0] == 0

    # intermediate p -> deterministic with seed, ensure returned shapes align
    torch.manual_seed(123)
    eim, eam = drop_edge(edge_index, edge_attr, p=0.5)
    assert 0 <= eim.shape[1] <= edge_index.shape[1]
    assert eam.shape[0] == eim.shape[1]


def test_batched_mpn_dropedge_training_and_eval():
    torch.manual_seed(0)
    node_dim = 3
    edge_dim = 2

    # One graph with 4 nodes and 2 edges
    h = torch.randn(4, node_dim)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.randn(2, edge_dim)
    batch_idx = torch.zeros(4, dtype=torch.long)
    batch = BatchedGraph(h=h, edge_index=edge_index, edge_attr=edge_attr, batch_idx=batch_idx, num_graphs=1)

    model = BatchedMPN(n_layers=1, node_dim=node_dim, edge_dim=edge_dim, latent_dim=6, drop_edge_p=1.0)

    # Training: edges dropped -> early-return path provides an empty message tensor as second return
    model.train()
    out_h, second = model(batch)
    # Second return may be an empty message tensor (shape [0, latent_dim])
    assert isinstance(out_h, torch.Tensor)
    assert (isinstance(second, torch.Tensor) and second.shape[0] == 0) or (isinstance(second, torch.Tensor) and second.shape[0] == 4)

    # Eval: edges kept -> second return should be batch.batch_idx (length == num nodes)
    model.eval()
    out_h2, second2 = model(batch)
    assert out_h2.shape[0] == 4
    assert second2.shape[0] == 4


def test_dropout_modules_present():
    node_dim = 3
    edge_dim = 2

    model = BatchedMPN(n_layers=1, node_dim=node_dim, edge_dim=edge_dim, latent_dim=5, dropout_embedding=0.2, dropout_node=0.3, dropout_edge=0.4)

    # embedding should be wrapped in Sequential when dropout_embedding > 0
    assert isinstance(model.node_embedding, nn.Sequential)

    # node and edge dropout modules should be Dropout instances
    assert isinstance(model.node_dropout, nn.Dropout)
    assert isinstance(model.edge_dropout, nn.Dropout)

    # Also check values are set
    assert abs(model.dropout_embedding - 0.2) < 1e-6


def test_attention_readout_dropout():
    torch.manual_seed(42)
    latent_dim = 8
    num_nodes = 6

    h = torch.randn(num_nodes, latent_dim)
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # Without dropout
    readout_no_drop = AttentionReadout(latent_dim=latent_dim, dropout=0.0)
    readout_no_drop.eval()
    out_no_drop = readout_no_drop(h, batch_idx)
    assert out_no_drop.shape == (2, latent_dim)

    # With dropout
    readout_with_drop = AttentionReadout(latent_dim=latent_dim, dropout=0.5)
    assert isinstance(readout_with_drop.dropout, nn.Dropout)

    # In eval mode, dropout should not change output
    readout_with_drop.eval()
    out_eval = readout_with_drop(h, batch_idx)
    assert out_eval.shape == (2, latent_dim)


def test_gated_pooling_readout_dropout():
    torch.manual_seed(42)
    latent_dim = 8
    num_nodes = 6

    h = torch.randn(num_nodes, latent_dim)
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # Without dropout
    readout_no_drop = GatedPoolingReadout(latent_dim=latent_dim, dropout=0.0)
    readout_no_drop.eval()
    out_no_drop = readout_no_drop(h, batch_idx)
    assert out_no_drop.shape == (2, latent_dim)

    # With dropout
    readout_with_drop = GatedPoolingReadout(latent_dim=latent_dim, dropout=0.5)
    assert isinstance(readout_with_drop.dropout, nn.Dropout)

    # In eval mode, dropout should not change output
    readout_with_drop.eval()
    out_eval = readout_with_drop(h, batch_idx)
    assert out_eval.shape == (2, latent_dim)


def test_mean_and_max_readout():
    torch.manual_seed(42)
    latent_dim = 8
    num_nodes = 6

    h = torch.randn(num_nodes, latent_dim)
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # Mean readout
    mean_readout = MeanReadout(latent_dim=latent_dim)
    out_mean = mean_readout(h, batch_idx)
    assert out_mean.shape == (2, latent_dim)

    # Max readout
    max_readout = MaxReadout(latent_dim=latent_dim)
    out_max = max_readout(h, batch_idx)
    assert out_max.shape == (2, latent_dim)
