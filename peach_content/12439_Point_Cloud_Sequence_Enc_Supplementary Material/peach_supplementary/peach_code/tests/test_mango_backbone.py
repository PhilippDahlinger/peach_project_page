import torch
import torch.nn as nn
import pytest
from pc_mango.network.util.mango.mango_backbone import MangoBackbone, MangoBackboneV2


class TestMangoBackboneV2:
    """Tests for MangoBackboneV2 implementation."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors for testing."""
        batch_dim = 2
        num_ts = 10
        num_nodes = 8
        world_dim = 3
        h_dim = 5
        edge_feature_dim = 4
        num_edges = 12

        x = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        v = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        h = torch.randn(batch_dim, num_nodes, h_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_fea = torch.randn(batch_dim, num_edges, edge_feature_dim)

        return {
            'x': x,
            'v': v,
            'h': h,
            'edge_index': edge_index,
            'edge_fea': edge_fea,
            'h_dim': h_dim,
            'edge_feature_dim': edge_feature_dim,
            'world_dim': world_dim,
        }

    def test_v2_instantiation(self, sample_input):
        """Test that MangoBackboneV2 can be instantiated."""
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            activation="leakyrelu",
            use_time_conv=True,
            time_conv_type="conv_decoder",
        )
        assert model is not None
        assert model.n_layers == 2
        assert model.latent_dim == 16

    def test_v2_forward_pass(self, sample_input):
        """Test that forward pass produces correct output shape."""
        latent_dim = 16
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=latent_dim,
            activation="leakyrelu",
            use_time_conv=True,
            time_conv_type="conv_decoder",
        )

        output = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        # Output shape should be [batch_dim, num_ts, num_nodes, latent_dim]
        batch_dim, num_ts, num_nodes, _ = sample_input['x'].shape
        assert output.shape == (batch_dim, num_ts, num_nodes, latent_dim)

    def test_v2_without_time_conv(self, sample_input):
        """Test MangoBackboneV2 without time convolution."""
        latent_dim = 16
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=latent_dim,
            activation="relu",
            use_time_conv=False,
        )

        output = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        batch_dim, num_ts, num_nodes, _ = sample_input['x'].shape
        assert output.shape == (batch_dim, num_ts, num_nodes, latent_dim)

    def test_v2_dropout_modules_present(self, sample_input):
        """Test that dropout modules are correctly initialized."""
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            dropout=0.1,
            dropout_embedding=0.2,
            dropout_node=0.3,
            dropout_edge=0.4,
        )

        # Check embedding dropout
        assert model.dropout_embedding == 0.2
        assert isinstance(model.node_embedding, nn.Sequential)
        assert isinstance(model.node_embedding[1], nn.Dropout)

        # Check node and edge dropouts
        assert isinstance(model.node_dropout, nn.Dropout)
        assert model.node_dropout.p == 0.3
        assert isinstance(model.edge_dropout, nn.Dropout)
        assert model.edge_dropout.p == 0.4

    def test_v2_dropout_fallback_to_default(self, sample_input):
        """Test that specific dropouts fall back to default dropout value."""
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            dropout=0.25,
            # Not specifying specific dropouts - should use dropout value
        )

        assert model.dropout_embedding == 0.25
        assert isinstance(model.node_dropout, nn.Dropout)
        assert model.node_dropout.p == 0.25
        assert isinstance(model.edge_dropout, nn.Dropout)
        assert model.edge_dropout.p == 0.25

    def test_v2_no_dropout_when_zero(self, sample_input):
        """Test that Identity is used when dropout is 0."""
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            dropout=0.0,
        )

        assert isinstance(model.node_dropout, nn.Identity)
        assert isinstance(model.edge_dropout, nn.Identity)
        # node_embedding should be a simple Linear, not Sequential
        assert isinstance(model.node_embedding, nn.Linear)

    def test_v2_drop_edge_training_vs_eval(self, sample_input):
        """Test that edge dropping only occurs during training."""
        torch.manual_seed(42)
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            drop_edge_p=0.5,
            use_time_conv=False,
        )

        # Run in training mode multiple times - results should vary due to edge dropping
        model.train()
        torch.manual_seed(0)
        out_train_1 = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )
        torch.manual_seed(1)
        out_train_2 = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        # Run in eval mode - edges should not be dropped
        model.eval()
        torch.manual_seed(0)
        out_eval_1 = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )
        torch.manual_seed(1)
        out_eval_2 = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        # In eval mode, outputs should be identical regardless of seed
        assert torch.allclose(out_eval_1, out_eval_2)
        # Training outputs may differ (unless all edges kept by chance)
        # We just check the shapes are correct
        assert out_train_1.shape == out_train_2.shape

    def test_v2_drop_edge_all_edges_dropped(self, sample_input):
        """Test handling when all edges are dropped (drop_edge_p=1.0)."""
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            drop_edge_p=1.0,
            use_time_conv=False,
        )
        model.train()

        # Should not raise an error even when all edges are dropped
        output = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        batch_dim, num_ts, num_nodes, _ = sample_input['x'].shape
        assert output.shape == (batch_dim, num_ts, num_nodes, 16)

    def test_v2_spectral_time_conv(self, sample_input):
        """Test MangoBackboneV2 with spectral time convolution."""
        model = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            use_time_conv=True,
            time_conv_type="spectral",
        )

        output = model(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        batch_dim, num_ts, num_nodes, _ = sample_input['x'].shape
        assert output.shape == (batch_dim, num_ts, num_nodes, 16)

    def test_v2_h_with_time_dimension(self, sample_input):
        """Test that h input with time dimension is handled correctly."""
        batch_dim, num_ts, num_nodes, _ = sample_input['x'].shape
        h_dim = sample_input['h_dim']

        # Create h with time dimension already included
        h_with_time = torch.randn(batch_dim, num_ts, num_nodes, h_dim)

        model = MangoBackboneV2(
            n_layers=2,
            h_dim=h_dim,
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=16,
            use_time_conv=False,
        )

        output = model(
            sample_input['x'],
            h_with_time,
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        assert output.shape == (batch_dim, num_ts, num_nodes, 16)


class TestMangoBackboneV1VsV2Comparison:
    """Compare V1 and V2 outputs under same conditions."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors for testing."""
        torch.manual_seed(123)
        batch_dim = 2
        num_ts = 8
        num_nodes = 6
        world_dim = 3
        h_dim = 4
        edge_feature_dim = 3
        num_edges = 10

        x = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        v = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        h = torch.randn(batch_dim, num_nodes, h_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_fea = torch.randn(batch_dim, num_edges, edge_feature_dim)

        return {
            'x': x,
            'v': v,
            'h': h,
            'edge_index': edge_index,
            'edge_fea': edge_fea,
            'h_dim': h_dim,
            'edge_feature_dim': edge_feature_dim,
            'world_dim': world_dim,
        }

    def test_v1_and_v2_have_same_output_shape(self, sample_input):
        """Test that V1 and V2 produce same output shape."""
        latent_dim = 16

        model_v1 = MangoBackbone(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=latent_dim,
            use_time_conv=False,
        )

        model_v2 = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=latent_dim,
            use_time_conv=False,
            dropout=0.0,
            drop_edge_p=0.0,
        )

        out_v1 = model_v1(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        out_v2 = model_v2(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        assert out_v1.shape == out_v2.shape

    def test_v1_and_v2_different_implementations(self, sample_input):
        """Test that V1 and V2 produce different outputs (due to Pre-LN vs Post-LN)."""
        torch.manual_seed(42)
        latent_dim = 16

        model_v1 = MangoBackbone(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=latent_dim,
            use_time_conv=False,
        )

        # Initialize V2 with same weights to compare structure
        torch.manual_seed(42)
        model_v2 = MangoBackboneV2(
            n_layers=2,
            h_dim=sample_input['h_dim'],
            edge_feature_dim=sample_input['edge_feature_dim'],
            world_dim=sample_input['world_dim'],
            latent_dim=latent_dim,
            use_time_conv=False,
            dropout=0.0,
            drop_edge_p=0.0,
        )

        model_v1.eval()
        model_v2.eval()

        out_v1 = model_v1(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        out_v2 = model_v2(
            sample_input['x'],
            sample_input['h'],
            sample_input['v'],
            sample_input['edge_index'],
            sample_input['edge_fea'],
        )

        # They should produce different results because Pre-LN vs Post-LN
        # (unless weights are zero which is extremely unlikely)
        assert not torch.allclose(out_v1, out_v2, atol=1e-5)


class TestMangoBackboneV2EdgeCases:
    """Test edge cases for MangoBackboneV2."""

    def test_single_node_graph(self):
        """Test with a graph that has only one node."""
        batch_dim = 1
        num_ts = 5
        num_nodes = 1
        world_dim = 3
        h_dim = 4
        edge_feature_dim = 2

        x = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        v = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        h = torch.randn(batch_dim, num_nodes, h_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        edge_fea = torch.randn(batch_dim, 0, edge_feature_dim)

        model = MangoBackboneV2(
            n_layers=2,
            h_dim=h_dim,
            edge_feature_dim=edge_feature_dim,
            world_dim=world_dim,
            latent_dim=8,
            use_time_conv=False,
        )

        output = model(x, h, v, edge_index, edge_fea)
        assert output.shape == (batch_dim, num_ts, num_nodes, 8)

    def test_different_activations(self):
        """Test all supported activation functions."""
        h_dim = 4
        edge_feature_dim = 2
        world_dim = 3

        for activation in ["leakyrelu", "relu", "silu"]:
            model = MangoBackboneV2(
                n_layers=1,
                h_dim=h_dim,
                edge_feature_dim=edge_feature_dim,
                world_dim=world_dim,
                latent_dim=8,
                activation=activation,
                use_time_conv=False,
            )
            assert model is not None

    def test_invalid_activation_raises_error(self):
        """Test that invalid activation raises an error."""
        with pytest.raises(ValueError, match="Unknown activation function"):
            MangoBackboneV2(
                n_layers=1,
                h_dim=4,
                edge_feature_dim=2,
                world_dim=3,
                latent_dim=8,
                activation="invalid_activation",
                use_time_conv=False,
            )

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        batch_dim = 2
        num_ts = 4
        num_nodes = 5
        world_dim = 3
        h_dim = 4
        edge_feature_dim = 2
        latent_dim = 8
        num_edges = 8

        x = torch.randn(batch_dim, num_ts, num_nodes, world_dim, requires_grad=True)
        v = torch.randn(batch_dim, num_ts, num_nodes, world_dim)
        h = torch.randn(batch_dim, num_nodes, h_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_fea = torch.randn(batch_dim, num_edges, edge_feature_dim)

        model = MangoBackboneV2(
            n_layers=2,
            h_dim=h_dim,
            edge_feature_dim=edge_feature_dim,
            world_dim=world_dim,
            latent_dim=latent_dim,
            use_time_conv=False,
        )

        output = model(x, h, v, edge_index, edge_fea)
        loss = output.sum()
        loss.backward()

        # Check that gradients are not None and not all zeros
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
