"""
Test cases for meta-aggregation modules.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from pc_mango.network.meta_aggregation import get_meta_aggregation
from pc_mango.network.meta_aggregation.base import BaseMetaAggregation
from pc_mango.network.meta_aggregation.max_aggregation import MaxAggregation
from pc_mango.network.meta_aggregation.mean_aggregation import MeanAggregation
from pc_mango.network.meta_aggregation.softmax_pool import SoftmaxPoolAggregation
from pc_mango.network.meta_aggregation.attention_aggregation import AttentionAggregation
from pc_mango.network.meta_aggregation.gated_aggregation import GatedAggregation
from pc_mango.network.meta_aggregation.set_transformer_aggregation import SetTransformerAggregation


# ============== Fixtures ==============

@pytest.fixture
def max_config():
    """Config for max aggregation."""
    return SimpleNamespace(name="max")


@pytest.fixture
def mean_config():
    """Config for mean aggregation."""
    return SimpleNamespace(name="mean")


@pytest.fixture
def softmax_config():
    """Config for softmax pool aggregation with fixed beta."""
    return SimpleNamespace(
        name="softmax_pool",
        beta_init=1.0,
        beta_trainable=False
    )


@pytest.fixture
def softmax_trainable_config():
    """Config for softmax pool aggregation with trainable beta."""
    return SimpleNamespace(
        name="softmax_pool",
        beta_init=2.0,
        beta_trainable=True
    )


@pytest.fixture
def attention_config():
    """Config for attention aggregation."""
    return SimpleNamespace(
        name="attention",
        latent_dim=16,
        hidden_dim=16
    )


@pytest.fixture
def gated_config():
    """Config for gated aggregation."""
    return SimpleNamespace(
        name="gated",
        latent_dim=16
    )


@pytest.fixture
def set_transformer_config():
    """Config for set transformer aggregation."""
    return SimpleNamespace(
        name="set_transformer",
        latent_dim=16,
        num_heads=4,
        num_isab_layers=2,
        dropout=0.0
    )


@pytest.fixture
def sample_latent_vectors():
    """Sample latent vectors for testing: (context_size=5, latent_dim=16)."""
    torch.manual_seed(42)
    return torch.randn(5, 16)


@pytest.fixture
def single_latent_vector():
    """Single latent vector for edge case testing: (context_size=1, latent_dim=16)."""
    torch.manual_seed(42)
    return torch.randn(1, 16)


# ============== Test get_meta_aggregation factory ==============

class TestGetMetaAggregation:
    """Test the factory function for creating aggregation modules."""
    
    def test_get_max_aggregation(self, max_config):
        """Test factory returns MaxAggregation for 'max' config."""
        agg = get_meta_aggregation(max_config)
        assert isinstance(agg, MaxAggregation)
        assert isinstance(agg, BaseMetaAggregation)
        assert isinstance(agg, nn.Module)
    
    def test_get_mean_aggregation(self, mean_config):
        """Test factory returns MeanAggregation for 'mean' config."""
        agg = get_meta_aggregation(mean_config)
        assert isinstance(agg, MeanAggregation)
        assert isinstance(agg, BaseMetaAggregation)
    
    def test_get_softmax_aggregation(self, softmax_config):
        """Test factory returns SoftmaxPoolAggregation for 'softmax_pool' config."""
        agg = get_meta_aggregation(softmax_config)
        assert isinstance(agg, SoftmaxPoolAggregation)
        assert isinstance(agg, BaseMetaAggregation)

    def test_get_attention_aggregation(self, attention_config):
        """Test factory returns AttentionAggregation for 'attention' config."""
        agg = get_meta_aggregation(attention_config)
        assert isinstance(agg, AttentionAggregation)
        assert isinstance(agg, BaseMetaAggregation)

    def test_get_gated_aggregation(self, gated_config):
        """Test factory returns GatedAggregation for 'gated' config."""
        agg = get_meta_aggregation(gated_config)
        assert isinstance(agg, GatedAggregation)
        assert isinstance(agg, BaseMetaAggregation)

    def test_get_set_transformer_aggregation(self, set_transformer_config):
        """Test factory returns SetTransformerAggregation for 'set_transformer' config."""
        agg = get_meta_aggregation(set_transformer_config)
        assert isinstance(agg, SetTransformerAggregation)
        assert isinstance(agg, BaseMetaAggregation)

    def test_unknown_aggregation_raises(self):
        """Test factory raises ValueError for unknown aggregation type."""
        config = SimpleNamespace(name="unknown_type")
        with pytest.raises(ValueError, match="Unknown MetaAggregation"):
            get_meta_aggregation(config)


# ============== Test MaxAggregation ==============

class TestMaxAggregation:
    """Test MaxAggregation module."""
    
    def test_output_shape(self, max_config, sample_latent_vectors):
        """Test output has correct shape (1, latent_dim)."""
        agg = MaxAggregation(max_config)
        output = agg(sample_latent_vectors)
        assert output.shape == (1, 16)
    
    def test_max_operation(self, max_config):
        """Test that max aggregation correctly takes element-wise maximum."""
        agg = MaxAggregation(max_config)
        # Create known input where max is predictable
        latent = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0],
            [2.0, 5.0, 1.0],
        ])
        output = agg(latent)
        expected = torch.tensor([[4.0, 5.0, 3.0]])
        assert torch.allclose(output, expected)
    
    def test_single_vector_passthrough(self, max_config, single_latent_vector):
        """Test that single vector is passed through unchanged (except shape)."""
        agg = MaxAggregation(max_config)
        output = agg(single_latent_vector)
        assert output.shape == (1, 16)
        assert torch.allclose(output, single_latent_vector)
    
    def test_gradient_flow(self, max_config, sample_latent_vectors):
        """Test that gradients flow through the max operation."""
        agg = MaxAggregation(max_config)
        latent = sample_latent_vectors.clone().requires_grad_(True)
        output = agg(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None
        # Gradients should be sparse (only max elements get gradient)
        assert (latent.grad != 0).sum() > 0


# ============== Test MeanAggregation ==============

class TestMeanAggregation:
    """Test MeanAggregation module."""
    
    def test_output_shape(self, mean_config, sample_latent_vectors):
        """Test output has correct shape (1, latent_dim)."""
        agg = MeanAggregation(mean_config)
        output = agg(sample_latent_vectors)
        assert output.shape == (1, 16)
    
    def test_mean_operation(self, mean_config):
        """Test that mean aggregation correctly computes element-wise mean."""
        agg = MeanAggregation(mean_config)
        latent = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0],
            [2.0, 5.0, 1.0],
        ])
        output = agg(latent)
        expected = torch.tensor([[7.0/3, 8.0/3, 6.0/3]])
        assert torch.allclose(output, expected)
    
    def test_single_vector_passthrough(self, mean_config, single_latent_vector):
        """Test that single vector is passed through unchanged."""
        agg = MeanAggregation(mean_config)
        output = agg(single_latent_vector)
        assert output.shape == (1, 16)
        assert torch.allclose(output, single_latent_vector)
    
    def test_gradient_flow(self, mean_config, sample_latent_vectors):
        """Test that gradients flow through the mean operation."""
        agg = MeanAggregation(mean_config)
        latent = sample_latent_vectors.clone().requires_grad_(True)
        output = agg(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None
        # All elements should receive equal gradient
        expected_grad = torch.ones_like(latent) / latent.shape[0]
        assert torch.allclose(latent.grad, expected_grad)


# ============== Test SoftmaxPoolAggregation ==============

class TestSoftmaxPoolAggregation:
    """Test SoftmaxPoolAggregation module."""
    
    def test_output_shape(self, softmax_config, sample_latent_vectors):
        """Test output has correct shape (1, latent_dim)."""
        agg = SoftmaxPoolAggregation(softmax_config)
        output = agg(sample_latent_vectors)
        assert output.shape == (1, 16)
    
    def test_beta_not_trainable(self, softmax_config):
        """Test that beta is not trainable when beta_trainable=False."""
        agg = SoftmaxPoolAggregation(softmax_config)
        assert not isinstance(agg.beta, nn.Parameter)
        # Should be a buffer
        assert "beta" in dict(agg.named_buffers())
        assert agg.beta.item() == pytest.approx(1.0)
    
    def test_beta_trainable(self, softmax_trainable_config):
        """Test that beta is trainable when beta_trainable=True."""
        agg = SoftmaxPoolAggregation(softmax_trainable_config)
        assert isinstance(agg.beta, nn.Parameter)
        assert "beta" in dict(agg.named_parameters())
        assert agg.beta.item() == pytest.approx(2.0)
    
    def test_high_beta_approaches_max(self):
        """Test that high beta makes softmax pool behave like max."""
        config = SimpleNamespace(name="softmax_pool", beta_init=100.0, beta_trainable=False)
        agg = SoftmaxPoolAggregation(config)
        max_agg = MaxAggregation(SimpleNamespace(name="max"))
        
        latent = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0],
            [2.0, 5.0, 1.0],
        ])
        
        softmax_output = agg(latent)
        max_output = max_agg(latent)
        
        # With high beta, should be very close to max
        assert torch.allclose(softmax_output, max_output, atol=0.1)
    
    def test_low_beta_approaches_mean(self):
        """Test that low beta (close to 0) makes softmax pool behave like mean."""
        config = SimpleNamespace(name="softmax_pool", beta_init=0.001, beta_trainable=False)
        agg = SoftmaxPoolAggregation(config)
        mean_agg = MeanAggregation(SimpleNamespace(name="mean"))
        
        latent = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0],
            [2.0, 5.0, 1.0],
        ])
        
        softmax_output = agg(latent)
        mean_output = mean_agg(latent)
        
        # With very low beta, should be close to mean
        assert torch.allclose(softmax_output, mean_output, atol=0.1)
    
    def test_single_vector_passthrough(self, softmax_config, single_latent_vector):
        """Test that single vector case works correctly."""
        agg = SoftmaxPoolAggregation(softmax_config)
        output = agg(single_latent_vector)
        assert output.shape == (1, 16)
        # With single element, softmax weight is 1, so output equals input
        assert torch.allclose(output, single_latent_vector)
    
    def test_gradient_flow(self, softmax_config, sample_latent_vectors):
        """Test that gradients flow through softmax pooling."""
        agg = SoftmaxPoolAggregation(softmax_config)
        latent = sample_latent_vectors.clone().requires_grad_(True)
        output = agg(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None
        # All elements should receive some gradient (unlike max)
        assert (latent.grad != 0).all()
    
    def test_beta_gradient_when_trainable(self, softmax_trainable_config, sample_latent_vectors):
        """Test that beta receives gradients when trainable."""
        agg = SoftmaxPoolAggregation(softmax_trainable_config)
        output = agg(sample_latent_vectors)
        loss = output.sum()
        loss.backward()
        assert agg.beta.grad is not None


# ============== Test AttentionAggregation ==============

class TestAttentionAggregation:
    """Test AttentionAggregation module."""

    def test_output_shape(self, attention_config, sample_latent_vectors):
        """Test output has correct shape (1, latent_dim)."""
        agg = AttentionAggregation(attention_config)
        output = agg(sample_latent_vectors)
        assert output.shape == (1, 16)

    def test_single_vector_passthrough(self, attention_config, single_latent_vector):
        """Test that single vector case works correctly."""
        agg = AttentionAggregation(attention_config)
        output = agg(single_latent_vector)
        assert output.shape == (1, 16)
        # With single element, attention weight is 1, so output equals input
        assert torch.allclose(output, single_latent_vector)

    def test_gradient_flow(self, attention_config, sample_latent_vectors):
        """Test that gradients flow through attention."""
        agg = AttentionAggregation(attention_config)
        latent = sample_latent_vectors.clone().requires_grad_(True)
        output = agg(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None

    def test_has_learnable_parameters(self, attention_config):
        """Test that attention has learnable parameters."""
        agg = AttentionAggregation(attention_config)
        params = list(agg.parameters())
        assert len(params) > 0

    def test_permutation_equivariance_weights(self, attention_config):
        """Test that attention weights change with permutation but output is consistent."""
        agg = AttentionAggregation(attention_config)
        agg.eval()  # Deterministic mode

        torch.manual_seed(42)
        latent = torch.randn(3, 16)

        output1 = agg(latent)

        # Permute input
        perm = torch.tensor([2, 0, 1])
        latent_perm = latent[perm]
        output2 = agg(latent_perm)

        # Output should be the same (permutation invariant)
        assert torch.allclose(output1, output2, atol=1e-5)


# ============== Test GatedAggregation ==============

class TestGatedAggregation:
    """Test GatedAggregation module."""

    def test_output_shape(self, gated_config, sample_latent_vectors):
        """Test output has correct shape (1, latent_dim)."""
        agg = GatedAggregation(gated_config)
        output = agg(sample_latent_vectors)
        assert output.shape == (1, 16)

    def test_single_vector(self, gated_config, single_latent_vector):
        """Test that single vector case works correctly."""
        agg = GatedAggregation(gated_config)
        output = agg(single_latent_vector)
        assert output.shape == (1, 16)

    def test_gradient_flow(self, gated_config, sample_latent_vectors):
        """Test that gradients flow through gating."""
        agg = GatedAggregation(gated_config)
        latent = sample_latent_vectors.clone().requires_grad_(True)
        output = agg(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None

    def test_has_learnable_parameters(self, gated_config):
        """Test that gated aggregation has learnable parameters."""
        agg = GatedAggregation(gated_config)
        params = list(agg.parameters())
        assert len(params) > 0

    def test_gate_values_bounded(self, gated_config, sample_latent_vectors):
        """Test that gate values are between 0 and 1 (sigmoid output)."""
        agg = GatedAggregation(gated_config)
        gate_values = agg.gate_net(sample_latent_vectors)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()


# ============== Test SetTransformerAggregation ==============

class TestSetTransformerAggregation:
    """Test SetTransformerAggregation module."""

    def test_output_shape(self, set_transformer_config, sample_latent_vectors):
        """Test output has correct shape (1, latent_dim)."""
        agg = SetTransformerAggregation(set_transformer_config)
        output = agg(sample_latent_vectors)
        assert output.shape == (1, 16)

    def test_single_vector(self, set_transformer_config, single_latent_vector):
        """Test that single vector case works correctly."""
        agg = SetTransformerAggregation(set_transformer_config)
        output = agg(single_latent_vector)
        assert output.shape == (1, 16)

    def test_gradient_flow(self, set_transformer_config, sample_latent_vectors):
        """Test that gradients flow through set transformer."""
        agg = SetTransformerAggregation(set_transformer_config)
        latent = sample_latent_vectors.clone().requires_grad_(True)
        output = agg(latent)
        loss = output.sum()
        loss.backward()
        assert latent.grad is not None

    def test_has_learnable_parameters(self, set_transformer_config):
        """Test that set transformer has learnable parameters."""
        agg = SetTransformerAggregation(set_transformer_config)
        params = list(agg.parameters())
        assert len(params) > 0

    def test_permutation_invariance(self, set_transformer_config):
        """Test that set transformer is permutation invariant."""
        agg = SetTransformerAggregation(set_transformer_config)
        agg.eval()  # Deterministic mode

        torch.manual_seed(42)
        latent = torch.randn(5, 16)

        output1 = agg(latent)

        # Permute input
        perm = torch.tensor([4, 2, 0, 3, 1])
        latent_perm = latent[perm]
        output2 = agg(latent_perm)

        # Output should be the same (permutation invariant)
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_different_context_sizes(self, set_transformer_config):
        """Test that set transformer handles different context sizes."""
        agg = SetTransformerAggregation(set_transformer_config)

        for context_size in [1, 3, 10, 50]:
            latent = torch.randn(context_size, 16)
            output = agg(latent)
            assert output.shape == (1, 16)


# ============== Test device compatibility ==============

class TestDeviceCompatibility:
    """Test that modules work on different devices."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_max_on_cuda(self, max_config):
        """Test MaxAggregation works on CUDA."""
        agg = MaxAggregation(max_config).cuda()
        latent = torch.randn(5, 16).cuda()
        output = agg(latent)
        assert output.device.type == "cuda"
        assert output.shape == (1, 16)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mean_on_cuda(self, mean_config):
        """Test MeanAggregation works on CUDA."""
        agg = MeanAggregation(mean_config).cuda()
        latent = torch.randn(5, 16).cuda()
        output = agg(latent)
        assert output.device.type == "cuda"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_softmax_on_cuda(self, softmax_config):
        """Test SoftmaxPoolAggregation works on CUDA."""
        agg = SoftmaxPoolAggregation(softmax_config).cuda()
        latent = torch.randn(5, 16).cuda()
        output = agg(latent)
        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_on_cuda(self, attention_config):
        """Test AttentionAggregation works on CUDA."""
        agg = AttentionAggregation(attention_config).cuda()
        latent = torch.randn(5, 16).cuda()
        output = agg(latent)
        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gated_on_cuda(self, gated_config):
        """Test GatedAggregation works on CUDA."""
        agg = GatedAggregation(gated_config).cuda()
        latent = torch.randn(5, 16).cuda()
        output = agg(latent)
        assert output.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_transformer_on_cuda(self, set_transformer_config):
        """Test SetTransformerAggregation works on CUDA."""
        agg = SetTransformerAggregation(set_transformer_config).cuda()
        latent = torch.randn(5, 16).cuda()
        output = agg(latent)
        assert output.device.type == "cuda"


# ============== Test edge cases ==============

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_large_context_size(self, max_config):
        """Test with large number of context trajectories."""
        agg = MaxAggregation(max_config)
        latent = torch.randn(1000, 64)
        output = agg(latent)
        assert output.shape == (1, 64)
    
    def test_single_dimension_latent(self, mean_config):
        """Test with single-dimensional latent vectors."""
        agg = MeanAggregation(mean_config)
        latent = torch.randn(5, 1)
        output = agg(latent)
        assert output.shape == (1, 1)
    
    def test_large_latent_dimension(self, softmax_config):
        """Test with large latent dimension."""
        agg = SoftmaxPoolAggregation(softmax_config)
        latent = torch.randn(5, 1024)
        output = agg(latent)
        assert output.shape == (1, 1024)
    
    def test_negative_beta(self):
        """Test softmax pool with negative beta (inverted behavior)."""
        config = SimpleNamespace(name="softmax_pool", beta_init=-1.0, beta_trainable=False)
        agg = SoftmaxPoolAggregation(config)
        latent = torch.randn(5, 16)
        output = agg(latent)
        # Should still produce valid output
        assert output.shape == (1, 16)
        assert torch.isfinite(output).all()

    def test_attention_large_context(self, attention_config):
        """Test attention with large context size."""
        agg = AttentionAggregation(attention_config)
        latent = torch.randn(100, 16)
        output = agg(latent)
        assert output.shape == (1, 16)

    def test_set_transformer_large_context(self, set_transformer_config):
        """Test set transformer with large context size."""
        agg = SetTransformerAggregation(set_transformer_config)
        latent = torch.randn(100, 16)
        output = agg(latent)
        assert output.shape == (1, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
