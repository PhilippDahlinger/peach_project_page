import torch
from torch import nn

from pc_mango.network.util.fourier_features import FourierFeatures
from pc_mango.network.util.positional_encoder import PositionalEncoder


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, nn_dims):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(nn_dims[i], nn_dims[i + 1])
            for i in range(len(nn_dims) - 1)
        ])
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


# -------------------------
# Spacetime MLP with kNN + distance-aware attention
# -------------------------
class SpacetimeMLP(nn.Module):
    def __init__(self, config, example_batch):
        super().__init__()
        self.config = config

        # ---- k for kNN ----
        self.k = getattr(config, "knn_k", 1)

        # ---- Positional encoding (4D spacetime) ----
        self.local_pos_encoder = FourierFeatures(
            input_dim=4,
            n_wavelengths=config.fourier_features.n_wavelengths,
            max_wavelength=config.fourier_features.max_wavelength,
            min_wavelength=config.fourier_features.min_wavelength,
        )

        # ---- Output dimension (supports multi-channel SDF) ----
        example_sdf = example_batch["sdf"]
        sdf_dim = example_sdf.shape[-1]

        # ---- Attention MLP ----
        hidden_dim = config.embed_dim + self.local_pos_encoder.out_features
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )


        # ---- Decoder ----
        decoder_dims = list(config.decoder_dims)
        decoder_dims[0] = hidden_dim
        decoder_dims[-1] = sdf_dim
        self.decoder = Decoder(decoder_dims)


        # ---- Distance scaling (learnable) ----
        self.distance_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, queries, tokens, centers):
        """
        queries: (B, Q, 4)
        tokens: (B, M, D)
        centers: (B, M, 4)
        """

        B, Q, _ = queries.shape
        M = centers.shape[1]
        k = self.k

        # ---- Pairwise distances ----
        # (B, Q, M)
        dists = torch.cdist(queries, centers)

        # ---- k-NN ----
        knn_dists, knn_idx = torch.topk(dists, k, dim=-1, largest=False)

        # ---- Gather centers ----
        knn_centers = torch.gather(
            centers.unsqueeze(1).expand(-1, Q, -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, centers.size(-1))
        )  # (B, Q, k, 4)

        # ---- Gather tokens ----
        knn_tokens = torch.gather(
            tokens.unsqueeze(1).expand(-1, Q, -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, tokens.size(-1))
        )  # (B, Q, k, D)

        # ---- Relative spacetime encoding ----
        rel_pos = knn_centers - queries.unsqueeze(2)  # (B, Q, k, 4)

        # Optional normalization (can help stability)
        rel_pos = rel_pos / (knn_dists.unsqueeze(-1) + 1e-6)

        rel_pos_enc = self.local_pos_encoder(rel_pos)  # (B, Q, k, PE)

        # ---- Combine features ----
        features = torch.cat([knn_tokens, rel_pos_enc], dim=-1)  # (B, Q, k, F)

        # ---- Attention (distance-aware) ----
        attn_logits = self.attn_mlp(features)  # (B, Q, k, 1)

        # normalize distances (important for stability)
        norm_dists = knn_dists / (knn_dists.mean(dim=2, keepdim=True) + 1e-6)

        distance_bias = norm_dists.unsqueeze(-1)

        attn_logits = attn_logits - self.distance_scale * distance_bias

        attn = torch.softmax(attn_logits, dim=2)  # (B, Q, k, 1)

        # ---- Aggregate ----
        aggregated = (features * attn).sum(dim=2)  # (B, Q, F)

        # ---- Decode ----
        sdf = self.decoder(aggregated)  # (B, Q, sdf_dim)

        return sdf