from __future__ import annotations

from typing import Callable

import hydra
import torch
import torch.nn as nn
from torch import Tensor

from pc_mango.network.pc_encoder.tokenizer import get_tokenizer
from pc_mango.util.pyg import fps, knn, ptr2batch


class IPTransformerEncoder(nn.Module):

    def __init__(
            self,
            config,
            example_batch,
    ):
        """
        Initialize the GNN Encoder.

        Args:
            config: Configuration object with tokenizer, graph, mpn, and readout settings
            example_batch: Optional example batch for shape inference (unused, for API compatibility)
        """
        super().__init__()
        self.config = config
        self.tokenizer = get_tokenizer(config.tokenizer, example_batch)
        if self.config.frozen_tokenizer:
            # freeze tokenizer weights
            self.tokenizer.requires_grad_(False)
            self.tokenizer.eval()
            self.freeze_tokenizer = True
        else:
            self.freeze_tokenizer = False

        self._output_dim = config.output_dim
        transformer_factory = hydra.utils.instantiate(config.transformer)
        self.transformer = transformer_factory(
            input_dim=self.tokenizer.config.embed_dim,
            output_dim=self._output_dim,
        )
        center_spatial_encoder_factory = hydra.utils.instantiate(config.center_spatial_encoder)
        self.center_spatial_encoder = center_spatial_encoder_factory(3 + 1)  # xyz + time
        spacetime_dim = self.center_spatial_encoder.out_features

        token_pos_encoder_factory = hydra.utils.instantiate(config.token_pos_encoder)
        self.token_pos_encoder = token_pos_encoder_factory(spacetime_dim, self.tokenizer.config.embed_dim)


    def forward(self, pos: Tensor, color: Tensor | None = None) -> Tensor:
        B, num_timesteps, num_points, _ = pos.shape
        L = num_timesteps * num_points
        time = torch.arange(0, num_timesteps, device=pos.device) / num_timesteps
        time = time.view(1, num_timesteps, 1, 1).repeat(B, 1, num_points, 1)
        time = time.flatten(0, -2)  # time -> (B*T*N, 1)
        # spacetime: (B*T*N, 4)

        batch = torch.arange(pos.shape[0] * pos.shape[1]).repeat_interleave(pos.shape[2]).to(pos.device)
        pos_flatten = pos.flatten(0, -2)
        spacetime = torch.cat([pos_flatten, time], dim=-1)

        assert color is not None
        color_flatten = color.flatten(0, -2)
        tokens, center_points, center_batch_idx, center_indices = self.tokenizer(
            color_flatten, pos_flatten, batch
        )
        # get pos/time encodings
        center_spacetime = spacetime[center_indices]
        center_spacetime = self.center_spatial_encoder(center_spacetime)
        center_spacetime = self.token_pos_encoder(center_spacetime)
        tokens = tokens + center_spacetime
        tokens = tokens.unflatten(0, (B, -1))  # tokens -> (B, C, D)
        embedding = self.transformer(tokens)  # embedding -> (B, N, D)
        embedding = embedding.squeeze(dim=1)  # embedding -> (B, D)
        return embedding



    @property
    def output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self._output_dim
