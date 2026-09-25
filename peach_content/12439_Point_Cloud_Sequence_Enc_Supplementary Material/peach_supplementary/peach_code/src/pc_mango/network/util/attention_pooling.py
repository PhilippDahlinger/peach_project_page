from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Mapping

import torch.nn as nn
from torch import Tensor
from pc_mango.network.util.attention import MultiHeadAttention, MultiHeadSelfAttention
from pc_mango.network.util.transformer import AttentionPoolingLayer, TransformerEncoder

log = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int,
        n_tokens: int,
        norm: type[nn.Module],
        attention: Mapping | Callable[..., nn.Module],
        residual_dropout: float | Callable[[], nn.Module],
        mlp1: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout: float | Callable[[], nn.Module],
        mlp2: type[nn.Module],
        norm_first: bool,
    ):
        super().__init__()

        if isinstance(attention, Mapping):
            attention_kwargs = {**attention}
        elif isinstance(attention, partial):
            attention_kwargs = (
                attention.keywords if attention.keywords is not None else {}
            )
        else:
            raise ValueError(
                "attention must be a Mapping or a functools.partial instance"
            )

        self_attention = partial(MultiHeadSelfAttention, **attention_kwargs)
        cross_attention = partial(MultiHeadAttention, **attention_kwargs)

        if n_layers > 0:
            self.transformer = TransformerEncoder(
                embed_dim=input_dim,
                n_layers=n_layers,
                norm=norm,
                attention=self_attention,
                residual_dropout=residual_dropout,
                mlp1=mlp1,
                activation=activation,
                mlp_dropout=mlp_dropout,
                mlp2=mlp2,
                norm_first=norm_first,
            )
        else:
            self.transformer = nn.Identity()

        self.pooling = AttentionPoolingLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            n_tokens=n_tokens,
            norm=norm,
            attention=cross_attention,
            residual_dropout=residual_dropout,
            mlp1=mlp1,
            activation=activation,
            mlp_dropout=mlp_dropout,
            mlp2=mlp2,
            norm_first=norm_first,
        )
        # TransformerEncoder uses a norm after each layer
        self.final_norm = norm(output_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.transformer(tokens)
        tokens = self.pooling(tokens)
        tokens = self.final_norm(tokens)
        return tokens
