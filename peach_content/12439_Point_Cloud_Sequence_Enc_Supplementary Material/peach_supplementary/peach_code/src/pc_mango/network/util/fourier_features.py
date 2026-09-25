from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_wavelengths: int | None = None,
        max_wavelength: float | None = None,
        min_wavelength: float | None = None,
        factor: float | None = None,
        embed_dim: int | None = None,
        components: Literal["sincos", "sin"] = "sincos",
        cat_input_to_out: bool = False,
    ) -> None:
        super().__init__()

        if components not in ("sincos", "sin"):
            raise ValueError("components must be 'sincos' or 'sin'")

        n_components = 2 if components == "sincos" else 1

        if embed_dim is not None:
            max_n_wavelengths = (
                embed_dim - (input_dim if cat_input_to_out else 0)
            ) // (input_dim * n_components)
        else:
            max_n_wavelengths = None

        # If only 2 of the 4 parameters are provided, but we are given an embed_dim,
        # use the maximum number of wavelengths that fit in the embed_dim
        args = (max_wavelength, min_wavelength, factor)
        if (
            n_wavelengths is None
            and max_n_wavelengths is not None
            and len([arg for arg in args if arg is not None]) == 2
        ):
            n_wavelengths = max_n_wavelengths
            log.info(
                f"Using maximum n_wavelengths={n_wavelengths} given embed_dim={embed_dim}"
            )

        log_frequencies = linspace_logfrequencies(
            n_wavelengths=n_wavelengths,
            max_wavelength=max_wavelength,
            min_wavelength=min_wavelength,
            factor=factor,
        )

        n_wavelengths = len(log_frequencies)
        min_wavelength = torch.exp(-log_frequencies[-1]).item()
        max_wavelength = torch.exp(-log_frequencies[0]).item()
        if len(log_frequencies) >= 2:
            factor = torch.exp(log_frequencies[1] - log_frequencies[0]).item()
        else:
            factor = float("nan")
        log.info(
            f"Using {n_wavelengths} Fourier wavelengths from {max_wavelength:.3f} to {min_wavelength:.3f} with an factor of {factor:.3f}"
        )

        frequencies = torch.exp(log_frequencies)
        frequencies = 2 * torch.pi * frequencies
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)

        # add 1 for the original coordinates if we are concatenating them
        feature_dim = input_dim * (
            n_components * n_wavelengths + (1 if cat_input_to_out else 0)
        )
        if embed_dim is not None:
            if feature_dim > embed_dim:
                raise ValueError(
                    f"embed_dim={embed_dim} is too small for the given configuration. "
                    f"Minimum required is {feature_dim}."
                )
            padding_dim = embed_dim - feature_dim
        else:
            padding_dim = 0
            embed_dim = feature_dim

        self.components = components
        self.cat_input_to_out = cat_input_to_out
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.padding_dim = padding_dim
        self.n_wavelengths = n_wavelengths
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.factor = factor

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 1 and pos.size(-1) != 1:
            pos = pos.unsqueeze(dim=-1)

        # multiply each coordinate by each frequency using broadcasting
        # pos -> (..., input_dim, 1) * (n_frequencies)
        # arg: (..., input_dim, n_frequencies)
        arg = pos.unsqueeze(dim=-1) * self.frequencies

        if self.components == "sincos":
            # concatenate sin(ωx) and cos(ωx) of each feature
            # if axis-aligned (not flattened), each coordinate is kept separate
            features = (arg.sin(), arg.cos())
        else:  # components == "sin"
            features = (arg.sin(),)

        # features: (..., input_dim, n_components * n_frequencies)

        if self.cat_input_to_out:
            # concatenate the original coordinates with the sin/cos components
            features = (pos.unsqueeze(dim=-1), *features)

        if len(features) > 1:
            # concatenate all features along the last dimension
            features = torch.cat(features, dim=-1)
        else:
            features = features[0]

        # features -> (..., input_dim * n_components * n_frequencies)
        features = features.flatten(start_dim=-2)

        if self.padding_dim > 0:
            padding_shape = pos.shape[:-1] + (self.padding_dim,)
            features = torch.cat((features, pos.new_zeros(padding_shape)), dim=-1)

        return features

    @property
    def wavelengths(self) -> Tensor:
        return (2 * torch.pi) / self.frequencies

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.input_dim

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.embed_dim

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        lines = extra_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self):
        args = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"wavelengths=logspace({self.max_wavelength:.2f}, {self.min_wavelength:.4f}, n={self.n_wavelengths}, factor={self.factor:.2f})",
        ]

        components = "sin/cos" if self.components == "sincos" else "sin"
        args.append(f"components={components}")
        args.append(f"cat_input_to_out={self.cat_input_to_out}")

        s = ",\n".join(args)

        return s


def linspace_logfrequencies(
    n_wavelengths: int | None = None,
    max_wavelength: float | None = None,
    min_wavelength: float | None = None,
    factor: float | None = None,
) -> Tensor:
    if n_wavelengths is not None and n_wavelengths < 2:
        raise ValueError("n_wavelengths must be at least 2 if provided.")

    if factor is None:
        if any(x is None for x in (n_wavelengths, max_wavelength, min_wavelength)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert n_wavelengths is not None
        assert max_wavelength is not None
        assert min_wavelength is not None

        # frequencies increase exponentially from 1/max_wavelength to 1/min_wavelength
        log_wavelengths = torch.linspace(
            start=math.log(max_wavelength),
            end=math.log(min_wavelength),
            steps=n_wavelengths,
        )

    elif min_wavelength is None:
        if any(x is None for x in (n_wavelengths, max_wavelength, factor)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert n_wavelengths is not None
        assert max_wavelength is not None
        assert factor is not None

        # wavelengths decrease exponentially from max_wavelength by factor
        log_wavelengths = -torch.arange(n_wavelengths) * math.log(factor)
        log_wavelengths += math.log(max_wavelength)

    elif max_wavelength is None:
        if any(x is None for x in (n_wavelengths, min_wavelength, factor)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert n_wavelengths is not None
        assert min_wavelength is not None
        assert factor is not None

        log_wavelengths = torch.arange(n_wavelengths - 1, -1, -1) * math.log(factor)
        log_wavelengths += math.log(min_wavelength)

    elif n_wavelengths is None:
        if any(x is None for x in (max_wavelength, min_wavelength, factor)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert max_wavelength is not None
        assert min_wavelength is not None
        assert factor is not None

        log_wavelengths = torch.arange(
            math.log(max_wavelength), math.log(min_wavelength), -math.log(factor)
        )
        n_wavelengths = len(log_wavelengths)

    else:
        raise ValueError(
            "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
        )

    # frequencies are 1 / wavelengths
    return -log_wavelengths
