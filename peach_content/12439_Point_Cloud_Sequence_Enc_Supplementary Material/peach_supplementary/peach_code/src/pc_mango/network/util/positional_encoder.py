import torch
from torch import nn


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False,
            add_original_x: bool = True,
            scale: float = 1.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.scale = scale

        if add_original_x:
            self.embed_fns = [lambda x: x]
            self.d_output = d_input * (1 + 2 * self.n_freqs)
        else:
            self.embed_fns = []
            self.d_output = d_input * (2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x / self.scale * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x / self.scale * freq))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.cat([f(x) for f in self.embed_fns], dim=-1)