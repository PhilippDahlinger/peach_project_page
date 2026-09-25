import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra import initialize, compose
from torchinfo import summary


class Dummy(nn.Module):
    def __init__(self, config, example_batch):
        self.config = config
        super().__init__()

    def forward(self, x, v, h=None):
        return torch.zeros(x.shape[0], self.config.output_dim).to(x.device)
