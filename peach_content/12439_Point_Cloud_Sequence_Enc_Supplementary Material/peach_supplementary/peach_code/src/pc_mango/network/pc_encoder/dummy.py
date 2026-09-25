import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra import initialize, compose
from torchinfo import summary


class Dummy(nn.Module):
    def __init__(self, config, example_batch):
        self.config = config
        super().__init__()

    def forward(self, xyzs, colors):
        return torch.zeros(xyzs.shape[0], self.config.output_dim).to(xyzs.device)
