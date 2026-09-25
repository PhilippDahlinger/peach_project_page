from __future__ import annotations

import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, knn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import unbatch

from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset


class PointnetTokenizer(MessagePassing):
    def __init__(
            self,
            mlp_1: MLP | Callable,
            mlp_2: MLP | Callable,
            group_size: int,
            sampling_ratio: float,
            point_dim: int | None,
            embed_dim: int | None,
            random_start: bool = True,
            padding_value: float = 0.0,
            **kwargs,
    ):
        """
        Embedding module, which divides a point cloud into groups and uses the two MLPs to embed each group.

        :param hidden_layers: the dimensions of the hidden layers of both MLPs combined.
        :param embed_dim: the size of the embeddings
        :param group_size: the number of points contained in each neighborhood
        :param sampling_ratio: sampling ratio of the furthest point sampling algorithm
        :param random_start: whether or not to use a random point as the first neighborhood center
        """
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self._group_size = group_size
        self.sampling_ratio = sampling_ratio
        self.random_start = random_start
        self.padding_value = padding_value

        if isinstance(mlp_1, nn.Module):
            self.mlp_1 = mlp_1
        else:
            if not point_dim:
                raise ValueError(f"{point_dim=}")
            self.mlp_1 = mlp_1(input_size=point_dim)

        if isinstance(mlp_2, nn.Module):
            self.mlp_2 = mlp_2
        else:
            if not embed_dim:
                raise ValueError(f"{embed_dim=}")
            self.mlp_2 = mlp_2(output_size=embed_dim)

        assert (
                self.mlp_1.out_channels * 2 == self.mlp_2.in_channels
        ), f"The last layer of mlp_1 (size {self.mlp_1.out_channels}) must be half the size of the first layer of mlp_2 (size {self.mlp_2.in_channels})"

        self.reset_parameters()

    @property
    def embed_dim(self) -> int:
        return self.mlp_2.out_channels

    @property
    def point_dim(self) -> int:
        return self.mlp_1.in_channels

    @property
    def group_size(self) -> int:
        return self._group_size

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_1)
        reset(self.mlp_2)

    def forward(
            self, pos: Tensor, batch: Tensor, color: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Takes points as input, selects center points via furthest point
        sampling, creates local neighborhoods via k-nearest-neighbors sampling,
        and embeds the local neighborhoods with the two MLPs.

        B: batch size
        N: number of points
        G: number of groups
        M: neighborhood size
        E: embedding size

        :param pos: [B * N, 3] Tensor containing the points
        :param batch: [B * N] Tensor assigning each point in 'pos' to a batch
        (Correction from me, original comment said [B * N, 1], but it is actually [B * N])
        :returns:
            - x - [B, G, M, E] Tensor containing the embeddings
            - neighborhoods - [B, G, N, 3] Tensor containing the neighborhoods in local coordinates (with respect to the neighborhood center)
            - center_points - [B, G, 3] Tensor containing the center points of each neighborhood
        """
        _, count = torch.unique(batch, return_counts=True)

        assert torch.all(count >= self._group_size), f"COUNT: {count}"

        center_points_idx = fps(
            pos, batch, ratio=self.sampling_ratio, random_start=self.random_start
        )
        center_points = pos[center_points_idx]
        batch_y = batch[center_points_idx]

        from_idx, to_idx = knn(
            pos, center_points, self._group_size, batch_x=batch, batch_y=batch_y
        )
        edges = torch.stack([to_idx, from_idx], dim=0)
        color = (color, color[center_points_idx]) if color is not None else (None, None)
        try:
            x, neighborhoods = self.propagate(edges, pos=(pos, center_points), x=color)
        except RuntimeError as e:
            torch.set_printoptions(threshold=sys.maxsize)
            print(
                f"POS SHAPE: {pos.shape}\n BATCH SHAPE: {batch.shape}\n CENTER POINTS SHAPE: {center_points.shape}\n EDGES SHAPE: {edges.shape}\n"
            )
            print(f"{type(e).__name__}: {e}")
            raise e

        # pad and reshape into [B, M, E]
        x = pad_sequence(
            unbatch(x, batch_y),
            padding_value=self.padding_value,
            batch_first=True,
        )

        # pad and reshape into [B, G, 3]
        center_points = pad_sequence(
            unbatch(center_points, batch_y),
            padding_value=self.padding_value,
            batch_first=True,
        )

        return x, center_points, batch_y

    def aggregate(
            self,
            inputs: Tensor,
            index: Tensor,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
    ):
        msg, relative_pos = inputs
        return super().aggregate(msg, index, ptr, dim_size), relative_pos

    def message(self, pos_i: Tensor, pos_j: Tensor, x_j: Tensor):
        neighborhood = pos_j - pos_i

        if x_j is not None:
            neighborhood = torch.cat([neighborhood, x_j], dim=1)

        msg = self.mlp_1(neighborhood)
        # reshape into shape [G, M, mlp_1_out_dim]
        msg = msg.reshape(-1, self._group_size, msg.shape[-1])
        # get max over neighborhood
        msg_max = torch.max(msg, dim=1, keepdim=True)[0]
        # add the neighborhood max to the original msg for each node
        msg = torch.cat([msg_max.expand(-1, self._group_size, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(-1, msg.shape[-1]))

        return msg, neighborhood

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"


if __name__ == "__main__":
    dataset_config = {
        "root": "../datasets/pc_mango/db_v1.hdf5",
        "step": False,
        "batch_size": 1,
        "initial_vel": "zero",  # choose between "zero" and "constant"
        "dataset_size": 1,  # artificially reduce dataset size for debugging
        "random_traj_selection": True,
        "min_context_size": 1,
        "max_context_size": 8,
        "min_target_size": 1,
        "max_target_size": 1,
        "material_properties": False,
        "material_properties_dim": 2,  # 2, but one is fixed for easy
        "time_subsampling": 1,
        "data_modalities": ['pc'],
        "pc_preprocessing": {
            "output_type": "tensor",  # options: "tensor", "graph"
            "temporal_stride": 2,
            "num_subsample_points": 512,  # None means no subsampling
        }
    }
    dataset_config: DictConfig = OmegaConf.create(dataset_config)
    train_ds = DeformableBlockDataset(dataset_config, split="train")
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=1)
    batch = next(iter(train_dl))
    pc = batch["pc"][0]  # shape (traj_length, num_points, point_dim)
    mlp_1 = MLP([3, 64, 128])
    mlp_2 = MLP([256, 128, 64])
    tokenizer = PointnetTokenizer(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        group_size=16,
        sampling_ratio=0.1,
        point_dim=3,
        embed_dim=64,
        random_start=True,
        padding_value=0.0,
    )
    first_pos = pc[0]  # shape (batch_size, num_points, point_dim)
    # increment batch_idx for each batch
    batch_size, num_points, _ = first_pos.shape
    batch_idx = torch.arange(batch_size, device=first_pos.device).unsqueeze(1)
    batch_idx = batch_idx.repeat(1, num_points)
    first_pos = first_pos.reshape(-1, 3)
    batch_idx = batch_idx.reshape(-1, 1)
    x, center_points, batch_idx = tokenizer(first_pos, batch_idx[:, 0])
    print("stop")
