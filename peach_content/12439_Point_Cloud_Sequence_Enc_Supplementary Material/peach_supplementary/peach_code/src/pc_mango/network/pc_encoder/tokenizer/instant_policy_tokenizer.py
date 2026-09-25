import logging
from typing import Optional, Union

from torch_cluster import nearest
from torch_geometric.typing import OptTensor, PairOptTensor, Adj, PairTensor, Tensor
import torch
from torch_geometric.nn import PointNetConv, MLP, fps

from pc_mango.network.util.positional_encoder import PositionalEncoder

log = logging.getLogger(__name__)

class InstantPolicyTokenizer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_freqs = config.pos_encoder_n_freqs
        self.sa1_module = SAModule(config.ratios[0], [1 + 3, 128, 128],  # one dim feature + xyz
                                   global_nn_dims=[128, 256],
                                   num_freqs=num_freqs,
                                   scale=self.config.scales[0],
                                   norm=None)
        self.sa2_module = SAModule(config.ratios[1], [256 + 3, 512, 512],
                                   global_nn_dims=[512, self.config.embed_dim],
                                   num_freqs=num_freqs,
                                   scale=self.config.scales[1],
                                   norm=None,
                                   plain_last=True)

        if config.checkpoint is not None:
            self.load_state_dict(torch.load(config.checkpoint, map_location="cpu"), strict=True)
            log.info(
                f"Loaded Instant Policy tokenizer checkpoint from {config.checkpoint}"
            )

    def forward(self, x, pos, batch):
        idx0 = torch.arange(pos.size(0), device=pos.device)

        x, pos, batch, idx1 = self.sa1_module(x, pos, batch, idx0)
        x, pos, batch, idx2 = self.sa2_module(x, pos, batch, idx1)

        return x, pos, batch, idx2


class SAModule(torch.nn.Module):
    def __init__(self, ratio, nn_dims, global_nn_dims=None, num_freqs=4, aggr='mean', cat_pos=False,
                 scale=1., plain_last=False, norm=None):
        super().__init__()
        self.cat_pos = cat_pos
        self.ratio = ratio
        self.conv = PointNetConvPE(nn_dims, global_nn_dims, aggr=aggr, num_freqs=num_freqs, cat_pos=cat_pos,
                                   scale=scale, plain_last=plain_last, norm=norm)

    def forward(self, x, pos, batch, idx_in=None):
        if idx_in is None:
            idx_in = torch.arange(pos.size(0), device=pos.device)

        idx = fps(pos, batch, ratio=self.ratio)

        # track original indices
        idx_out = idx_in[idx]

        row = nearest(pos, pos[idx], batch, batch[idx])
        col = torch.arange(pos.size(0), device=pos.device)
        edge_index = torch.stack([col, row], dim=0)

        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        pos, batch = pos[idx], batch[idx]

        return x, pos, batch, idx_out


class PointNetConvPE(PointNetConv):
    def __init__(self, nn_dims, global_nn_dims=None, aggr='mean', num_freqs=4, cat_pos=False,
                 scale=1., plain_last=False, norm=None):
        self.scale = scale

        # Adjust nn_dims to include positional encoding.
        nn_dims[0] += 3 * (2 * num_freqs)
        nn = MLP(nn_dims, norm=None, act=torch.nn.GELU(approximate='tanh'), plain_last=False)

        if cat_pos and global_nn_dims is not None:
            global_nn_dims[0] += 3 * (2 * num_freqs + 1)

        global_nn = None if global_nn_dims is None else \
            MLP(global_nn_dims, norm=None, act=torch.nn.GELU(approximate='tanh'),
                plain_last=plain_last)

        self.cat_pos = cat_pos
        super().__init__(nn, global_nn=global_nn, add_self_loops=False, aggr=aggr)
        self.pe = PositionalEncoder(3, num_freqs)

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        msg = self.pe((pos_j - pos_i) * self.scale)
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:

        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            if self.cat_pos:
                out = torch.cat([out, self.pe(pos[1])], dim=1)
            out = self.global_nn(out)

        return out
