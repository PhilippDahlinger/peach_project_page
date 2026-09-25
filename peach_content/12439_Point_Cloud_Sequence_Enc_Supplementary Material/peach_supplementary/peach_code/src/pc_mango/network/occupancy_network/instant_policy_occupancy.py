import torch
from torch import nn
from torch_geometric.nn import nearest

from pc_mango.network.util.positional_encoder import PositionalEncoder


class Decoder(nn.Module):
    def __init__(self, nn_dims):
        super().__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i + 1]) for i in range(len(nn_dims) - 1)])
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            if i == 0 or i == len(self.linear_layers) - 1:
                x = layer(x)
            else:
                x = x + layer(x)
            if i != len(self.linear_layers) - 1:
                x = self.act(x)
        return x

class OccupancyNetwork(nn.Module):
    def __init__(self, config, example_batch):
        super().__init__()
        self.config = config
        self.local_pos_encoder = PositionalEncoder(d_input=3, n_freqs=config.pos_encoder_n_freqs,
                                                   log_space=False)

        decoder_dims = config.decoder_dims
        # replace first dim with input dim
        decoder_dims[0] = config.embed_dim + self.local_pos_encoder.d_output
        self.local_decoder = Decoder(decoder_dims)

    def forward(self, queries, queries_batch, node_embds, node_pos, node_batch):
        row = nearest(queries, node_pos, queries_batch, node_batch)
        local_queries = self.local_pos_encoder(node_pos[row] - queries)
        query_x = torch.cat([node_embds[row], local_queries], dim=1)
        occupancy = self.local_decoder(query_x)
        return occupancy



