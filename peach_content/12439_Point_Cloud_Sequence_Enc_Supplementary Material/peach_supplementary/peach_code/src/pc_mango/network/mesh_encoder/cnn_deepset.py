import torch
from torch import nn

from pc_mango.dataset.util.graph_input_output_util import unpack_ml_batch
from torch_geometric.nn import MLP



class CNNDeepSet(nn.Module):
    """
    Makes the time aggregation first and then does deepset on initial pos with computed time h
    """

    def __init__(self, config, example_input_batch):
        super().__init__()
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(
            example_input_batch, remove_batch_dim=True)
        self.centralize_input = config.get("centralize_input", False)
        self.z_pos_feature = config.get("z_pos_feature", False)
        self.input_dim = x.shape[-1] + v.shape[-1] + h.shape[-1] + 1 * self.z_pos_feature
        input_length = x.shape[1]
        self.cnn_backbone = self.make_cnn_layers(input_length, config, h)
        # self.embedding = MLP([self.input_dim, config.latent_dimension], norm=None)
        self.node_mlp_inner = MLP([self.input_dim, config.latent_dimension, config.latent_dimension],
                                  norm="layer_norm")
        self.node_mlp_outer = MLP([config.latent_dimension, config.latent_dimension, config.output_dim],
                                  norm="layer_norm")

    def make_cnn_layers(self, input_length, config, h):
        layers = []
        current_len = input_length
        out_channels_list = (
                [config.latent_dimension] * 4 + [h.shape[-1]]
        )
        in_ch = self.input_dim

        for i, out_ch in enumerate(out_channels_list):
            is_last = i == len(out_channels_list) - 1

            # Clamp kernel size so it never exceeds the current length
            k = min(3, current_len)
            pad = 0 if is_last else 1  # last layer uses padding=0 as before

            # If kernel > current length after clamping, skip (shouldn't happen now)
            layers.append(
                torch.nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad)
            )

            # Update length after conv
            current_len = current_len + 2 * pad - k + 1

            if not is_last:
                # Clamp pool kernel too
                pool_k = min(2, current_len)
                if pool_k >= 1 and current_len > 1:
                    layers.append(torch.nn.MaxPool1d(kernel_size=pool_k, stride=pool_k))
                    current_len = current_len // pool_k
                layers.append(torch.nn.LeakyReLU())
            else:
                layers.append(torch.nn.LeakyReLU())

            in_ch = out_ch

        return torch.nn.Sequential(*layers)

    def forward(self, x, v, h) -> torch.Tensor:
        if self.z_pos_feature:
            z_pos = x[:, :, :, 2:3]
        if self.centralize_input:
            x = x - x[:, 0:1, :, :].mean(dim=2, keepdim=True)
        if len(h.shape) == 3:
            h = h[:, None, :, :].expand(-1, x.shape[1], -1, -1)
        batch_dim = x.shape[0]
        num_nodes = x.shape[2]
        num_ts = x.shape[1]
        if self.z_pos_feature:
            cnn_features = torch.cat([x, v, h, z_pos], dim=-1)
        else:
            cnn_features = torch.cat([x, v, h], dim=-1)
        cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(-1, self.input_dim, num_ts)
        cnn_features = self.cnn_backbone(cnn_features)
        output_num_ts = cnn_features.shape[2]
        output = cnn_features.reshape(batch_dim, num_nodes, -1, output_num_ts).permute(0, 3, 1, 2)
        h = output[:, 0, :, :]
        # mesh pos and vel with h
        if self.z_pos_feature:
            input_vector = torch.cat([x[:, 0, :, :], v[:, 0, :, :], h, z_pos[:, 0, :, :]], dim=-1)
        else:
            input_vector = torch.cat([x[:, 0, :, :], v[:, 0, :, :], h], dim=-1)
        all_nodes_output = self.node_mlp_inner(input_vector)
        all_nodes_output = all_nodes_output.mean(dim=1)
        context_output = self.node_mlp_outer(all_nodes_output)
        # shape [num_context_trajs, output_dim]
        return context_output
