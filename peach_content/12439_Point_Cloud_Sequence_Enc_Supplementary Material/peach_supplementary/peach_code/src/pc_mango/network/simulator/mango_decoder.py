import torch
from omegaconf import OmegaConf

from pc_mango.dataset.util.graph_input_output_util import unpack_ml_batch
from pc_mango.network.util.mango.decoder import Decoder
from pc_mango.network.util.mango.scaled_tanh import ScaledTanh
from pc_mango.network.util.mango.mango_backbone import MangoBackbone, MangoBackboneV2
from pc_mango.network.util.mango.mlp import MLP

class MangoDecoder(torch.nn.Module):

    def __init__(self, config, example_input_batch, context_encoding_dimension):
        super().__init__()
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(
            example_input_batch,
            remove_batch_dim=True)
        self.z_pos_feature = config.z_pos_feature
        self.context_encoding_dimension = context_encoding_dimension
        # although we will only use the first step and repeat that for decoding, the shape is the same for x and v
        world_dim = x.shape[-1]

        # Select backbone version based on config
        backbone_version = getattr(config, 'backbone_version', 'v1')
        if backbone_version == 'v2':
            self.mango_backbone = MangoBackboneV2(
                n_layers=config.n_layers,
                h_dim=h.shape[-1] + context_encoding_dimension + 1 * self.z_pos_feature,
                edge_feature_dim=edge_features.shape[-1],
                world_dim=world_dim,
                latent_dim=config.latent_dimension,
                activation=config.activation,
                time_emb_dim=config.time_emb_dim,
                scatter_reduce=config.scatter_reduce,
                use_hidden_layers=config.use_hidden_layers,
                use_time_conv=True,
                time_conv_type=config.time_conv_type,
                dropout_embedding=getattr(config, 'dropout_embedding', None),
                dropout_node=getattr(config, 'dropout_node', None),
                dropout_edge=getattr(config, 'dropout_edge', None),
                drop_edge_p=getattr(config, 'drop_edge_p', 0.0),
            )
        else:
            # Default to original MangoBackbone (v1)
            self.mango_backbone = MangoBackbone(
                n_layers=config.n_layers,
                h_dim=h.shape[-1] + context_encoding_dimension + 1 * self.z_pos_feature,
                edge_feature_dim=edge_features.shape[-1],
                world_dim=world_dim,
                latent_dim=config.latent_dimension,
                activation=config.activation,
                time_emb_dim=config.time_emb_dim,
                scatter_reduce=config.scatter_reduce,
                use_hidden_layers=config.use_hidden_layers,
                use_time_conv=True,
                time_conv_type=config.time_conv_type,
                dilation_in_later_conv_layers=config.get("dilation_in_later_conv_layers", False),
            )

        decode_module = MLP(in_features=config.latent_dimension,
                            latent_dimension=config.output_decoder.latent_dimension,
                            config=OmegaConf.create(dict(activation_function="relu",
                                                         add_output_layer=False,
                                                         num_layers=1,
                                                         regularization={
                                                             "dropout": config.output_decoder.regularization.dropout,
                                                         },
                                                         )),
                            )
        readout_module = torch.nn.Linear(config.output_decoder.latent_dimension, world_dim)
        if config.output_decoder.tanh.enabled:
            output_activation = ScaledTanh(config.output_decoder.tanh.scale_factor,
                                           config.output_decoder.tanh.input_scale_factor)
        else:
            output_activation = torch.nn.Identity()
        self.output_decoder = Decoder(decode_module, readout_module, output_activation)

    def forward(self, batch, encoding) -> torch.Tensor:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)
        # add context encoding to h tensor
        if len(h.shape) == 4:
            encoding = encoding.unsqueeze(0).expand(h.shape[0], h.shape[1], h.shape[2], -1)
        else:
            encoding = encoding.unsqueeze(0).expand(h.shape[0], h.shape[1], -1)
        mango_h = torch.cat((h, encoding), dim=-1)
        # take initial x/v state and repeat it for all time steps
        num_timesteps = x.shape[1]
        init_x = x[:, 0:1, :, :]
        init_v = v[:, 0:1, :, :]
        mango_x = init_x.repeat(1, num_timesteps, 1, 1)
        mango_v = init_v.repeat(1, num_timesteps, 1, 1)
        # collider nodes should be present in all time steps
        last_index = len(h_description) - 1 - h_description[::-1].index("one_hot")
        if last_index == 2 or last_index == 1:
            collider_index = 1
            # deformable, collider, encoder_node
            collider_mask = h[0, :, collider_index] == 1
            collider_nodes = x[:, :, collider_mask, :]
            mango_x[:, :, collider_mask, :] = collider_nodes
            collider_v = v[:, :, collider_mask, :]
            mango_v[:, :, collider_mask, :] = collider_v
        if last_index > 2:
            # something not implemented yet
            raise NotImplementedError("More than 3 node types detected.")
        # predict all trajectories
        mango_x = mango_x[target_trajs]
        mango_v = mango_v[target_trajs]
        mango_h = mango_h[target_trajs]
        edge_features = edge_features[target_trajs]
        if self.z_pos_feature:
            # ml case with time dimension
            mango_h = mango_h[:, None, :, :].repeat(1, x.shape[1], 1, 1)
            mango_h = torch.cat((mango_h, mango_x[:, :, :, -1:]), dim=-1)
        latent_prediction = self.mango_backbone(mango_x, mango_h, mango_v, edge_indices, edge_features)
        displacements = self.output_decoder(latent_prediction)
        # add displacements to initial prediction, but only for deformable nodes
        if len(h.shape) == 4:
            # since node type does not change over time, take the first time step
            deformable_mask = h[0, 0, :, 0] == 1
        else:
            deformable_mask = h[0, :, 0] == 1
        if "collider_identifier" in h_description:
            # rigid predicted collider present
            predicted_collider_mask = h[0, :, h_description.index("collider_identifier")] == 1
            rigid_collider_displacements = displacements[:, :, predicted_collider_mask, :]
            # average the predicted collider displacements to get a single rigid transformation
            mean_collider_displacement = rigid_collider_displacements.mean(dim=2, keepdim=True)
            # update the displacements for the predicted collider nodes to be the mean displacement
            displacements[:, :, predicted_collider_mask, :] = mean_collider_displacement

        predictions = mango_x[:, :, deformable_mask, :] + displacements[:, :, deformable_mask, :]
        return predictions
