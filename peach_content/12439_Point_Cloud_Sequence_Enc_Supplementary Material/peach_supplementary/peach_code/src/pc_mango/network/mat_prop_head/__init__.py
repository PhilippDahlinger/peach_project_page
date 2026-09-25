import torch

from pc_mango.util.own_types import ConfigDict


def get_mat_prop_head(config: ConfigDict, input_dim, output_dim) -> torch.nn.Module:
    mat_prop_head_name = config.name
    match mat_prop_head_name:
        case "mlp":
            from pc_mango.network.mat_prop_head.mlp import MLP
            return MLP(config, input_dim=input_dim, output_dim=output_dim)
        case "linear":
            return torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        case _:
            raise ValueError(f"Unknown MatPropHead {mat_prop_head_name}")

