import torch

from pc_mango.util.own_types import ConfigDict


def get_sdf_head(config: ConfigDict, example_batch) -> torch.nn.Module:
    encoder_name = config.name
    match encoder_name:
        case "spacetime_mlp":
            from pc_mango.network.sdf_head.spacetime_mlp import SpacetimeMLP
            return SpacetimeMLP(config, example_batch)
        case _:
            raise ValueError(f"Unknown SDF head {encoder_name}")