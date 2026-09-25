import torch

from pc_mango.util.own_types import ConfigDict


def get_occupancy_network(config: ConfigDict, example_batch) -> torch.nn.Module:
    occupancy_name = config.name
    if occupancy_name == "instant_policy":
            from pc_mango.network.occupancy_network.instant_policy_occupancy import OccupancyNetwork
            return OccupancyNetwork(config, example_batch)
    else:
        raise ValueError(f"Unknown occupancy network {occupancy_name}")
