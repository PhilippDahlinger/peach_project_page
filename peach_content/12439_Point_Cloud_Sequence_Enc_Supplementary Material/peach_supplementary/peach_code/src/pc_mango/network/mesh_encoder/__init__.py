import torch

from pc_mango.dataset.util.graph_input_output_util import unpack_ml_batch
from pc_mango.util.own_types import ConfigDict


def get_mesh_encoder(config: ConfigDict, example_batch) -> torch.nn.Module:
    encoder_name = config.name
    # check that example batch has a mesh inside
    try:
        _ = unpack_ml_batch(
            example_batch, remove_batch_dim=False)
    except KeyError:
        raise ValueError("Example batch does not contain mesh data required for mesh encoder. Change data_modalities config and include mesh")

    match encoder_name:
        case "cnn_deepset":
            from pc_mango.network.mesh_encoder.cnn_deepset import CNNDeepSet
            return CNNDeepSet(config, example_batch)
        case "dummy":
            from pc_mango.network.mesh_encoder.dummy import Dummy
            return Dummy(config, example_batch)
        case _:
            raise ValueError(f"Unknown mesh encoder {encoder_name}")