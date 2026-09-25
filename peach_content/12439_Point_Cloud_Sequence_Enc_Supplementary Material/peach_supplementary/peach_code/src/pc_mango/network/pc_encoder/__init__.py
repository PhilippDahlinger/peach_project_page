import hydra
import torch

from pc_mango.util.own_types import ConfigDict



def get_pc_encoder(config: ConfigDict, example_batch) -> torch.nn.Module:
    encoder_name = config.name
    match encoder_name:
        case "pstnet":
            from pc_mango.network.pc_encoder.pstnet import PSTNet
            return PSTNet(config, example_batch)
        case "dummy":
            from pc_mango.network.pc_encoder.dummy import Dummy
            return Dummy(config, example_batch)
        case "gnn_encoder":
            from pc_mango.network.pc_encoder.gnn_encoder import GNNEncoder
            return GNNEncoder(config, example_batch)
        case "ip_transformer_encoder":
            from pc_mango.network.pc_encoder.ip_transformer_encoder import IPTransformerEncoder
            return IPTransformerEncoder(config, example_batch)
        case _:
            if "_target_" in config:
                return hydra.utils.instantiate(config, example_batch=example_batch)
            raise ValueError(f"Unknown PC encoder {encoder_name}")


if __name__ == "__main__":
    config = ConfigDict({
        "name": "pstnet",
        "radius": 0.1,
        "nsamples": 16,
        "output_dim": 2048
    })
    get_pc_encoder(config, example_batch=torch.randn(4, 2048))
    print("worked")