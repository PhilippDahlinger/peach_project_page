import torch


def get_simulator(config, example_batch, encoder_output_dim) -> torch.nn.Module:
    simulator_name = config.name
    match simulator_name:
        case "mango_decoder":
            from pc_mango.network.simulator.mango_decoder import MangoDecoder
            return MangoDecoder(config, example_batch, encoder_output_dim)
        case "mgn":
            from pc_mango.network.simulator.mgn import MGN
            return MGN(config, example_batch, encoder_output_dim)
        case _:
            raise ValueError(f"Unknown simulator {simulator_name}")