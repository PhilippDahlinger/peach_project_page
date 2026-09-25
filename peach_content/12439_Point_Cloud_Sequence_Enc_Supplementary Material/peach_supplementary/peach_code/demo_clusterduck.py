import os
import hydra
import torch
import traceback
import sys

from pc_mango.util.hydra_initialization import load_omega_conf_resolvers

# full stack trace
os.environ['HYDRA_FULL_ERROR'] = '1'

load_omega_conf_resolvers()

config_path = os.path.join(os.path.dirname(__file__), "config")
print(config_path)
@hydra.main(version_base=None, config_path=config_path, config_name="demo_clusterduck_config")
def train(config) -> None:
    print(f"Raw config: {config}")
    print(f"Config keys: {list(config.keys()) if config else 'EMPTY'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
        print(f"Device properties: {torch.cuda.get_device_properties(0)}")

    try:
       seed = config.seed
       print(f"Using seed: {seed}")
       data = torch.ones(100, 10) * seed
       print(f"Data sample: {data[0, 0]}")
       data = data.to("cuda")
       print(f"Data device: {data.device}")
       print(f"Data sample on GPU: {data[0, 0]}")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == '__main__':
    train()
