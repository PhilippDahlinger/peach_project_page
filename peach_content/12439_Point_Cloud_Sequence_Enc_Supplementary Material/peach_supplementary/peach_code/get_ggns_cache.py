import os

import hydra

from pc_mango.dataset import get_dataset
from pc_mango.util.hydra_initialization import load_omega_conf_resolvers
from pc_mango.util.initialization import get_data
from pc_mango.util.own_types import ConfigDict


# full stack trace
os.environ['HYDRA_FULL_ERROR'] = '1'

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def train(config: ConfigDict) -> None:
    print(config)
    train_ds, eval_ds = get_dataset(config.dataset, loading=False)






if __name__ == "__main__":
    train()