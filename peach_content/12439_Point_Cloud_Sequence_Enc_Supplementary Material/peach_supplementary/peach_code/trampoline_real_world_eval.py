# full stack trace
import os

import hydra
from hydra.utils import to_absolute_path
from lightning import seed_everything

from pc_mango.util.hydra_initialization import load_omega_conf_resolvers
from pc_mango.util.own_types import ConfigDict
from pc_mango.util.trampoline_real_world_eval.real_world_evaluator import (
    RealWorldEvaluator,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


@hydra.main(
    version_base=None, config_path="config", config_name="trampoline_real_world_eval_local"
)
def real_world_eval(config: ConfigDict) -> None:
    seed_everything(config.seed)
    config.model_checkpoints = [
        to_absolute_path(model_checkpoint)
        for model_checkpoint in config.model_checkpoints
    ]
    config.pointcloud_root = to_absolute_path(config.pointcloud_root)
    config.vis_output_path = to_absolute_path(config.vis_output_path)
    config.results_output_path = to_absolute_path(config.results_output_path)
    config.cache_path = to_absolute_path(config.cache_path)

    evaluator = RealWorldEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    real_world_eval()
