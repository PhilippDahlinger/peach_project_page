import os
import hydra
import torch
import traceback
import sys
from lightning import Trainer

from omegaconf import OmegaConf, open_dict

from pc_mango.algorithm import get_algorithm
from pc_mango.logger.test_logger import TestLogger
from pc_mango.util.eval import get_best_checkpoint
from pc_mango.util.hydra_initialization import load_omega_conf_resolvers
from pc_mango.util.initialization import initialize_config, \
    initialize_seed, get_data
from pc_mango.util.own_types import ConfigDict

# full stack trace
os.environ['HYDRA_FULL_ERROR'] = '1'

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


@hydra.main(version_base=None, config_path="config", config_name="test_config")
def test(config: ConfigDict) -> None:
    try:
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(OmegaConf.to_yaml(config, resolve=True))
        initialize_config(config)  # put slurm stuff into config
        initialize_seed(config)  # set seed

        # # Update config so that recent features are incorporated
        # with open_dict(config):
        #     config.algorithm.data_modalities = ["pc", "mesh"]


        train_ds, eval_ds, train_dl, eval_dl = get_data(config.dataset, loading=config.loading.enable_loading)
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        loading_path = config.loading.root_path
        if loading_path.endswith("/"):
            loading_path = loading_path[:-1]
        output_path = config.loading.output_path
        os.makedirs(output_path, exist_ok=True)
        # last folder is exp_name
        exp_name = loading_path.split("/")[-1]
        allowed_job_types = config.loading.get("job_types", None)
        if not allowed_job_types:
            allowed_job_types = None
        allowed_seeds = config.loading.get("seeds", None)
        for job_type in os.listdir(loading_path):
            if allowed_job_types is not None and job_type not in allowed_job_types:
                continue
            job_type_path = os.path.join(loading_path, job_type)
            for seed in os.listdir(job_type_path):
                if allowed_seeds is not None and seed not in allowed_seeds:
                    continue
                seed_path = os.path.join(job_type_path, seed)
                checkpoint_path = os.path.join(seed_path, "checkpoints")
                if config.loading.checkpoint == "last":
                    checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
                elif config.loading.checkpoint == "best":
                    best_chkpt = get_best_checkpoint(checkpoint_path)
                    checkpoint_path = os.path.join(checkpoint_path, best_chkpt)
                algorithm = get_algorithm(config=config.algorithm, train_dl=train_dl, train_ds=train_ds,
                                          eval_ds=eval_ds, loading=True,
                                          checkpoint_path=checkpoint_path)

                # we do not want sdf predictions during test, so disable the sdf head
                # check for attribute
                if hasattr(algorithm, "sdf_head"):
                    algorithm.sdf_head = None

                # this is the correct parent directory to load the algorithm and the checkpoints
                test_logger = TestLogger(config.loading.output_path, exp_name, job_type, seed, meta_data={
                    "env_name": config.dataset.wandb_name,
                    "algorithm_type": algorithm.config.type,
                    "simulator_name": algorithm.config.simulator.name if "simulator" in algorithm.config else "None",
                })

                # add the test_opt config to the algorithm config if it exists
                if "test_opt_oracle" in config.algorithm:
                    with open_dict(algorithm.config):
                        algorithm.config.test_opt_oracle = config.algorithm.test_opt_oracle


                trainer = Trainer(
                    logger=test_logger,  # results will be written to disk
                    max_epochs=config.trainer.epochs,  # Max number of epochs for training
                    accelerator=config.trainer.accelerator,  # what type of accelerator to use
                    devices=config.trainer.devices,  # how many devices to use (if accelerator is not None)
                    precision=config.trainer.precision,  # Precision setting (e.g., 16-bit)
                    callbacks=None,
                    default_root_dir=exp_root,  # dummy path, will be used multiple times presumably
                    enable_checkpointing=False,  # no checkpointing
                    inference_mode=False,  # disable due to bug in AttentionPoolingLayer
                )

                # Now, start the test
                trainer.test(algorithm, dataloaders=eval_dl)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == '__main__':
    test()
