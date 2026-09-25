import os
import warnings
import random

import numpy as np
import torch
from omegaconf import open_dict

from pc_mango.dataset.step_datasets.abstract_step_ggns_dataset import worker_init_fn


def main_initialization(config):
    """
    Initializes the config, the seed, the device and the env, algorithm, evaluator and recorder.
    :param config:
    :param env: If None, the env will be initialized. Otherwise, the env will be used.
    :return:
    """
    from pc_mango.algorithm import get_algorithm


    initialize_config(config)  # put slurm stuff into config
    initialize_seed(config)  # set seed
    train_ds, eval_ds, train_dl, eval_dl = get_data(config.dataset, loading=config.loading.enable_loading)

    if config.loading.enable_loading:
        warnings.warn("Loading is enabled. Use test.py directly. This is deprecated. Make sure that the loading is set up correctly.")
        algorithm = get_algorithm(config=config.algorithm, train_dl=train_dl, train_ds=train_ds, eval_ds=eval_ds, loading=True,
                                  checkpoint_path="")
    else:
        algorithm = get_algorithm(config=config.algorithm, train_dl=train_dl, train_ds=train_ds, eval_ds=eval_ds)
    if config.wandb.enabled:
        # Initialize WandB logger
        from pc_mango.logger import get_wandb_logger
        wandb_logger = get_wandb_logger(config=config, algorithm=algorithm)
    else:
        wandb_logger = False
    return train_ds, eval_ds, train_dl, eval_dl, algorithm, wandb_logger


def get_data(config, loading=False):
    from torch_geometric.loader import DataLoader
    from pc_mango.dataset import get_dataset

    train_ds, eval_ds = get_dataset(config, loading=loading)
    # check for a cache path in the train ds
    if hasattr(train_ds, "cache_path") and train_ds.cache_path is not None:
        train_dl = DataLoader(train_ds, batch_size=config.train_dataset.batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    else:
        train_dl = DataLoader(train_ds, batch_size=config.train_dataset.batch_size, shuffle=True, num_workers=2)
    if hasattr(eval_ds, "cache_path") and eval_ds.cache_path is not None:
        eval_dl = DataLoader(eval_ds, batch_size=config.eval_dataset.get("batch_size", 1), shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    else:
        eval_dl = DataLoader(eval_ds, batch_size=config.eval_dataset.get("batch_size", 1), shuffle=False, num_workers=2)
    return train_ds, eval_ds, train_dl, eval_dl


def initialize_config(config):
    try:
        with open_dict(config):
            config.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
            config.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    except KeyError:
        pass


def initialize_seed(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    from pytorch_lightning import seed_everything
    seed_everything(config.seed)


