import os
import hydra
import torch
import traceback
import sys

from pc_mango.util.hydra_initialization import load_omega_conf_resolvers
from pc_mango.util.own_types import ConfigDict

# full stack trace
os.environ['HYDRA_FULL_ERROR'] = '1'

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def train(config: ConfigDict) -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
        print(f"Device properties: {torch.cuda.get_device_properties(0)}")

    test_gpu_tensor = torch.tensor([1.0], device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test GPU tensor: {test_gpu_tensor}, device: {test_gpu_tensor.device}")
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from pc_mango.util.initialization import main_initialization
    from omegaconf import OmegaConf

    try:
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        print(OmegaConf.to_yaml(config, resolve=True))
        train_ds, eval_ds, train_dl, eval_dl, algorithm, wandb_logger = main_initialization(config)
        print("Initialized everything.")
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        monitor = config.algorithm.checkpoint_metric  # e.g. "val_mse"

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=os.path.join(exp_root, "checkpoints"),
            filename=f"best-checkpoint-{{epoch:02d}}-{{{monitor}:.8f}}",
            save_top_k=3,
            mode="min",
            save_last=True
        )
        learning_rate_monitor = LearningRateMonitor(logging_interval='epoch')

        callbacks = []
        if config.trainer.checkpointing:
            callbacks.append(checkpoint_callback)
        if wandb_logger:
            callbacks.append(learning_rate_monitor)
        if len(callbacks) == 0:
            callbacks = None


        trainer = Trainer(
            num_sanity_val_steps=config.trainer.num_sanity_val_steps,
            logger=wandb_logger,  # Use the wandb logger
            max_epochs=config.trainer.epochs,  # Max number of epochs for training
            accelerator=config.trainer.accelerator,  # what type of accelerator to use
            devices=config.trainer.devices,  # how many devices to use (if accelerator is not None)
            precision=config.trainer.precision,  # Precision setting (e.g., 16-bit)
            callbacks=callbacks,  # Checkpointing callback
            accumulate_grad_batches=config.trainer.accumulate_grad_batches,  # Gradient accumulation
            check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,  # How often to validate (in epochs)
            default_root_dir=exp_root,  # Where to save logs and checkpoints
            enable_checkpointing=config.trainer.checkpointing,  # Enable checkpointing
            enable_progress_bar=config.trainer.enable_progress_bar,  # Enable progress bar
        )

        # Now, start the training
        if config.get("resume_from_checkpoint", False):
            try:
                assert config.old_exp_folder is not None, "old_exp_folder must be specified when resume_from_checkpoint is True"
                checkpoint_path = os.path.join(config.old_exp_folder, f"seed_{config.seed}", "checkpoints", "last.ckpt")
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                trainer.fit(algorithm, train_dataloaders=train_dl, val_dataloaders=eval_dl, ckpt_path=checkpoint_path)
            except Exception as e:
                print(f"Error while resuming from checkpoint: {e}")
                print("Starting training from scratch.")
                trainer.fit(algorithm, train_dataloaders=train_dl, val_dataloaders=eval_dl)
        else:
            # start fresh
            trainer.fit(algorithm, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == '__main__':
    train()
