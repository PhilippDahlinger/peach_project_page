import os

import hydra
import torch
from lightning import LightningModule

from pc_mango.dataset.util.graph_input_output_util import unpack_ml_batch
from pc_mango.algorithm.util.get_optimizer import _get_optimizer, _get_scheduler
from pc_mango.network.mat_prop_head import get_mat_prop_head
from pc_mango.network.mesh_encoder import get_mesh_encoder
from pc_mango.network.meta_aggregation import get_meta_aggregation
from pc_mango.util.own_types import ConfigDict


class MeshToMatProp(LightningModule):
    """
    Takes a trajectory of a mesh, predicts a material property for this. No simulation involved.
    """

    def __init__(self, config: ConfigDict, train_dl, train_ds: torch.utils.data.Dataset,
                 eval_ds: torch.utils.data.Dataset):
        super().__init__()
        self.config: ConfigDict = config
        self._train_ds = train_ds
        self._eval_ds = eval_ds
        self._vis_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "visualizations")

        # load networks
        example_batch = None
        for example_batch in train_dl:
            break
        self.encoder = get_mesh_encoder(config.encoder, example_batch)
        self.mat_prop_head = get_mat_prop_head(config.mat_prop_head, input_dim=self.config.encoder.latent_dimension,
                                               output_dim=example_batch["regression_features"].shape[-1])

        self.meta_aggregation = get_meta_aggregation(config.meta_aggregation)

        # temp dicts to save eval results
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.best_val_mse = 1000000.0

        # save config as hyperparameter
        self.save_hyperparameters("config")

    def training_step(self, batch, batch_idx):
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)  # shape (context_size, traj_length, num_points, point_dim)
        latent_pred = self.encoder(x, v, h)  # (context_size, latent_dim)
        # aggregate over context trajs
        latent_pred = self.meta_aggregation(latent_pred) # shape (1, latent_dim)
        # predict material properties
        pred_mat_prop = self.mat_prop_head(latent_pred)  # shape (1, mat_prop_dim)
        # compute mse
        gth_mat_prop = batch["regression_features"]  # shape (1, mat_prop_dim)
        loss = torch.mean((pred_mat_prop - gth_mat_prop) ** 2)
        loss = self.config.mat_prop_loss_weight * loss
        self.training_step_outputs.append(loss.detach())

        if batch_idx == 0:
            print(f"(TRAINING) Predicted material properties: {pred_mat_prop} GTH: {gth_mat_prop}")
        return loss

    def on_train_epoch_start(self):
        print("Training Epoch: ", self.current_epoch)

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from the `training_step` for each batch
        # Calculate the mean loss for the epoch
        epoch_average = torch.stack(self.training_step_outputs).mean()
        # Log the mean epoch loss to WandB
        self.log('train_loss', epoch_average, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)  # shape (context_size, traj_length, num_points, point_dim)
        latent_pred = self.encoder(x, v, h)  # (context_size, latent_dim)
        # aggregate over context trajs
        latent_pred = self.meta_aggregation(latent_pred) # shape (1, latent_dim)
        # predict material properties
        pred_mat_prop = self.mat_prop_head(latent_pred)  # shape (1, mat_prop_dim)
        # compute mse
        gth_mat_prop = batch["regression_features"]  # shape (1, mat_prop_dim)
        mse = torch.mean((pred_mat_prop - gth_mat_prop) ** 2)
        metric_results = {"mse": mse}
        vis_results = {}
        result = {"metrics": metric_results, "visualizations": vis_results}
        self.validation_step_outputs.append(result)
        print(f"(VALIDATION) Predicted material properties: {pred_mat_prop} GTH: {gth_mat_prop}")
        return result

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        current_val_mse = None
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            val_loss = metric_values.mean()
            if metric_name == "mse":
                current_val_mse = val_loss
            # if self.logger is not None:
            self.log(f"val_{metric_name}", val_loss, on_epoch=True, prog_bar=True)
        if current_val_mse < self.best_val_mse:
            self.best_val_mse = current_val_mse
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        mat_prop_loss = []
        for context_size in range(1, self.config.evaluator.max_test_context_size + 1):
            context_trajs = torch.tensor([[i for i in range(context_size)]])
            batch["context_trajs"] = context_trajs
            # ignore the batch size of 1 in validation
            x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(batch,
                                                                                                               remove_batch_dim=True)  # shape (context_size, traj_length, num_points, point_dim)
            x = x[context_trajs]
            v = v[context_trajs]
            h = h[context_trajs]
            latent_pred = self.encoder(x, v, h)  # (context_size, latent_dim)
            # aggregate over context trajs
            latent_pred = self.meta_aggregation(latent_pred) # shape (1, latent_dim)
            # predict material properties
            pred_mat_prop = self.mat_prop_head(latent_pred)  # shape (1, mat_prop_dim)
            # compute mse
            gth_mat_prop = batch["regression_features"]  # shape (1, mat_prop_dim)
            mse = torch.mean((pred_mat_prop - gth_mat_prop) ** 2)
            mat_prop_loss.append(mse)
        metric_results = {"mat_prop_mse": torch.stack(mat_prop_loss)}  # shape (8,)
        vis_results = {}
        result = {"metrics": metric_results, "visualizations": vis_results}
        self.test_step_outputs.append(result)
        return result

    def on_test_end(self) -> None:
        outputs = self.test_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        epoch_average = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            test_loss = metric_values.mean(dim=0)
            epoch_average[f"test_{metric_name}"] = test_loss
        visualizations = [output["visualizations"] for output in outputs]
        to_log = {
            "metrics": epoch_average,
            "config": self.config,
            "visualizations": visualizations,
        }
        self.logger.log_metrics(to_log, self.current_epoch)
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = _get_optimizer(
            config=self.config.optimizer,
            models={
                "encoder": self.encoder,
                "mat_prop_head": self.mat_prop_head,
                "meta_aggregation": self.meta_aggregation,
            }
        )
        scheduler = _get_scheduler(config=self.config.scheduler, optimizer=optimizer)
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": self.config.check_val_every_n_epoch,
            "monitor": "val_mse",
            "strict": False,
        }
        return [optimizer], [scheduler]
