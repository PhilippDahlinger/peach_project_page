import copy
import os
from typing import Any

import hydra
import torch
from lightning import LightningModule
from torch import Tensor

from pc_mango.algorithm.util.get_optimizer import _get_optimizer, _get_scheduler
from pc_mango.dataset.util.graph_input_output_util import unpack_ml_batch
from pc_mango.dataset.util.util import get_mesh_connectivity, get_collider_connectivity
from pc_mango.network.mat_prop_head import get_mat_prop_head
from pc_mango.network.mesh_encoder import get_mesh_encoder
from pc_mango.network.meta_aggregation import get_meta_aggregation
from pc_mango.network.simulator import get_simulator
from pc_mango.util.own_types import ConfigDict


class MeshToSim(LightningModule):
    """
    Takes a trajectory of a mesh, predicts a material property for this, simulates with predicted material property and compares the simulated trajectory to the ground truth trajectory. The loss is a weighted sum of the material property prediction loss and the trajectory prediction loss.
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
        self.mat_prop_head = get_mat_prop_head(config.mat_prop_head, input_dim=self.config.encoder.output_dim,
                                               output_dim=example_batch["regression_features"].shape[-1])
        self.simulator = get_simulator(config.simulator, example_batch, self.config.encoder.output_dim)
        self.meta_aggregation = get_meta_aggregation(config.meta_aggregation)

        self.one_step_training = self.config.one_step_training

        # temp dicts to save eval results
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.best_val_mse = 1000000.0

        self.criterion = torch.nn.MSELoss()

        # save config as hyperparameter
        self.save_hyperparameters("config")

    def training_step(self, batch, batch_idx):
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=not self.one_step_training)  # shape (context_size, traj_length, num_points, point_dim)
        if self.one_step_training:
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: One-step training")
            # import os, psutil
            # process = psutil.Process(os.getpid())
            # print(f"RAM: {process.memory_info().rss / 1e9:.2f} GB")
            latent_pred = None # no material prediction
            pred_mat_prop = torch.zeros_like(batch["regression_features"])  # dummy tensor for loss calculation
        else:
            latent_pred = self.encoder(x, v, h)  # (context_size, latent_dim)
            # aggregate over context trajs
            latent_pred = self.meta_aggregation(latent_pred)  # shape (1, latent_dim)
            # predict material properties
            pred_mat_prop = self.mat_prop_head(latent_pred)  # shape (1, mat_prop_dim)
        # simulate with predicted material properties
        prediction = self.simulator(batch, latent_pred)  # shape (1, traj_length, num_points, point_dim) (for ml methods)
        # compute mat prop loss
        gth_mat_prop = batch["regression_features"]  # shape (1, mat_prop_dim)
        mat_prop_loss = torch.mean((pred_mat_prop - gth_mat_prop) ** 2)
        # compute trajectory loss
        if self.one_step_training:
            # one-step case, no target trajs, predict the next step and compare to gth next step
            ground_truth = batch["y"]  # shape (batch_size, num_points, point_dim)
            deformable_mask = batch["h"][0, :, 0] == 1
            ground_truth = ground_truth[:, deformable_mask, :]
        else:
            ground_truth = batch["x"][0]
            # only select target trajs
            ground_truth = ground_truth[batch["target_trajs"][0]]
            # only select deformable nodes
            if len(batch["h"].shape) == 5:
                deformable_mask = batch["h"][0, 0, 0, :, 0] == 1
            else:
                deformable_mask = batch["h"][0, 0, :, 0] == 1
            ground_truth = ground_truth[:, :, deformable_mask, :]
        ml_loss = self.criterion(prediction, ground_truth)
        if "edge_loss_type" in self.config:
            # this only works for bbv!!
            edge_indices = batch["edge_indices"][0]
            # edge_features = batch["edge_features"][0, 0]
            # mesh_edge_mask = edge_features[:, 0] == 1
            # mesh_edge_indices = edge_indices[:, mesh_edge_mask]
            if self.one_step_training:
                pred_edges = prediction[ :, edge_indices[1], :] - prediction[ :, edge_indices[0], :]
                gth_edges = ground_truth[ :, edge_indices[1], :] - ground_truth[ :, edge_indices[0], :]
            else:
                pred_edges = prediction[:, :, edge_indices[1], :] - prediction[:, :, edge_indices[0], :]
                gth_edges = ground_truth[:, :, edge_indices[1], :] - ground_truth[:, :, edge_indices[0], :]
            if self.config.edge_loss_type == "length":
                edge_loss = self.criterion(pred_edges.norm(dim=-1), gth_edges.norm(dim=-1))
            elif self.config.edge_loss_type == "vector":
                edge_loss = self.criterion(pred_edges, gth_edges)
            else:
                raise ValueError(f"Unknown edge loss type: {self.config.edge_loss_type}")
        else:
            edge_loss = None

        # combine losses
        loss = self.config.mat_prop_loss_weight * mat_prop_loss + (1 - self.config.mat_prop_loss_weight) * ml_loss
        if edge_loss is not None:
            loss += self.config.edge_loss_weight * edge_loss
        self.training_step_outputs.append({
            "loss": loss.detach(),
            "mat_prop_loss": mat_prop_loss.detach(),
            "ml_loss": ml_loss.detach()
        })
        if edge_loss is not None:
            self.training_step_outputs[-1]["edge_loss"] = edge_loss.detach()
        if batch_idx == 0 and not self.one_step_training:
            print(f"(TRAINING) Predicted material properties: {pred_mat_prop} GTH: {gth_mat_prop}")
        return loss

    def on_load_checkpoint(self, checkpoint):
        if self.config.simulator.name == "mgn":
            # only do this if mgn is there. it should crash for other sims since we have to fix it in that case
            # only in mgn the matprophead is not relevant
            matprophead_statedict = self.mat_prop_head.state_dict()
            for key in matprophead_statedict:
                ckpt_key = "mat_prop_head." + key
                if ckpt_key in checkpoint["state_dict"]:
                    # check that shapes match
                    if checkpoint["state_dict"][ckpt_key].shape != matprophead_statedict[key].shape:
                        print("Shape mismatch, generating dummy weights with proper shape for key ", ckpt_key)
                        checkpoint["state_dict"][ckpt_key] = torch.zeros_like(matprophead_statedict[key])
                else:
                    print("Key ", ckpt_key, " not found in checkpoint, generating dummy weights with proper shape")
                    checkpoint["state_dict"][ckpt_key] = torch.zeros_like(matprophead_statedict[key])

    def on_train_epoch_start(self):
        print("Training Epoch: ", self.current_epoch)

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from the `training_step` for each batch
        # Calculate the mean loss for the epoch
        epoch_average = {}
        for key in self.training_step_outputs[0].keys():
            epoch_average[key] = torch.stack([output[key] for output in self.training_step_outputs]).mean()

        # Log the mean epoch loss to WandB
        self.log_dict(epoch_average, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        gth_mat_prop, pred_mat_prop, result = self._evaluate_batch(batch, batch_idx)
        self.validation_step_outputs.append(result)
        print(f"(VALIDATION) Predicted material properties: {pred_mat_prop} GTH: {gth_mat_prop}")
        return result

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        current_val_mse = None
        epoch_average = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            val_loss = metric_values.mean()
            if metric_name == "ml_loss":
                current_val_mse = val_loss
            epoch_average[f"val_{metric_name}"] = val_loss
        self.log_dict(epoch_average, on_epoch=True, prog_bar=True)
        if current_val_mse < self.best_val_mse:
            self.best_val_mse = current_val_mse
            # save visualizations of current best model
            # TODO: implement visualization saving

        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        # evaluate different context sizes in test
        target_trajs = torch.tensor([self.config.evaluator.target_test_trajs])
        # only select unseen target trajs
        batch["target_trajs"] = target_trajs
        mat_prop_loss = []
        ml_loss = []
        for context_size in range(1, self.config.evaluator.max_test_context_size + 1):
            context_batch = copy.deepcopy(batch)
            context_trajs = torch.tensor([[i for i in range(context_size)]])
            context_batch["context_trajs"] = context_trajs
            context_batch_idx = -1 if context_size < self.config.evaluator.max_test_context_size else batch_idx  # no vis if not full context
            gth_mat_prop, pred_mat_prop, result = self._evaluate_batch(context_batch, context_batch_idx)
            mat_prop_loss.append(result["metrics"]["mat_prop_loss"])
            ml_loss.append(result["metrics"]["ml_loss"])
        # create tensor
        result = {
            "metrics": {
                "mat_prop_loss": torch.stack(mat_prop_loss),
                "ml_loss": torch.stack(ml_loss),
            },
            # add last visualizations
            "visualizations": result["visualizations"],
        }
        self.test_step_outputs.append(result)
        return result

    def _evaluate_batch(self, batch, batch_idx) -> tuple[dict[str, dict[str, Tensor | Any] | dict[Any, Any]], Any, Any]:
        x, v, h, h_description, edge_indices, edge_features, context_trajs, target_trajs, meta_data = unpack_ml_batch(batch,
                                                                                                           remove_batch_dim=True)  # shape (context_size, traj_length, num_points, point_dim)
        x = x[context_trajs]
        v = v[context_trajs]
        h = h[context_trajs]
        latent_pred = self.encoder(x, v, h)  # (context_size, latent_dim)
        # aggregate over context trajs
        latent_pred = self.meta_aggregation(latent_pred)  # shape (1, latent_dim)
        # predict material properties
        pred_mat_prop = self.mat_prop_head(latent_pred)  # shape (1, mat_prop_dim)
        # simulate with predicted material properties
        prediction = self.simulator(batch, latent_pred)  # shape (target_traj_size, traj_length, num_points, point_dim)
        # compute mat prop loss
        gth_mat_prop = batch["regression_features"]  # shape (1, mat_prop_dim)
        mat_prop_loss = torch.mean((pred_mat_prop - gth_mat_prop) ** 2)
        # compute trajectory loss
        ground_truth = batch["x"][0]
        # only select target trajs
        ground_truth = ground_truth[batch["target_trajs"][0]]
        # only select deformable nodes
        if len(batch["h"].shape) == 5:
            deformable_mask = batch["h"][0, 0, 0, :, 0] == 1
        else:
            deformable_mask = batch["h"][0, 0, :, 0] == 1
        ground_truth = ground_truth[:, :, deformable_mask, :]
        ml_loss = self.criterion(prediction, ground_truth)
        if "edge_loss_type" in self.config:
            # this only works for bbv!!
            edge_indices = batch["edge_indices"][0]
            # edge_features = batch["edge_features"][0, 0]
            # mesh_edge_mask = edge_features[:, 0] == 1
            # mesh_edge_indices = edge_indices[:, mesh_edge_mask]
            pred_edges = prediction[:, :, edge_indices[1], :] - prediction[:, :, edge_indices[0], :]
            gth_edges = ground_truth[:, :, edge_indices[1], :] - ground_truth[:, :, edge_indices[0], :]
            if self.config.edge_loss_type == "length":
                edge_loss = self.criterion(pred_edges.norm(dim=-1), gth_edges.norm(dim=-1))
            elif self.config.edge_loss_type == "vector":
                edge_loss = self.criterion(pred_edges, gth_edges)
            else:
                raise ValueError(f"Unknown edge loss type: {self.config.edge_loss_type}")
        else:
            edge_loss = None

        to_visualize = self.config.evaluator.task_ids_to_visualize
        vis_results = {}
        if batch_idx in to_visualize:
            meta_data = batch["meta_data"] if "meta_data" in batch else {}
            meta_data = {key: value[0] for key, value in meta_data.items()}
            # save all stuff for visualization
            # TODO: Right now only save the last traj in the batch, change that to save multiple trajs if needed
            vis_results["predicted_trajectory"] = prediction[-1].detach().cpu()
            vis_results["ground_truth_trajectory"] = ground_truth[-1].detach().cpu()
            # there are 2 cases: one where collider is predicted, and one where is not. Check h-description and decide upon that
            if ("collider_identifier",) in batch["h_description"]:
                # predicted collider
                # decide upon the collider identifier to get the predicted and the gth collider nodes
                collider_id_index = batch["h_description"].index(("collider_identifier",))
                collider_mask = batch["h"][0, 0, :, collider_id_index] == 1
                vis_results["predicted_collider"] = prediction[-1, :, collider_mask, :].detach().cpu()
                vis_results["ground_truth_collider"] = ground_truth[-1, :, collider_mask, :].detach().cpu()
                # also update the deformable trajectory: keep only the ones which are not collider nodes
                vis_results["predicted_trajectory"] = prediction[-1, :, ~collider_mask, :].detach().cpu()
                vis_results["ground_truth_trajectory"] = ground_truth[-1, :, ~collider_mask, :].detach().cpu()
                # save connectivity
                vis_results["collider_connectivity"] = get_collider_connectivity(self._train_ds, meta_data)
            else:
                # fixed collider, get from dataset if existent
                if len(batch["h"].shape) == 5:
                    collider_mask = batch["h"][0, 0, 0, :, 0] == 0
                else:
                    collider_mask = batch["h"][0, 0, :, 0] == 0
                collider = batch["x"][0][batch["target_trajs"][0]][:, :, collider_mask, :]
                # check for empty collider
                if collider.shape[2] == 0:
                    collider = None
                vis_results["collider"] = collider[-1].detach().cpu() if collider is not None else None
                collider_connectivity = get_collider_connectivity(self._train_ds, meta_data)
                if collider_connectivity is not None:
                    vis_results["collider_connectivity"] = collider_connectivity
            vis_results["mesh_connectivity"] = get_mesh_connectivity(self._train_ds, meta_data)

        metric_results = {
            "mat_prop_loss": mat_prop_loss.detach().cpu(),
            "ml_loss": ml_loss.detach().cpu(),
        }
        if edge_loss is not None:
            metric_results["edge_loss"] = edge_loss.detach().cpu()
        result = {"metrics": metric_results, "visualizations": vis_results}
        return gth_mat_prop, pred_mat_prop, result

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
                "simulator": self.simulator,
            }
        )
        scheduler = _get_scheduler(config=self.config.scheduler, optimizer=optimizer)
        # TODO: update monitor value, val_mse is not universal anymore
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": self.config.check_val_every_n_epoch,
            "monitor": "val_mse",
            "strict": False,
        }
        return [optimizer], [scheduler]
