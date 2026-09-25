import copy
import os
from abc import ABC, abstractmethod
from typing import Any

import hydra
import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn.functional import smooth_l1_loss

from pc_mango.algorithm.util.get_optimizer import _get_optimizer, _get_scheduler
from pc_mango.dataset.ml_datasets.bbv import BBVDataset
from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset
from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset
from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
from pc_mango.dataset.util.util import get_mesh_connectivity, get_collider_connectivity
from pc_mango.network.mat_prop_head import get_mat_prop_head
from pc_mango.network.meta_aggregation import get_meta_aggregation
from pc_mango.network.pc_encoder import get_pc_encoder
from pc_mango.network.sdf_head import get_sdf_head
from pc_mango.network.simulator import get_simulator
from pc_mango.util.own_types import ConfigDict
from pc_mango.util.trampoline_real_world_eval.point_to_mesh_loss import compute_point_to_mesh_loss, \
    compute_point_to_mesh_loss2
from pc_mango.visualization.debug_pc_plot import visualize_pointcloud_sequence
from pc_mango.visualization.save_trajectory import export_deforming_mesh_to_xdmf
from scripts.preprocess.create_ood_dataset_split import visualize


class PcToSim(LightningModule):
    """
    Takes a stream of pointclouds, predicts material properties, simulates with predicted material properties and computes loss on predicted trajectory. The main model of the paper.
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
        self.encoder = get_pc_encoder(config.encoder, example_batch)
        self.mat_prop_head = get_mat_prop_head(config.mat_prop_head, input_dim=self.config.encoder.output_dim,
                                               output_dim=example_batch["regression_features"].shape[-1])
        if "sdf_head" in config and config.sdf_head is not None:
            self.sdf_head = get_sdf_head(config.sdf_head, example_batch)
        else:
            self.sdf_head = None
        self.simulator = get_simulator(config.simulator, example_batch, self.config.encoder.output_dim)
        self.meta_aggregation = get_meta_aggregation(config.meta_aggregation)

        # temp dicts to save eval results
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.best_val_mse = 1000000.0

        self.criterion = torch.nn.MSELoss()

        # save config as hyperparameter
        self.save_hyperparameters("config")

    def on_fit_start(self):
        for module in self.modules():
            if hasattr(module, "freeze_tokenizer") and module.freeze_tokenizer:
                module.tokenizer.eval()

    def training_step(self, batch, batch_idx):
        # get predictions
        prediction, mat_prop_prediction, sdf_prediction = self.predict(batch)

        # compute all three losses via helper
        ml_loss, mat_prop_loss, sdf_loss, ground_truth, edge_loss = self.compute_loss(batch, prediction, mat_prop_prediction, sdf_prediction)

        # combine losses
        loss = ml_loss + self.config.mat_prop_loss_weight * mat_prop_loss + self.config.sdf_loss_weight * sdf_loss
        if edge_loss is not None:
            loss += self.config.edge_loss_weight * edge_loss
        self.training_step_outputs.append({
            "loss": loss.detach(),
            "mat_prop_loss": mat_prop_loss.detach(),
            "sdf_loss": sdf_loss.detach(),
            "ml_loss": ml_loss.detach()
        })
        if edge_loss is not None:
            self.training_step_outputs[-1]["edge_loss"] = edge_loss.detach()
        # small debug print: keep the ground truth mat props for readability (if present)
        if batch_idx == 0 and mat_prop_prediction is not None:
            gth_mat_prop = batch["regression_features"]
            print(f"(TRAINING) Predicted material properties: {mat_prop_prediction} GTH: {gth_mat_prop}")
        return loss

    def compute_loss(self, batch, prediction, pred_mat_prop=None, sdf_prediction=None):
        """
        Compute ml_loss, mat_prop_loss, sdf_loss.
        - Moves ground-truth tensors to the device of `prediction`.
        - Uses self._last_gth_sdf (set by predict) for sdf loss if sdf_prediction is provided.
        Returns: (ml_loss, mat_prop_loss, sdf_loss) — torch scalars.
        """
        device = prediction.device if isinstance(prediction, torch.Tensor) else next(self.parameters()).device

        # trajectory loss (ml_loss)
        ground_truth = batch["x"][0].to(device)
        # only select target trajs
        ground_truth = ground_truth[batch["target_trajs"][0]]
        # only select deformable nodes
        if len(batch["h"].shape) == 5:
            deformable_mask = batch["h"][0, 0, 0, :, 0] == 1
        else:
            deformable_mask = batch["h"][0, 0, :, 0] == 1
        ground_truth = ground_truth[:, :, deformable_mask, :].to(device)
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

        # material property loss
        if pred_mat_prop is not None:
            gth_mat_prop = batch["regression_features"].to(pred_mat_prop.device)
            mat_prop_loss = torch.mean((pred_mat_prop - gth_mat_prop) ** 2)
        else:
            mat_prop_loss = torch.tensor(0.0, device=device)

        # sdf loss
        if sdf_prediction is not None:
            # self._last_gth_sdf was set in predict when sdf was prepared
            gth_sdf = self._last_gth_sdf.to(sdf_prediction.device)
            sdf_loss = smooth_l1_loss(sdf_prediction, gth_sdf)
        else:
            sdf_loss = torch.tensor(0.0, device=device)

        return ml_loss, mat_prop_loss, sdf_loss, ground_truth, edge_loss

    def predict(self, batch):
        """
        Encodes the input batch, optionally predicts sdf, aggregates latent, predicts material properties
        and runs the simulator. If sdf is prepared, store the prepared gth sdf in self._last_gth_sdf to avoid
        duplicating prepare_sdf logic elsewhere.

        Returns: (prediction, pred_mat_prop, sdf_prediction)
        """
        pc = batch["pc"][0]  # shape (context_size, traj_length, num_points, point_dim)
        color = batch["pc_color"][0]  # shape (context_size, traj_length, num_points, color_dim)
        encoder_results = self.encoder(pc, color)  # (context_size, latent_dim)
        if isinstance(encoder_results, tuple):
            latent_pred, tokens, center_scaled_spacetime = encoder_results
        else:
            latent_pred = encoder_results
            tokens, center_scaled_spacetime = None, None

        # sdfs (optional)
        sdf_prediction = None
        self._last_gth_sdf = None
        if tokens is not None and self.config.get("predict_sdf", False) and self.sdf_head is not None:
            queries = batch["queries"][0]
            gth_sdf = batch["sdf"][0]
            queries_scaled_spacetime, prepared_gth_sdf = self.prepare_sdf(queries, gth_sdf)
            sdf_prediction = self.sdf_head(queries_scaled_spacetime, tokens, center_scaled_spacetime)
            # store prepared ground-truth sdf for later loss computation (avoid duplicated prepare_sdf)
            self._last_gth_sdf = prepared_gth_sdf

        # aggregate and predict material properties
        latent_pred = self.meta_aggregation(latent_pred)  # shape (1, latent_dim)
        pred_mat_prop = self.mat_prop_head(latent_pred) if self.config.get("predict_mat_prop", False) else None

        if self.config.get("test_opt_oracle", False):
            batch = self.test_opt_oracle(batch, latent_pred)

        # simulate with predicted material properties
        prediction = self.simulator(batch, latent_pred)
        return prediction, pred_mat_prop, sdf_prediction

    def prepare_sdf(self, queries, sdf) -> Any:
         B, _, _, _, queries_scaled_spacetime, spacetime = self.encoder.compute_spacetime(queries)
         C = queries_scaled_spacetime.size(0) // B
         queries_scaled_spacetime = queries_scaled_spacetime.unflatten(0, (B, C))
         sdf = sdf.flatten(1, 2)  # (B, T*N, sdf_dim)
         return queries_scaled_spacetime, sdf

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
        sdf_loss = []
        for context_size in range(1, self.config.evaluator.max_test_context_size + 1):
            context_batch = copy.deepcopy(batch)
            context_trajs = torch.tensor([[i for i in range(context_size)]])
            context_batch["context_trajs"] = context_trajs
            context_batch["pc"] = context_batch["pc"][:, context_trajs[0], :, :, :]
            context_batch["pc_color"] = context_batch["pc_color"][:, context_trajs[0], :, :, :]
            context_batch_idx = -1 if context_size < self.config.evaluator.max_test_context_size else batch_idx  # no vis if not full context
            gth_mat_prop, pred_mat_prop, result = self._evaluate_batch(context_batch, context_batch_idx)
            mat_prop_loss.append(result["metrics"]["mat_prop_loss"])
            ml_loss.append(result["metrics"]["ml_loss"])
            sdf_loss.append(result["metrics"]["sdf_loss"])
        # create tensor
        result = {
            "metrics": {
                "mat_prop_loss": torch.stack(mat_prop_loss),
                "ml_loss": torch.stack(ml_loss),
                "sdf_loss": torch.stack(sdf_loss),
            },
            # add last visualizations with biggest context
            "visualizations": result["visualizations"],
            "material_properties": result["material_properties"],
            "mat_prop_description": result.get("mat_prop_description", None),
        }
        self.test_step_outputs.append(result)
        return result

    def test_opt_oracle(self, batch, latent_pred):
        from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset
        if isinstance(self._eval_ds, DeformableBlockDataset):
            dataset = "db_v4"
        elif isinstance(self._eval_ds, SheetDeformationDataset):
            dataset = "sd_v1"
        elif isinstance(self._eval_ds, TrampolineDataset):
            dataset = "trampoline_v4"
        elif isinstance(self._eval_ds, BBVDataset):
            dataset = "bbv_v3"
        else:
            raise NotImplementedError("Dataset type not supported for test_opt_oracle")
        mat_prop_dim = batch["regression_features"].shape[-1]
        PC_SUBSAMPLE = None
        if dataset == "db_v4":
            mat_params = torch.nn.Parameter(
                torch.tensor([-0.25113255], device=self.device)
            )
        elif dataset == "sd_v1":
            mat_params = torch.nn.Parameter(
                torch.tensor([-1.5595], device=self.device)
            )
        elif dataset == "bbv_v3":
            mat_params = torch.nn.Parameter(
                torch.tensor([-1.7691, -1.8543,  0.4353], device=self.device)
            )
        elif dataset == "trampoline_v4":
            mat_params = torch.nn.Parameter(
                torch.tensor([ 0.7935, -0.2937, -1.7150, -2.1124, -1.1252], device=self.device)
            )
            PC_SUBSAMPLE = None  # number of point cloud points to keep  (None = all)
            # TIME_SUBSAMPLE = 10  # use every Nth timestep during opt      (1 = all)
        else:
            raise NotImplementedError

        if dataset == "db_v4":
            lr = 0.03
            num_steps = 50
        elif dataset == "sd_v1":
            lr = 0.015
            num_steps = 200
        elif dataset == "trampoline_v4":
            lr = 0.06
            num_steps = 300
        elif dataset == "bbv_v3":
            lr = 0.015
            num_steps = 200
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam([mat_params], lr=lr)

        if dataset == "db_v4":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        elif dataset == "sd_v1":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
        elif dataset == "trampoline_v4":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
        elif dataset == "bbv_v3":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
        else:
            raise NotImplementedError

        best_loss = float('inf')
        best_params = mat_params.detach().clone()

        original_batch = copy.deepcopy(batch)

        try:

            # set target_trajs as context_trajs
            batch["target_trajs"] = batch["context_trajs"]
            gth_pointcloud = batch["pc"][0]

            # get faces for the trajectory and the collider
            meta_data = batch["meta_data"] if "meta_data" in batch else {}
            if "diameter" in meta_data:
                col_meta_data = {"diameter":  meta_data["diameter"][0]}
            else:
                col_meta_data = meta_data
            faces_trajectory = get_mesh_connectivity(self._train_ds, meta_data)
            if dataset == "bbv_v3":
                faces_trajectory = faces_trajectory[0]
            faces_collider = get_collider_connectivity(self._train_ds, col_meta_data)
            if faces_collider is None:
                faces_collider = torch.empty((0, 3), dtype=torch.long)
            else:
                faces_collider = torch.tensor(faces_collider)

            # remove h last columns (or set to zero rather)
            if len(batch["h"].shape) == 5:
                batch["h"][:, :, :, :, -mat_prop_dim:] = 0
            else:
                batch["h"][:, :, :, -mat_prop_dim:] = 0
            if dataset == "sd_v1":
                batch["regression_features"] = batch["regression_features"].view(-1, mat_prop_dim)
            print("------------------------------------------------")
            for step in range(num_steps):
                optimizer.zero_grad()

                with torch.enable_grad():  # re-enable gradients inside no_grad context
                    if dataset == "db_v4":
                        # set the current mat_params in the batch for simulator
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            torch.tensor(0.5).to("cuda").expand_as(batch["h"][..., 0:1]),
                            mat_params.expand_as(batch["h"][..., 0:1])
                        ], dim=-1)
                    elif dataset == "sd_v1":
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            mat_params.expand_as(batch["h"][..., 0:1])
                        ], dim=-1)
                    elif dataset == "trampoline_v4":
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            mat_params.expand_as(batch["h"][..., 0:mat_prop_dim])
                        ], dim=-1)
                    elif dataset == "bbv_v3":
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            mat_params.expand_as(batch["h"][..., 0:mat_prop_dim])
                        ], dim=-1)
                    else:
                        raise NotImplementedError
                    prediction = self.simulator(batch, latent_pred)
                    if dataset in ["db_v4", "sd_v1", "bbv_v3"]:
                        # for now pointclouds always have stride 2. if that is not the case: change this and make it parametric
                        prediction = prediction[:, ::2, :, :]
                    # add z dim for loss
                    if prediction.shape[-1] == 2:
                        prediction = torch.cat([prediction, torch.zeros_like(prediction[:, :, :, :1])], dim=-1)
                    assert prediction.shape[1] == gth_pointcloud.shape[1], "Prediction and ground truth pointcloud have different number of time steps, adjust stride or check data"
                    # check for each context traj separately and average the loss over them
                    # TODO: does not work for BBV currently
                    if dataset != "bbv_v3":
                        collider = batch["x"][0][batch["target_trajs"][0]][:, ::2, batch["h"][0, 0, :, 0] == 0, :]
                    else:
                        # there is no collider
                        collider = torch.empty((prediction.shape[0], prediction.shape[1], 0, 3), device=prediction.device)
                    # add z dim
                    if collider.shape[-1] == 2:
                        collider = torch.cat([collider, torch.zeros_like(collider[:, :, :, :1])], dim=-1)
                    total_loss = torch.tensor(0.0).to(prediction.device)
                    for pc_traj, prediction_traj, collider_traj in zip(gth_pointcloud, prediction, collider):
                        # debug plot
                        # visualize_pointcloud_sequence(pc_traj, types=torch.zeros_like(pc_traj)[..., 0:1])
                        # visualize_pointcloud_sequence(prediction_traj.detach(), types=torch.zeros_like(prediction_traj.detach())[..., 0:1])
                        # loss = compute_point_to_mesh_loss(pc_traj.to("cuda"), prediction_traj.to("cuda"), faces_trajectory.to("cuda"),
                        #                            collider_traj.to("cuda"), faces_collider.to("cuda"), verbose=False).sum()
                        if dataset == "trampoline_v4":
                            if PC_SUBSAMPLE is not None and pc_traj.shape[1] > PC_SUBSAMPLE:
                                perm = torch.randperm(pc_traj.shape[1], device=pc_traj.device)[:PC_SUBSAMPLE]
                                pc_traj = pc_traj[:, perm]  # [T, PC_SUBSAMPLE, 3]
                            mesh_traj = prediction_traj[:, batch["h"][0, 0, :, 2] == 1, :]
                            collider_traj = prediction_traj[:, batch["h"][0, 0, :, 3] == 1, :]
                            loss = compute_point_to_mesh_loss2(pc_traj[:, ...].to("cuda"),
                                                               mesh_traj[:, ...].to("cuda"),
                                                               faces_trajectory.to("cuda"),
                                                               collider_traj[:, ...].to("cuda"), faces_collider.to("cuda"),
                                                               verbose=False).sum()
                        else:
                            loss = compute_point_to_mesh_loss2(pc_traj[:, ...].to("cuda"), prediction_traj[:,...].to("cuda"), faces_trajectory.to("cuda"),
                                                       collider_traj[:,...].to("cuda"), faces_collider.to("cuda"), compiled=False, verbose=False).sum()
                        total_loss += loss
                    # correct magnitude
                    total_loss *= 10000
                    total_loss.backward()
                    print("total_loss: ", total_loss.item())
                optimizer.step()
                scheduler.step()

                print("current mat params: ", mat_params.detach().cpu().numpy(), "gth: ", batch["regression_features"][0, :mat_prop_dim].cpu().numpy())
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_params = mat_params.detach().clone()
        except torch.OutOfMemoryError:
            print("OOM during test-time optimization, returning best found parameters so far.")
        # insert best params into batch for final evaluation
        if dataset == "db_v4":
            batch["h"] = torch.cat([
                original_batch["h"][..., :-mat_prop_dim],
                torch.tensor(0.5).to("cuda").expand_as(original_batch["h"][..., 0:1]),
                best_params.expand_as(original_batch["h"][..., 0:1])
            ], dim=-1)
        elif dataset == "sd_v1":
            batch["h"] = torch.cat([
                original_batch["h"][..., :-mat_prop_dim],
                best_params.expand_as(original_batch["h"][..., 0:1])
            ], dim=-1)
        elif dataset in ["trampoline_v4", "bbv_v3"]:
            batch["h"] = torch.cat([
                original_batch["h"][..., :-mat_prop_dim],
                best_params.expand_as(original_batch["h"][..., 0:mat_prop_dim])
            ], dim=-1)
        else:
            raise NotImplementedError
        return batch

    def _evaluate_batch(self, batch, batch_idx) -> tuple[dict[str, dict[str, Tensor | Any] | dict[Any, Any]], Any, Any]:
        # Use the unified predict function to get predictions and optional sdf prediction.
        prediction, pred_mat_prop, sdf_prediction = self.predict(batch)
        # compute losses via helper
        ml_loss, mat_prop_loss, sdf_loss, ground_truth, edge_loss = self.compute_loss(batch, prediction, pred_mat_prop, sdf_prediction)

        # ground truth mat props (for return / visualization)
        gth_mat_prop = batch["regression_features"]

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
            if ("collider_identifier", ) in batch["h_description"]:
                # predicted collider
                # decide upon the collider identifier to get the predicted and the gth collider nodes
                collider_id_index = batch["h_description"].index(("collider_identifier", ))
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
            "sdf_loss": sdf_loss.detach().cpu()
        }
        if edge_loss is not None:
            metric_results["edge_loss"] = edge_loss.detach().cpu()
        try:
            mat_prop_description = self._eval_ds.mat_prop_description
        except AttributeError:
            mat_prop_description = None
        result = {"metrics": metric_results, "visualizations": vis_results, "material_properties": gth_mat_prop.detach().cpu(),
                  "mat_prop_description": mat_prop_description}
        return gth_mat_prop, pred_mat_prop, result

    def on_test_end(self) -> None:
        outputs = self.test_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        epoch_average = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, context_size)
            test_loss = metric_values.mean(dim=0)  # average over batches, shape (context_size)
            epoch_average[f"test_{metric_name}"] = test_loss
        visualizations = [output["visualizations"] for output in outputs]
        mat_properties = [output["material_properties"] for output in outputs]
        mat_prop_description = outputs[0].get("mat_prop_description", None)
        to_log = {
            "metrics": epoch_average,
            "config": self.config,
            "visualizations": visualizations,
            "material_properties": mat_properties,
            "mat_prop_description": mat_prop_description,
        }
        self.logger.log_metrics(to_log, self.current_epoch)
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = _get_optimizer(
            config=self.config.optimizer,
            models={
                "encoder": self.encoder,
                "mat_prop_head": self.mat_prop_head,
                "sdf_head": self.sdf_head,
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
