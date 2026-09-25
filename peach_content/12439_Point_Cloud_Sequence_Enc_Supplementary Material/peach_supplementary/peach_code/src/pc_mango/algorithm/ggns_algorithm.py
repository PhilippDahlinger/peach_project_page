import copy
import json

import torch
from lightning import LightningModule
from omegaconf import OmegaConf
from torch import Tensor
from torch_geometric.data import Batch, Data

from pc_mango.algorithm.util.get_optimizer import _get_optimizer, _get_scheduler
from pc_mango.dataset.edges.edge_indices import get_edge_indices
from pc_mango.ggns.modules.gnn_modules.get_gnn_model import get_gnn_model
from pc_mango.ggns.src.utils.data_utils import (
    get_feature_info_from_data,
    transform_position_to_edges,
    convert_to_hetero_data,
    predict_velocity,
)
from pc_mango.ggns.src.utils.eval_utils import get_radius_dict_for_evaluation_mode
from pc_mango.ggns.src.utils.get_connectivity_setting import get_connectivity_setting
from pc_mango.ggns.src.utils.graph_utils import create_graph_from_raw
from pc_mango.ggns.src.utils.train_utils import (
    get_network_config,
    add_noise_to_mesh_nodes,
    add_noise_to_pcd_points,
    add_pointcloud_dropout,
)


class GGNSAlgorithm(LightningModule):
    """
    Lightning wrapper for GGNS one-step training.
    Faithfully reproduces the training loop from models/ggns/src/algorithms/standard_train.py.
    """

    def __init__(self, config, train_dl, train_ds, eval_ds):
        super().__init__()
        self.config = config
        self._train_ds = train_ds
        self._eval_ds = eval_ds

        # Extract config params (matching hdf5_plate_lightweight.yaml)
        self.hetero = config.get("hetero", False)
        self.mgn_hetero = config.get("mgn_hetero", False)
        self.use_color = config.get("use_colors", False)
        self.use_mesh_coordinates = config.get("use_mesh_coordinates", True)
        self.use_world_edges = "world" in config.connectivity_setting or self.mgn_hetero
        self.input_mesh_noise = config.input_mesh_noise
        self.input_pcd_noise = config.input_pcd_noise
        self.pointcloud_dropout = config.pointcloud_dropout
        self.loss_normalizer = config.get("loss_normalizer", 1)

        # Get connectivity settings (tissue_task always False — not used in this repo)
        _, _, self.euclidian_distance, _ = get_connectivity_setting(config.connectivity_setting)
        self.is_3d = config.get("is_3d", False)

        # Build network config (needs plain dict, not OmegaConf)
        config_dict = OmegaConf.to_container(config, resolve=True)
        network_config = get_network_config(config_dict, self.hetero, self.use_world_edges)

        # Inspect a training sample to get feature dimensions
        example_data = copy.deepcopy(train_ds[0])
        example_data = transform_position_to_edges(example_data, self.euclidian_distance)
        in_node_features, in_edge_features, _, _ = get_feature_info_from_data(
            example_data, "cpu", self.hetero, self.use_color,
            self.is_3d, self.use_world_edges, self.use_mesh_coordinates, self.mgn_hetero,
        )
        out_node_features = example_data.y.shape[1]  # 2 for 2D
        in_global_features = 1

        # Build GNN model (Lightning handles device placement)
        self.gnn = get_gnn_model(
            in_node_features, in_edge_features, in_global_features, out_node_features,
            network_config, self.hetero, "cpu",
        )

        self.criterion = torch.nn.MSELoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters("config")

    def training_step(self, batch, batch_idx):
        data = batch
        # col_mask = batch.node_type == 1
        # if batch.pos[col_mask][:, 2].max() < 0.8:
        #     plot_pyg_graph(data[0].to("cpu"))
        #     plt.show()
        # else:
        #     print("too high: ", batch.pos[col_mask][:, 2].max())
        # plot_pyg_graph(data[0].to("cpu"), plot_pc=False)
        # plt.show()

        data = add_pointcloud_dropout(data, self.pointcloud_dropout, self.hetero, self.use_world_edges)

        # # extract first graph
        # edge_batch = data.batch[data.edge_index[0]]
        # first_edge_batch = edge_batch == 0
        # first_batch = data.batch == 0
        # first_x = data.x[first_batch]
        # first_pos = data.pos[first_batch]
        # first_edge_index = data.edge_index[:, first_edge_batch]
        # first_edge_attr = data.edge_attr[first_edge_batch]
        # first_data = Data(
        #     x=first_x,
        #     pos=first_pos,
        #     edge_index=first_edge_index,
        #     edge_attr=first_edge_attr,
        # )
        # first_batch = Batch.from_data_list([first_data])
        #
        # plot_pyg_graph(first_batch.to("cpu"), plot_pc=True)

        device = data.x.device
        data = add_noise_to_mesh_nodes(data, self.input_mesh_noise, device)
        data = add_noise_to_pcd_points(data, self.input_pcd_noise, device)
        data = transform_position_to_edges(data, self.euclidian_distance)
        data = convert_to_hetero_data(
            data, self.hetero, self.use_color, device,
            self.is_3d, self.use_world_edges, self.use_mesh_coordinates, self.mgn_hetero,
        )

        node_features_out, _, _ = self.gnn(data)
        velocity = predict_velocity(node_features_out, data, self.hetero, self.mgn_hetero)
        predict_collider = "collider_identifier" in batch.x_description[0]
        if predict_collider:
            # average velocities for the collider nodes to get a rigid prediction
            deformable_mask = batch.x[:, 2] == 1
            collider_mask = batch.x[:, batch.x_description[0].index("collider_identifier")] == 1
            collider_mask = collider_mask[deformable_mask]  # (num_deformable_nodes,)
            velocity_collider = velocity[collider_mask]
            # average
            velocity_collider_mean = velocity_collider.mean(dim=0, keepdim=True)
            velocity[collider_mask] = velocity_collider_mean

        predicted_position = data.y_old + velocity
        loss = self.criterion(predicted_position, data.y)

        self.training_step_outputs.append({"loss": loss.detach()})
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([o["loss"] for o in self.training_step_outputs]).mean()
        normalized_loss = avg_loss / self.loss_normalizer
        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_loss_normalized", normalized_loss)
        print(f"Epoch {self.current_epoch}: train_loss = {avg_loss:.6f}")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # get complete data, there is no context/target split, just predict all trajs in the val split of the tasks
        mesh_traj = batch.mesh_traj.cpu()  # (T, num_mesh_nodes, 2)
        x_desc = batch.x_description[0]
        if "collider_identifier" in x_desc:
            # predict collider case, update the mesh_traj with the collider nodes
            collider_traj = batch.collider_traj.cpu()  # (T, num_collider_nodes, 2)
            mesh_traj = torch.cat([collider_traj, mesh_traj, ], dim=2)  # (T, num_mesh_nodes + num_collider_nodes, 2)
        predicted_deformable = []
        for data in batch.to_data_list():
            data = Batch.from_data_list([data])
            predicted_deformable_traj = self.full_rollout(data, pc_k=self.config.eval_k, pc_cutoff=-1,
                                                          no_pcd=self.config.get("no_pcd", False))
            predicted_deformable.append(predicted_deformable_traj)
        predicted_deformable = torch.stack(predicted_deformable)
        ml_loss = self.criterion(predicted_deformable, mesh_traj)
        self.validation_step_outputs.append({"metrics": {"ml_loss": ml_loss.detach()}})
        return ml_loss

    def real_full_rollout(self, stacked_pc_data,
                          stacked_pc_type,
                          stacked_sheet_pos,
                          stacked_sphere_pos,
                          sheet_faces,
                          sphere_faces,
                          diameter,
                          stacked_time_stamps):
        edge_radius_dict, _, _, _ = get_connectivity_setting(
            self._eval_ds.config.connectivity_setting
        )
        pcd_traj = stacked_pc_data
        sheet_indices = get_edge_indices(sheet_faces)
        sphere_indices = get_edge_indices(sphere_faces)
        with open("config/trampoline_collider_connection_nodes_v4.json", "r") as f:
            connection_nodes = json.load(f)
        mesh_connection_nodes = torch.tensor(connection_nodes["sheet"])
        sphere_connection_nodes = torch.tensor(connection_nodes[str(diameter)])


        rollout_length = 25
        print("stop")
        initial_mesh_pos = ...
        predicted_trajectory = [torch.clone(current_deformable_pos.detach().cpu())]

    def full_rollout(self, batch, pc_k=2, pc_cutoff=-1, no_pcd=False) -> Tensor:
        edge_radius_dict, _, _, _ = get_connectivity_setting(
            self._eval_ds.config.connectivity_setting
        )
        # remove batch dim
        collider_traj = batch.collider_traj[0].cpu()  # (T, num_collider_nodes, 2)
        pcd_traj = batch.pcd_traj[0].cpu()  # (T, num_pcd_nodes, 2)
        mesh_mesh_indices = batch.mesh_mesh_indices.cpu()  # (num_mesh_edges, 2)

        rollout_length = batch.mesh_traj.shape[1] - 1  # first step already given as input

        device = batch.x.device
        deformable_mask = batch.x[:, 2] == 1
        current_deformable_pos = batch.pos[deformable_mask]
        initial_mesh_pos = torch.clone(current_deformable_pos).cpu()
        predicted_trajectory = [torch.clone(current_deformable_pos.detach().cpu())]
        x_desc = batch.x_description[0]
        predict_collider = "collider_identifier" in x_desc
        if predict_collider:
            # get a collider mask from the current deformable pos
            predicted_collider_mask = batch.x[:, x_desc.index("collider_identifier")] == 1
            predicted_collider_mask = predicted_collider_mask[deformable_mask]  # (num_deformable_nodes,)
        if pc_cutoff < 0:
            # set it to max value
            pc_cutoff = rollout_length
        for t in range(rollout_length):
            if t % pc_k == 0 and (t <= pc_cutoff) and not no_pcd:
                pc = pcd_traj[t]
            else:
                # no pointcloud
                pc = torch.empty((0, pcd_traj[t].shape[1]))
            # create datastamp, y is not used, so reuse current_deformable_pos as placeholder
            if predict_collider:
                current_collider = current_deformable_pos[predicted_collider_mask]
                current_mesh = current_deformable_pos[~predicted_collider_mask]
                if t == 0:
                    current_vel = current_deformable_pos.cpu() - predicted_trajectory[-1]
                else:
                    current_vel = current_deformable_pos.cpu() - predicted_trajectory[-2]
                # add zero for point cloud
                pc_vel = torch.zeros_like(pc)
                current_vel = torch.cat([pc_vel, current_vel], dim=0)
                data_timestep = (pc, current_collider.cpu(), current_mesh.cpu(), mesh_mesh_indices,
                                 current_mesh.cpu(),
                                 None, initial_mesh_pos, None
                                 )
            else:
                current_col_vel = collider_traj[t] - collider_traj[max(t - 1, 0)]
                if t == 0:
                    current_mesh_vel = current_deformable_pos.cpu() - predicted_trajectory[-1]
                else:
                    current_mesh_vel = current_deformable_pos.cpu() - predicted_trajectory[-2]
                pc_vel = torch.zeros_like(pc)
                current_vel = torch.cat([pc_vel, current_col_vel.cpu(), current_mesh_vel.cpu()], dim=0)
                data_timestep = (pc, collider_traj[t], current_deformable_pos.cpu(), mesh_mesh_indices,
                                 current_deformable_pos.cpu(),
                                 None, initial_mesh_pos, None
                                 )
            # show current prediction
            # plt.scatter(current_deformable_pos[:, 0].cpu(), current_deformable_pos[:, 1].cpu(), label="predicted",
            #             color="blue")
            # plt.scatter(collider_traj[t][:, 0].cpu(), collider_traj[t][:, 1].cpu(), label="collider",
            #             color="black")
            # if len(pc) > 0:
            #     plt.scatter(pc[:, 0].cpu(), pc[:, 1].cpu(), label="pcd", color="red")
            # plt.show()
            # add forces if they are there
            if "forces" in batch.keys():
                forces = batch["forces"]
                data_timestep = (*data_timestep, forces)
            else:
                data_timestep = (*data_timestep, None)
            # add computed velocities
            data_timestep = (*data_timestep, current_vel)
            current_edge_radius_dict = get_radius_dict_for_evaluation_mode(t, edge_radius_dict, self.config.eval_mode,
                                                                           pc_k)
            data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device,
                                         use_mesh_coordinates=True, hetero=False, tissue_task=False,
                                         predict_collider=predict_collider)
            data = transform_position_to_edges(data, self.euclidian_distance)
            node_features_out, _, _ = self.gnn(data)
            velocity = predict_velocity(node_features_out, data, self.hetero, self.mgn_hetero)
            if predict_collider:
                velocity_collider = velocity[predicted_collider_mask]
                # average
                velocity_collider_mean = velocity_collider.mean(dim=0, keepdim=True)
                velocity[predicted_collider_mask] = velocity_collider_mean

            current_deformable_pos = current_deformable_pos + velocity
            predicted_trajectory.append(torch.clone(current_deformable_pos.detach().cpu()))

        # compare predicted trajectory to ground truth
        predicted_trajectory = torch.stack(predicted_trajectory, dim=0)  # (T, num_deformable_nodes, 2)
        return predicted_trajectory

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        epoch_average = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            val_loss = metric_values.mean()
            epoch_average[f"val_{metric_name}"] = val_loss
        self.log_dict(epoch_average, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        # get complete data, there is no context/target split, just predict all trajs in the val split of the tasks
        ml_loss_list = []
        pc_k_list = []
        pc_cutoff_list = []
        vis_list = []
        # TODO: adapt to desired vis selection. Currently only one of the visualizations is saved, select the right one
        vis_pc_k = 1
        vis_pc_cutoff = -1

        for pc_k in [1, 2]:
            for pc_cutoff in [-1]:
                mesh_traj = batch.mesh_traj.cpu()  # (batchsize, T, num_mesh_nodes, 2)
                x_desc = batch.x_description[0]
                if "collider_identifier" in x_desc:
                    # predict collider case, update the mesh_traj with the collider nodes
                    collider_traj = batch.collider_traj.cpu()  # (T, num_collider_nodes, 2)
                    mesh_traj = torch.cat([collider_traj, mesh_traj],
                                          dim=2)  # (T, num_mesh_nodes + num_collider_nodes, 2)
                predicted_deformable = []
                for data in batch.to_data_list():
                    data = Batch.from_data_list([data])
                    predicted_deformable_traj = self.full_rollout(data, pc_k=pc_k, pc_cutoff=pc_cutoff,
                                                                  no_pcd=self.config.get("no_pcd", False))
                    predicted_deformable.append(predicted_deformable_traj)
                predicted_deformable = torch.stack(predicted_deformable)
                ml_loss = self.criterion(predicted_deformable, mesh_traj)
                ml_loss_list.append(ml_loss.detach())
                pc_k_list.append(torch.tensor(pc_k, dtype=torch.float))
                pc_cutoff_list.append(torch.tensor(pc_cutoff, dtype=torch.float))

                to_visualize = self.config.evaluator.task_ids_to_visualize
                vis_results = {}
                if batch_idx in to_visualize and pc_k == vis_pc_k and pc_cutoff == vis_pc_cutoff:
                    # save all stuff for visualization
                    # TODO: Right now only save the last traj in the batch, change that to save multiple trajs if needed
                    vis_results["predicted_trajectory"] = predicted_deformable[-1].detach().cpu()
                    vis_results["ground_truth_trajectory"] = mesh_traj[-1].detach().cpu()
                    # there are 2 cases: one where collider is predicted, and one where is not. Check h-description and decide upon that
                    if "collider_identifier" in x_desc:
                        # predicted collider
                        # decide upon the collider identifier to get the predicted and the gth collider nodes
                        collider_id_index = x_desc.index("collider_identifier")
                        # use the last data point in the batch so that only 1 set of points is selected. we have an explicit batch dim in the vis data
                        # while in batch it is implicit..
                        collider_mask = batch[len(batch) - 1]["x"][:, collider_id_index] == 1
                        deformable_mask = batch[len(batch) - 1].x[:, 2] == 1
                        collider_mask = collider_mask[deformable_mask].cpu()  # (num_deformable_nodes,)
                        vis_results["predicted_collider"] = predicted_deformable[-1, :, collider_mask, :].detach().cpu()
                        vis_results["ground_truth_collider"] = mesh_traj[-1, :, collider_mask, :].detach().cpu()
                        # also update the deformable trajectory: keep only the ones which are not collider nodes
                        vis_results["predicted_trajectory"] = predicted_deformable[
                            -1, :, ~collider_mask, :].detach().cpu()
                        vis_results["ground_truth_trajectory"] = mesh_traj[-1, :, ~collider_mask, :].detach().cpu()
                        # save connectivity
                        vis_results["collider_connectivity"] = batch.vis_collider_faces[-1].cpu()
                        vis_results["mesh_connectivity"] = batch.vis_mesh_faces[-1].cpu()
                    else:
                        # fixed collider, get from dataset if existent
                        collider = batch[len(batch) - 1]["collider_traj"][0]  # (T, num_collider_nodes, 2/3)
                        # check for empty collider
                        if collider.shape[1] == 0:
                            collider = None
                        vis_results["collider"] = collider.detach().cpu() if collider is not None else None
                        if "vis_collider_faces" in batch[len(batch) - 1]:
                            collider_connectivity = batch[len(batch) - 1].vis_collider_faces
                        else:
                            collider_connectivity = None
                        if collider_connectivity is not None:
                            vis_results["collider_connectivity"] = collider_connectivity[-1].cpu()
                        vis_results["mesh_connectivity"] = batch.vis_mesh_faces[-1].cpu()
                    vis_list.append(vis_results)

                print("Evaluated pc_k =", pc_k, "pc_cutoff =", pc_cutoff, "ml_loss =", ml_loss.item())
        result = {"metrics": {
            "ml_loss": torch.stack(ml_loss_list), "pc_k": torch.stack(pc_k_list),
            "pc_cutoff": torch.stack(pc_cutoff_list)},
            "visualizations": vis_list[-1],
            "material_properties": None,
        }
        self.test_step_outputs.append(result)
        return result

    def on_test_end(self) -> None:
        outputs = self.test_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        epoch_average = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, different k configurations)
            test_loss = metric_values.mean(dim=0)  # average over batches, shape (different k configurations)
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
        optimizer = _get_optimizer(config=self.config.optimizer,
                                   models={
                                       "gnn": self.gnn})
        scheduler = _get_scheduler(config=self.config.scheduler, optimizer=optimizer)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": self.config.check_val_every_n_epoch,
            "monitor": "val_mse",
            "strict": False,
        }
        return [optimizer], [scheduler_config]


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


def plot_pyg_graph(graph, ax=None, plot_pc=True):
    node_types_np = graph.node_stores[0]["x"].numpy()
    type_ids = node_types_np.argmax(axis=1)

    if plot_pc:
        keep_mask = np.ones(len(type_ids), dtype=bool)
    else:
        # mask out type 0 (i.e. [1,0,0])
        keep_mask = type_ids != 0
    keep_indices = np.where(keep_mask)[0]
    keep_set = set(keep_indices.tolist())

    pos = graph.pos.numpy()
    pos_filtered = pos[keep_mask]
    type_ids_filtered = type_ids[keep_mask]

    # Check if 3D
    is_3d = pos_filtered.shape[1] == 3

    # remap old node indices to new ones
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    # filter edges: keep only if BOTH endpoints are kept
    edge_index = graph.edge_index.numpy()
    edge_mask = np.array([
        s in keep_set and d in keep_set
        for s, d in edge_index.T
    ])
    edges_filtered = edge_index[:, edge_mask]
    edges_remapped = np.array([
        [old_to_new[s], old_to_new[d]]
        for s, d in edges_filtered.T
    ]).T

    unique_types = np.unique(type_ids_filtered)
    cmap = plt.cm.get_cmap("tab10", len(unique_types))
    color_map = {t: cmap(i) for i, t in enumerate(unique_types)}
    node_colors = [color_map[t] for t in type_ids_filtered]

    if ax is None:
        if is_3d:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

    # edges
    if is_3d:
        for src, dst in edges_remapped.T:
            xs = [pos_filtered[src, 0], pos_filtered[dst, 0]]
            ys = [pos_filtered[src, 1], pos_filtered[dst, 1]]
            zs = [pos_filtered[src, 2], pos_filtered[dst, 2]]
            ax.plot(xs, ys, zs, color="gray", linewidth=0.7, alpha=0.5, zorder=1)
    else:
        for src, dst in edges_remapped.T:
            xs = [pos_filtered[src, 0], pos_filtered[dst, 0]]
            ys = [pos_filtered[src, 1], pos_filtered[dst, 1]]
            ax.plot(xs, ys, color="gray", linewidth=0.7, alpha=0.5, zorder=1)

    # nodes
    if is_3d:
        ax.scatter(
            pos_filtered[:, 0], pos_filtered[:, 1], pos_filtered[:, 2],
            c=node_colors,
            s=60, zorder=2, edgecolors="white", linewidths=0.5
        )
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.scatter(
            pos_filtered[:, 0], pos_filtered[:, 1],
            c=node_colors,
            s=60, zorder=2, edgecolors="white", linewidths=0.5
        )
        ax.set_aspect("equal")

    # legend
    patches = [
        mpatches.Patch(color=color_map[t], label=f"Type {t}")
        for t in unique_types
    ]
    ax.legend(handles=patches, framealpha=0.9, fontsize=9)

    if not is_3d:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return ax
