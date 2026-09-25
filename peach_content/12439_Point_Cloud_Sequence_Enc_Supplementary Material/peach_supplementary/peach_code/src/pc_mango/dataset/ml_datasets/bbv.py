import json
from typing import Any

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

from pc_mango.dataset.edges.edge_indices import get_edge_indices
from pc_mango.dataset.ml_datasets.abstract_ml_dataset import AbstractMLDataset
from pc_mango.util.own_types import ConfigDict
from pc_mango.util.util import to_numpy


class BBVDataset(AbstractMLDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        # normalization
        self.min_x = torch.min(torch.tensor([[0.0, 0.0]]))
        self.max_x = torch.max(torch.tensor([[20.0, 20.0]]))
        # edge indices tensor: no global mesh available, have to compute it live (maybe cache it)
        assert "external_amg_root" in self.config and self.config.external_amg_root is not None, "external_amg_root must be specified in config for BBV dataset"
        with h5py.File(self.config.external_amg_root, "r") as amg_hdf:
            for task_dict in tqdm(self.tasks, "Loading AMG data for tasks"):
                task_key = task_dict["task_key"]
                amg_path = amg_hdf[task_key]["trajs"]["traj_000"]
                amg_lvl_1 = torch.tensor(amg_path["amg_lvl_1"][:], dtype=torch.long)
                mask = amg_lvl_1[0] != amg_lvl_1[1]
                amg_lvl_1 = amg_lvl_1[:, mask]
                if "amg_lvl_2" in amg_path:
                    amg_lvl_2 = torch.tensor(amg_path["amg_lvl_2"][:], dtype=torch.long)
                    mask = amg_lvl_2[0] != amg_lvl_2[1]
                    amg_lvl_2 = amg_lvl_2[:, mask]
                else:
                    amg_lvl_2 = None
                for traj_dict in task_dict["trajs"].values():
                    traj_dict["amg_lvl_1"] = amg_lvl_1
                    traj_dict["amg_lvl_2"] = amg_lvl_2



    def normalization(self, x):
        return (x - self.min_x) / (self.max_x - self.min_x)

    def get(self, task_idx):
        task_dict = self.tasks[task_idx]
        result = {}
        context_trajs, target_trajs = self.get_context_target_trajs(len(task_dict["trajs"].keys()))
        result["context_trajs"] = context_trajs  # shape (num_context_trajs)
        result["target_trajs"] = target_trajs  # shape (num_target_trajs)
        material_properties = self.get_material_properties(task_dict)
        result["regression_features"] = material_properties.squeeze()
        # mesh pos
        if "mesh" in self.config.data_modalities:
            mesh_pos = torch.stack(
                [task_dict["trajs"][traj_key]["mesh_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
            # remove z dim
            mesh_pos = mesh_pos[:, :, :, :2]
            # normalization
            mesh_pos = self.normalization(mesh_pos)
            x = mesh_pos
            # apply tail removal and temporal stride
            if self.config.cut_length > 0:
                x = x[:, :self.config.cut_length, :, :]
            if self.config.temporal_stride > 1:
                x = x[:, ::self.config.temporal_stride, :, :]
            h = []
            for traj_key in sorted(task_dict["trajs"].keys()):
                traj_dict = task_dict["trajs"][traj_key]
                current_h, h_description = self.get_current_node_features(traj_dict, material_properties, x)
                h.append(current_h)
            h = torch.stack(h)
            # velocity
            v = self.compute_velocity(x)
            # edge features and other stuff
            edge_feature_description, edge_features, edge_indices, faces = self.build_edges(h, task_dict, x)
            # debug vis
            # pos_np = x[0, 0].numpy() if isinstance(x, torch.Tensor) else x
            # ei = edge_indices.numpy() if isinstance(edge_indices, torch.Tensor) else edge_indices
            # ef = edge_features[0].numpy() if isinstance(edge_features, torch.Tensor) else edge_features
            #
            # colors = ['#888888', '#e74c3c', '#2ecc71']  # mesh, lvl1, lvl2
            # labels = ['Original mesh', 'AMG level 1', 'AMG level 2']
            # alphas = [0.2, 0.7, 0.9]
            # linewidths = [0.8, 1.5, 2.0]
            #
            # fig, ax = plt.subplots(figsize=(9, 9))
            #
            # for lvl in range(2):
            #     mask = ef[:, lvl] == 1
            #     idx = np.where(mask)[0]
            #     for i in idx:
            #         src, dst = ei[0, i], ei[1, i]
            #         ax.plot(
            #             [pos_np[src, 0], pos_np[dst, 0]],
            #             [pos_np[src, 1], pos_np[dst, 1]],
            #             color=colors[lvl], alpha=alphas[lvl],
            #             linewidth=linewidths[lvl],
            #         )
            #     # dummy line for legend
            #     ax.plot([], [], color=colors[lvl], alpha=alphas[lvl],
            #             linewidth=linewidths[lvl], label=f"{labels[lvl]} ({mask.sum()} edges)")
            #
            # ax.scatter(pos_np[:, 0], pos_np[:, 1], c='black', s=12, zorder=5)
            # for i, (nx, ny) in enumerate(pos_np):
            #     ax.annotate(str(i), (nx, ny), fontsize=5, color='black',
            #                 xytext=(3, 3), textcoords='offset points')
            # print("lvl 1 edges:", ei[:, ef[:, 1] == 1].T)
            # print("lvl 2 edges:", ei[:, ef[:, 2] == 1].T)
            # ax.axis('equal')
            # ax.legend()
            # ax.set_title("Mesh + AMG hierarchy edges")
            # plt.tight_layout()
            # plt.show()

            result.update({
                "x": x,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
                "v": v,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
                "h": h,  # shape (num_trajs_in_task, num_nodes, node_feature_dim)
                "h_description": h_description,
                "edge_indices": edge_indices,  # shape (2, num_edges)
                "edge_features": edge_features,  # shape (num_trajs_in_task, num_edges, num_edge_features)
                "edge_feature_description": edge_feature_description,
            })
            if self.split == "test":
                # also add faces, since they are different per task, needed for vis
                result["meta_data"] = {
                    "faces": faces
                }
        if "pc" in self.config.data_modalities:
            if self.config.pc_preprocessing.output_type == "tensor":
                # load tensor of shape (context_size, traj_length, num_points, point_dim)
                pcs = []
                colors = []
                for traj_key in sorted(task_dict["trajs"].keys()):
                    pc = task_dict["trajs"][traj_key]["processed_pc"]
                    color = task_dict["trajs"][traj_key]["processed_pc_color"]
                    color = color[:, :, None]  # add feature dim
                    colors.append(color)

                    pcs.append(pc)

                pc = torch.stack(pcs)
                # normalize
                pc = self.normalization(pc)
                color = torch.stack(colors)
                assert self.config.temporal_stride == 1, "The inclusion of the force assumes that general temporal stride is 1"
                pc_temporal_stride = self.config.pc_preprocessing.temporal_stride

                # add feature dim to color
                force_values = h[:, :, :, 3:5]  # shape (num_trajs_in_task, num_nodes, 2)
                force_values = force_values.max(dim=2)[0]
                force_values = force_values[:, ::pc_temporal_stride, None, :]  # apply same temporal stride as pc, add point dim
                force_values = force_values.repeat(1, 1, color.shape[2], 1)  # repeat for all points
                # add to color
                color = torch.cat([color, force_values], dim=-1)
                if self.split != "test":
                    # filter to context trajs only
                    pc = pc[context_trajs]
                    color = color[context_trajs]
                result["pc"] = pc  # shape (num_trajs_in_task, traj_length, num_points, point_dim)
                result["pc_color"] = color  # shape (num_trajs_in_task, traj_length, num_points, 3) (color, fx, fy)
            elif self.config.pc_preprocessing.output_type == "graph":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown pc output type: {self.config.pc_preprocessing.output_type}")
        if "sdf" in self.config.data_modalities:
            sdfs = []
            queries = []
            for traj_key in sorted(task_dict["trajs"].keys()):
                if int(traj_key[-3:]) not in context_trajs and self.split != "test":
                    # in normal training/validation, only use context trajs, but in test use all trajs, since we want to evaluate different context sizes at once
                    continue
                sdf = task_dict["trajs"][traj_key]["sdf"]
                query = task_dict["trajs"][traj_key]["queries"]
                sdfs.append(sdf)
                queries.append(query)
            sdfs = torch.stack(sdfs)
            queries = torch.stack(queries)
            # add z dim (set to 0)
            queries = torch.cat([queries, torch.zeros_like(queries[:, :, :, :1])], dim=-1)
            selected_sdfs, selected_queries = self.sdf_subsample(sdfs, queries)
            # normalization
            selected_queries = self.normalization(selected_queries)
            result["sdf"] = selected_sdfs  # shape (num_trajs_in_task, traj_length, num_query_points_per_ts, num_sdf_channels)
            result["queries"] = selected_queries  # shape (num_trajs_in_task, traj_length, num_query_points_per_ts, 3)
        return result

    def build_edges(self, h: Tensor, task_dict: list[Any] | Any,
                    x: Tensor | Any) -> tuple[Tensor, Tensor, list[str], Any]:
        faces = task_dict["trajs"]["traj_000"]["faces"]
        edge_indices_lvl_0 = get_edge_indices(faces)
        edge_features_lvl_0 = torch.zeros(edge_indices_lvl_0.shape[1], 3)
        edge_features_lvl_0[:, 0] = 1.0
        edge_indices_lvl_1 = task_dict["trajs"]["traj_000"]["amg_lvl_1"]
        edge_features_lvl_1 = torch.zeros(edge_indices_lvl_1.shape[1], 3)
        edge_features_lvl_1[:, 1] = 1.0
        if task_dict["trajs"]["traj_000"]["amg_lvl_2"] is not None:
            edge_indices_lvl_2 = task_dict["trajs"]["traj_000"]["amg_lvl_2"]
            edge_features_lvl_2 = torch.zeros(edge_indices_lvl_2.shape[1], 3)
            edge_features_lvl_2[:, 2] = 1.0
            edge_indices = torch.cat([edge_indices_lvl_0, edge_indices_lvl_1, edge_indices_lvl_2], dim=1)
            edge_features = torch.cat([edge_features_lvl_0, edge_features_lvl_1, edge_features_lvl_2], dim=0)
        else:
            edge_indices = torch.cat([edge_indices_lvl_0, edge_indices_lvl_1], dim=1)
            edge_features = torch.cat([edge_features_lvl_0, edge_features_lvl_1], dim=0)

        edge_features = edge_features[None, :, :].repeat(h.shape[0], 1, 1)
        init_r_ij = self.get_initial_rel_features(x, edge_indices)
        edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
        edge_feature_description = ["one_hot"] * 3 + ["init_r_ij"] * 2
        return edge_feature_description, edge_features, edge_indices, faces

    def get_current_node_features(self, traj_dict, material_properties, x):
        h_description = ["one_hot", "handle", "force", "force_value_x", "force_value_y"]
        # only one general node type, namely a mesh node
        one_hot = torch.ones(x.shape[1], x.shape[2], 1)
        # however, we have 2 additional types: if is a handle (i.e. fixed) or controlled by force
        # this is the only task where h is time dependent
        handle = torch.zeros_like(one_hot)
        handle_mask = traj_dict["physical_node_type"][:, 0] == 3
        handle[:, handle_mask, :] = 1
        force = torch.zeros_like(one_hot)
        force_mask = traj_dict["physical_node_type"][:, 0] == 8
        force[:, force_mask, :] = 1
        force_value = traj_dict["force_bc"][..., :2]
        # give every node the force value
        max_force_per_ts = force_value.max(dim=1)[0]
        max_force_per_ts = max_force_per_ts.unsqueeze(1).repeat(1, force_value.shape[1], 1)
        force_value = max_force_per_ts
        current_h = torch.cat([one_hot, handle, force, force_value], dim=-1)

        if self.config.material_properties:
            material_properties = material_properties.repeat(x.shape[1], x.shape[2], 1)
            current_h = torch.cat([current_h, material_properties], dim=-1)
            h_description += self.mat_prop_description
        return current_h, h_description

    def get_material_properties(self, task_dict):
        poisson = torch.log(task_dict["params"]["poisson"])
        viscosity = torch.log(task_dict["params"]["viscosity"])
        youngs_modulus = torch.log(task_dict["params"]["youngs_modulus"])
        material_properties = torch.stack([poisson, viscosity, youngs_modulus])[None, :]
        return material_properties

    def get_faces(self):
        faces = self.global_data
        faces = {key: to_numpy(face) for key, face in faces.items()}
        output_faces = {"sheet_faces": faces["sheet_cells"]}
        ball_faces = {}
        for key in faces.keys():
            if key.startswith("ball"):
                diameter = key.split("_")[1]
                ball_faces[diameter] = faces[key]
        output_faces["ball_faces"] = ball_faces
        return output_faces

    def _raw_traj_length(self):
        return 101

    @property
    def mat_prop_description(self):
        return ["log_poisson", "log_viscosity", "log_youngs_modulus"]