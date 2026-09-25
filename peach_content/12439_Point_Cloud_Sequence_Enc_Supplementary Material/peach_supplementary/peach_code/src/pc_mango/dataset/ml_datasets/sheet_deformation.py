import os
import os.path as osp
from typing import Union

import numpy as np
import torch

from pc_mango.dataset.edges.edge_indices import get_edge_indices
from pc_mango.dataset.ml_datasets.abstract_ml_dataset import AbstractMLDataset
from pc_mango.util.own_types import ConfigDict
from pc_mango.util.util import to_numpy


class SheetDeformationDataset(AbstractMLDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        # normalization
        self.normalization_factor = 140
        self.force_normalization_factor = 200
        self.border_indices = self.get_node_indices_on_border()
        # edge indices tensor
        faces = self.global_data["cells"]
        self.edge_indices = get_edge_indices(faces)

    def get_node_indices_on_border(self):
        # first step of first traj of first task
        nodes = np.array(self.tasks[0]["trajs"]["traj_000"]["mesh_pos"][0])
        # do it in numpy since code was already implemented like that
        x_min = np.min(nodes[:, 0])
        x_max = np.max(nodes[:, 0])
        y_min = np.min(nodes[:, 1])
        y_max = np.max(nodes[:, 1])
        # get the node with indices < size
        node_indices = \
            np.where((nodes[:, 0] == x_min) | (nodes[:, 0] == x_max) | (nodes[:, 1] == y_min) | (nodes[:, 1] == y_max))[
                0]
        return torch.tensor(node_indices)

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
            x = torch.stack([task_dict["trajs"][traj_key]["mesh_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
            # normalization
            x = x / self.normalization_factor
            # build h tensor
            h = []
            for traj_key in sorted(task_dict["trajs"].keys()):
                traj_dict = task_dict["trajs"][traj_key]
                current_h, h_description = self.get_current_node_features(traj_dict, material_properties)
                h.append(current_h)
            h = torch.stack(h)
            # velocity
            v = self.compute_velocity(x)
            edge_indices = self.edge_indices
            edge_features = torch.zeros(edge_indices.shape[1], 2)  # edge id: 0: node-node, 1: node-encoder
            edge_features[:, 0] = 1.0
            edge_features = edge_features[None, :, :].repeat(h.shape[0], 1, 1)
            init_r_ij = self.get_initial_rel_features(x, edge_indices)
            edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
            edge_feature_description = ["one_hot"] * 2 + ["init_r_ij"] * 3
            result.update({
                "x": x,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
                "v": v,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
                "h": h,  # shape (num_trajs_in_task, num_nodes, node_feature_dim)
                "h_description": h_description,
                "edge_indices": edge_indices,  # shape (2, num_edges)
                "edge_features": edge_features,  # shape (num_trajs_in_task, num_edges, num_edge_features)
                "edge_feature_description": edge_feature_description,
            })
        if "pc" in self.config.data_modalities:
            if self.config.pc_preprocessing.output_type == "tensor":
                # load tensor of shape (context_size, traj_length, num_points, point_dim)
                pcs = []
                colors = []
                for traj_key in sorted(task_dict["trajs"].keys()):
                    if int(traj_key[-3:]) not in context_trajs and self.split != "test":
                        # in normal training/validation, only use context trajs, but in test use all trajs, since we want to evaluate different context sizes at once
                        continue
                    pc = task_dict["trajs"][traj_key]["processed_pc"]
                    color = task_dict["trajs"][traj_key]["processed_pc_color"]
                    colors.append(color)
                    # normalization
                    pc = pc / self.normalization_factor
                    pcs.append(pc)
                pc = torch.stack(pcs)
                color = torch.stack(colors)
                result["pc_color"] = color  # shape (num_trajs_in_task, traj_length, num_points, 1)
                result["pc"] = pc  # shape (num_trajs_in_task, traj_length, num_points, point_dim)
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
            selected_sdfs, selected_queries = self.sdf_subsample(sdfs, queries)
            # normalization
            selected_queries = selected_queries / self.normalization_factor
            result["sdf"] = selected_sdfs  # shape (num_trajs_in_task, traj_length, num_query_points_per_ts, num_sdf_channels)
            result["queries"] = selected_queries  # shape (num_trajs_in_task, traj_length, num_query_points_per_ts, 3)
        return result

    def get_current_node_features(self, traj_dict, material_properties):
        h_description = ["one_hot", "one_hot"]
        init_pos = traj_dict["mesh_pos"][0]
        force_features = []
        for force_dict in traj_dict["params"].values():
            force_nodes = self.get_node_indices_with_force_influence(init_pos, force_dict["position"])
            force_feature = torch.zeros((init_pos.shape[0], 1))
            force_feature[force_nodes] = force_dict["direction"][2] / self.force_normalization_factor
            force_features.append(force_feature)
            h_description.append("force")
        # 2 node ids: normal mesh node: 0, output encoder node: 1
        node_id = torch.zeros(init_pos.shape[0], 2)
        node_id[:, 0] = 1.0
        force_features = torch.cat(force_features, dim=-1)
        border_features = torch.zeros(init_pos.shape[0], 1)
        border_features[self.border_indices] = 1.0
        current_h = torch.cat([node_id, force_features, border_features], dim=-1)
        h_description.append("border")
        if self.config.material_properties:
            material_properties = material_properties.repeat(current_h.shape[0], 1)
            current_h = torch.cat([current_h, material_properties], dim=-1)
            h_description += self.mat_prop_description
        return current_h, h_description

    @staticmethod
    def get_node_indices_with_force_influence(nodes, force_pos):
        force_application = {
            "length": 20,
            "search_mode": "Cylinder"
        }
        if force_application["search_mode"] == "Cylinder":
            radius = force_application["length"] / 2
            distances = torch.linalg.norm(nodes - force_pos, axis=1)
            # get the node with indices < size
            node_indices = torch.where(distances < radius)[0]
            return node_indices

    def get_faces(self):
        faces = self.global_data["cells"]
        faces = {"deformable": to_numpy(faces)}
        return faces

    def get_material_properties(self, task_dict):
        youngs_modulus = torch.tensor([task_dict["params"]["youngs_modulus"]])
        # max youngs_modulus is 500
        youngs_modulus = youngs_modulus / 500
        # take the log of the youngs modulus to make it easier to learn
        youngs_modulus = torch.log(youngs_modulus + 1e-8)
        material_properties = torch.cat([youngs_modulus], dim=-1)
        return material_properties


    def _raw_traj_length(self):
        return 51

    @property
    def mat_prop_description(self):
        return ["youngs_modulus"]