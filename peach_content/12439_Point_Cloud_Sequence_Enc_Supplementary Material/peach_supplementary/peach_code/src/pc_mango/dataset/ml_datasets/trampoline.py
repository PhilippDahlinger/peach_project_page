import json
from copy import deepcopy

import numpy as np
import torch

from pc_mango.dataset.edges.edge_indices import get_edge_indices
from pc_mango.dataset.ml_datasets.abstract_ml_dataset import AbstractMLDataset
from pc_mango.util.own_types import ConfigDict
from pc_mango.util.util import to_numpy
from pc_mango.visualization.debug_pc_plot import visualize_pointcloud_sequence


class TrampolineDataset(AbstractMLDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        # edge indices tensor
        self.faces = None
        self.compute_edge_indices()
        # normalization
        # do a scalar normalization to retain the relative distances to accurately present physics
        # this means that some of the points will be outside [-1, 1] since the z dim was a bit bigger in min/max, but thats ok

        self.min_x = torch.min(torch.tensor([-145.0, -145.0, -145.0]))
        self.max_x = torch.max(torch.tensor([[145.0, 145.0, 145.0]]))
        if "gravity_azimuth_deg" in self.tasks[0]["params"].keys():
            self.has_gravity_perturbation = True
        else:
            self.has_gravity_perturbation = False

    def compute_edge_indices(self):
        faces = self.get_faces()
        self.faces = deepcopy(
            faces)  # save self.faces without offset for visualization, used in get_collider_connectivity()
        # offset collider indices
        collider_offset = np.max(faces["sheet_faces"]) + 1
        for key, ball_face in faces["ball_faces"].items():
            faces["ball_faces"][key] = faces["ball_faces"][key] + collider_offset
        self.mesh_mesh_indices = get_edge_indices(faces["sheet_faces"])

        self.edge_indices = {}
        self.base_edge_features = {}

        # load the connection indices from the json
        with open(self.config.connection_node_indices_path, "r") as f:
            connection_nodes = json.load(f)

        # compute col-x indices per ball face
        mesh_connection_nodes = torch.tensor(connection_nodes["sheet"])
        for key, ball_face in faces["ball_faces"].items():
            col_col_indices = get_edge_indices(ball_face)
            collider_connection_nodes = torch.tensor(connection_nodes[key] + collider_offset)

            mesh_col_indices = torch.cartesian_prod(mesh_connection_nodes, collider_connection_nodes).T
            col_mesh_indices = torch.cartesian_prod(collider_connection_nodes, mesh_connection_nodes).T
            self.edge_indices[key] = torch.cat(
                [self.mesh_mesh_indices, col_col_indices, mesh_col_indices, col_mesh_indices], dim=-1)
            # edge id: 0: node-node, 1: collider-collider, 2: node-collider, 3: collider-node
            edge_features = torch.zeros(self.edge_indices[key].shape[1], 4)
            edge_features[:self.mesh_mesh_indices.shape[1], 0] = 1.0
            edge_features[
                self.mesh_mesh_indices.shape[1]:self.mesh_mesh_indices.shape[1] + col_col_indices.shape[
                    1], 1] = 1.0
            edge_features[
                self.mesh_mesh_indices.shape[1] + col_col_indices.shape[1]:self.mesh_mesh_indices.shape[1] +
                                                                           col_col_indices.shape[1] +
                                                                           mesh_col_indices.shape[
                                                                               1], 2] = 1.0
            edge_features[
                self.mesh_mesh_indices.shape[1] + col_col_indices.shape[1] + mesh_col_indices.shape[
                    1]:, 3] = 1.0
            self.base_edge_features[key] = edge_features

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
                [task_dict["trajs"][traj_key]["sheet_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
            collider_pos = torch.stack(
                [task_dict["trajs"][traj_key]["sphere_pos"] for traj_key in sorted(task_dict["trajs"].keys())])
            diameter = str(task_dict["params"]["ball_diameter"].item())

            # normalization
            mesh_pos = self.normalization(mesh_pos)
            collider_pos = self.normalization(collider_pos)
            x = torch.cat((mesh_pos, collider_pos), dim=2)
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
            # load the correct edge indices and features based on the diameter
            edge_indices = self.edge_indices[diameter]
            edge_features = self.base_edge_features[diameter]
            edge_features = edge_features[None, :, :].repeat(h.shape[0], 1, 1)
            init_r_ij = self.get_initial_rel_features(x, edge_indices)
            edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
            edge_feature_description = ["one_hot"] * 4 + ["init_r_ij"] * 3
            result.update({
                "x": x,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
                "v": v,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
                "h": h,  # shape (num_trajs_in_task, num_nodes, node_feature_dim)
                "h_description": h_description,
                "edge_indices": edge_indices,  # shape (2, num_edges)
                "edge_features": edge_features,  # shape (num_trajs_in_task, num_edges, num_edge_features)
                "edge_feature_description": edge_feature_description,
                "meta_data": {
                    "diameter": diameter
                }
            })
        if "pc" in self.config.data_modalities:
            if self.config.pc_preprocessing.output_type == "tensor":
                # load tensor of shape (context_size, traj_length, num_points, point_dim)
                pcs = []
                colors = []
                for traj_key in sorted(task_dict["trajs"].keys()):
                    # in trampoline, the trajs indices start from 1, so we need to subtract 1 from the key to get the correct index for context_trajs and target_trajs
                    if int(traj_key[-3:]) - 1 not in context_trajs and self.split != "test":
                        # in normal training/validation, only use context trajs, but in test use all trajs, since we want to evaluate different context sizes at once
                        continue
                    pc = task_dict["trajs"][traj_key]["processed_pc"]
                    color = task_dict["trajs"][traj_key]["processed_pc_color"]
                    color = color[:, :, None]  # add feature dim
                    colors.append(color)
                    pcs.append(pc)
                pc = torch.stack(pcs)
                color = torch.stack(colors)
                # cut the point cloud so that num_pc_ts * pc_temporal_stride = cut_traj_length
                # therefore num_pc_ts = cut_traj_length // pc_temporal_stride
                pc_temporal_stride = self.config.pc_preprocessing.temporal_stride
                if self.config.cut_length > 0:
                    cut_length = self.config.cut_length
                    num_pc_ts = int(np.ceil(cut_length / pc_temporal_stride))
                    pc = pc[:, :num_pc_ts, :, :]
                    color = color[:, :num_pc_ts, :, :]
                # in this task, pcs are not normalized, do that here
                pc = self.normalization(pc)
                if "pc_augmentation" in self.config and self.config.pc_augmentation.enabled:
                    pc, color = self.augment_pc(pc, color)

                result["pc"] = pc  # shape (num_trajs_in_task, traj_length, num_points, point_dim)
                result["pc_color"] = color  # shape (num_trajs_in_task, traj_length, num_points, 1)
            elif self.config.pc_preprocessing.output_type == "graph":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown pc output type: {self.config.pc_preprocessing.output_type}")
        if "sdf" in self.config.data_modalities:
            sdfs = []
            queries = []
            for traj_key in sorted(task_dict["trajs"].keys()):
                # in trampoline, the trajs indices start from 1, so we need to subtract 1 from the key to get the correct index for context_trajs and target_trajs
                if int(traj_key[-3:]) - 1 not in context_trajs and self.split != "test":
                    # in normal training/validation, only use context trajs, but in test use all trajs, since we want to evaluate different context sizes at once
                    continue
                sdf = task_dict["trajs"][traj_key]["sdf"]
                query = task_dict["trajs"][traj_key]["queries"]
                sdfs.append(sdf)
                queries.append(query)
            sdfs = torch.stack(sdfs)
            queries = torch.stack(queries)
            selected_sdfs, selected_queries = self.sdf_subsample(sdfs, queries)
            # cut the sdf so that num_sdf_ts * sdf_temporal_stride = cut_traj_length
            # therefore num_sdf_ts = cut_traj_length // sdf_temporal_stride
            sdf_temporal_stride = self.config.sdf_preprocessing.temporal_stride
            if self.config.cut_length > 0:
                cut_length = self.config.cut_length
                num_sdf_ts = cut_length // sdf_temporal_stride
                selected_sdfs = selected_sdfs[:, :num_sdf_ts, :, :]
                selected_queries = selected_queries[:, :num_sdf_ts, :, :]
            # normalization
            selected_queries =  self.normalization(selected_queries)
            result[
                "sdf"] = selected_sdfs  # shape (num_trajs_in_task, traj_length, num_query_points_per_ts, num_sdf_channels)
            result["queries"] = selected_queries  # shape (num_trajs_in_task, traj_length, num_query_points_per_ts, 3)
        return result

    def augment_pc(self, pc, color):
        def augment_dropout_batch(pc, p, min_r, max_r):
            """
            pc: [batch_dim_1, batch_dim_2, num_points, 3]
            """
            # Decide which clouds get augmented
            augment_mask = torch.rand(batch_dim_1, batch_dim_2, device=pc.device) < p

            # Sample radii for all clouds
            radii = torch.rand(batch_dim_1, batch_dim_2, device=pc.device) * (max_r - min_r) + min_r

            # Sample center indices
            center_indices = torch.randint(num_points, (batch_dim_1, batch_dim_2), device=pc.device)

            # Process augmented clouds
            for i in range(batch_dim_1):
                for j in range(batch_dim_2):
                    if augment_mask[i, j]:
                        center = pc[i, j, center_indices[i, j]]
                        distances = torch.norm(pc[i, j] - center, dim=-1)
                        crop_mask = distances < radii[i, j]

                        n_cropped = crop_mask.sum()
                        pc[i, j, crop_mask] = torch.rand(n_cropped, 3, device=pc.device) * 0.5 + 0.25

            return pc

        # make sure that the original data is not disturbed
        pc = pc.clone()

        augmentation_config = self.config.pc_augmentation
        # select a third of the points which get noised by noise 1, another third by nose 2 and the rest stays
        batch_dim_1, batch_dim_2, num_points, _ = pc.shape

        pc = augment_dropout_batch(
            pc, augmentation_config.patch_crop.p,
            augmentation_config.patch_crop.min_radius,
            augmentation_config.patch_crop.max_radius
        )

        # Create random assignment: 0=no noise, 1=noise_std_1, 2=noise_std_2
        assignment = torch.randint(0, 3, (batch_dim_1, batch_dim_2, num_points), device=pc.device)

        mask_1 = assignment == 1
        mask_2 = assignment == 2

        noise_std_1 = augmentation_config.noise_std_1
        noise_std_2 = augmentation_config.noise_std_2

        # Apply flat Gaussian noise
        pc[mask_1] += torch.normal(0, noise_std_1, size=pc[mask_1].shape, device=pc.device)
        pc[mask_2] += torch.normal(0, noise_std_2, size=pc[mask_2].shape, device=pc.device)

        # example_pc = pc[0, : ,: ,:]
        # example_color = color[0, : ,: ,:]
        # visualize_pointcloud_sequence(example_pc, example_color)
        return pc, color




    def get_current_node_features(self, traj_dict, material_properties, x):
        h_description = ["one_hot", "one_hot", "mesh_identifier", "collider_identifier"]
        init_pos = traj_dict["sheet_pos"][0]
        # 2 node ids: deformable node: 0, collider_node: 1
        # however, we want to control all nodes, so make all of them deformable, but distinguish them in the last 2 one hot encodings
        node_id = torch.zeros(x.shape[2], 4)
        node_id[:, 0] = 1.0  # all nodes are deformable
        # cloth identifier
        node_id[0:len(init_pos), 2] = 1.0
        # sphere identifier
        node_id[len(init_pos):, 3] = 1.0
        current_h = node_id
        if self.config.material_properties:
            material_properties = material_properties.repeat(x.shape[2], 1)
            current_h = torch.cat([current_h, material_properties], dim=-1)
            h_description += self.mat_prop_description
        return current_h, h_description

    def get_material_properties(self, task_dict):
        normalized_ball_diameter = torch.tensor(
            [task_dict["params"]["ball_diameter"]]) / 60.0  # max diameter is 60, so this should be in [0, 1]
        ball_mass = torch.tensor([task_dict["params"]["ball_mass"]])
        youngs_modulus = torch.tensor([task_dict["params"]["youngs_modulus"]])
        tape_thickness = torch.tensor([task_dict["params"]["tape_thickness"]])
        # ym and thickness are very dependent on each other, since they both control the stiffness of the trampoline, so we normalize them together by their product, which is proportional to the bending stiffness of the trampoline
        Et = youngs_modulus * tape_thickness
        log_Et = torch.log(Et)
        log_m_over_Et = torch.log(ball_mass / Et)
        viscous_g = torch.tensor([task_dict["params"]["viscous"]["g"]])
        # viscous_k = torch.tensor([task_dict["params"]["viscous"]["k"]])
        log_g = torch.log(viscous_g)
        viscous_tau = torch.tensor([task_dict["params"]["viscous"]["tau"]])
        log_tau = torch.log(viscous_tau)
        material_properties = torch.cat([normalized_ball_diameter, log_Et, log_m_over_Et, log_g, log_tau], dim=-1)
        # v5 includes gravity perturbation, check if present and add it to material properties
        if self.has_gravity_perturbation:
            gravity_azimuth_deg = torch.tensor(
                [task_dict["params"]["gravity_azimuth_deg"]]) / 360.0  # max azimuth is 360, so this should be in [0, 1]
            log_gravity_tilt_deg = torch.log(
                torch.tensor([task_dict["params"]["gravity_tilt_deg"]]) + 1.0e-6)  # max values is 5
            material_properties = torch.cat([material_properties, gravity_azimuth_deg, log_gravity_tilt_deg], dim=-1)

        # # legacy mat props
        # material_properties = torch.cat(
        #     [normalized_ball_diameter, ball_mass, youngs_modulus, tape_thickness, viscous_g, viscous_k, viscous_tau],
        #     dim=-1)
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
        # # legacy mat props
        # return ["normalized_ball_diameter", "ball_mass", "youngs_modulus", "tape_thickness", "viscous_g",
        #                   "viscous_k", "viscous_tau"]

        mat_prop_desc = ["normalized_ball_diameter", "log_Et", "log_m_over_Et", "log_g", "log_tau"]
        if self.has_gravity_perturbation:
            mat_prop_desc += ["gravity_azimuth_deg", "log_gravity_tilt_deg"]
        return mat_prop_desc

    def get_batch_from_real_data(self, pc_data, pc_type, initial_sheet, initial_sphere, diameter, context_trajs, target_trajs,
                                 oracle=False, oracle_data={}):
        result = {}
        result["context_trajs"] = context_trajs
        result["target_trajs"] = target_trajs
        # mesh
        # repeat initial mesh and collider positions to get same shape.
        mesh_pos = initial_sheet.repeat(1, 25, 1, 1)
        collider_pos = initial_sphere.repeat(1, 25, 1, 1)
        mesh_pos = self.normalization(mesh_pos)
        collider_pos = self.normalization(collider_pos)
        x = torch.cat((mesh_pos, collider_pos), dim=2)
        h_description = ["one_hot", "one_hot", "mesh_identifier", "collider_identifier"]
        node_id = torch.zeros(x.shape[2], 4)
        node_id[:, 0] = 1.0  # all nodes are deformable
        # cloth identifier
        node_id[0:mesh_pos.shape[2], 2] = 1.0
        # sphere identifier
        node_id[mesh_pos.shape[2]:, 3] = 1.0
        h = node_id[None, : ,:].repeat(x.shape[0], 1, 1)
        v = self.compute_velocity(x) # should be all 0
        edge_indices = self.edge_indices[str(float(diameter))]
        edge_features = self.base_edge_features[str(float(diameter))]
        edge_features = edge_features[None, :, :].repeat(h.shape[0], 1, 1)
        init_r_ij = self.get_initial_rel_features(x, edge_indices)
        edge_features = torch.cat([edge_features, init_r_ij], dim=-1)
        edge_feature_description = ["one_hot"] * 4 + ["init_r_ij"] * 3

        if oracle:
            # guesstimate the materials as good as possible
            mass_dict = {
                1: 15.0,
                2: 116.0,
                3: 269.0,
                4: 520.0,
                5: 37.0,
                6: 69.0,
                7: 110.0,
                8: 162.0,
                9: 1.0,
                10: 4.0,
                11: 9.0,
                12: 1.0,
                13: 9.0,
                14: 21.0,
            }
            thickness_dict = {
                1: 0.381,
                3: 0.254,
                5: 0.1524,
            }
            normalized_ball_diameter = torch.tensor(diameter) / 60.0
            ball_mass = torch.tensor(mass_dict[oracle_data["sphere_id"]]) / 1000 # g to kg
            youngs_modulus = torch.tensor(3.0)  # average of dataset
            tape_thickness = torch.tensor(thickness_dict[oracle_data["latex"]])
            viscous_g = torch.tensor(0.1523514325)
            viscous_tau = torch.tensor(1.502004)
            Et = youngs_modulus * tape_thickness
            log_Et = torch.log(Et)
            log_m_over_Et = torch.log(ball_mass / Et)
            log_g = torch.log(viscous_g)
            log_tau = torch.log(viscous_tau)
            material_properties = torch.stack([
                normalized_ball_diameter,
                log_Et,
                log_m_over_Et,
                log_g,
                log_tau,
            ])
            # insert it into the h
            h = torch.cat([h, material_properties[None, None, :].repeat(h.shape[0], h.shape[1], 1)], dim=-1)

        result.update({
            "x": x,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
            "v": v,  # shape (num_trajs_in_task, traj_length, num_nodes, world_dim)
            "h": h,  # shape (num_trajs_in_task, num_nodes, node_feature_dim)
            "h_description": h_description,
            "edge_indices": edge_indices,  # shape (2, num_edges)
            "edge_features": edge_features,  # shape (num_trajs_in_task, num_edges, num_edge_features)
            "edge_feature_description": edge_feature_description,
            "meta_data": {
                "diameter": torch.tensor(diameter, dtype=torch.float32),
            }
        })
        # pc
        result["pc"] = self.normalization(pc_data)
        result["pc_color"] = pc_type[:, :, :, None]  # add feature dim

        # add batch dim to all tensors
        for key in result.keys():
            if key in ["h_description", "edge_feature_description", "meta_data"]:
                continue
            result[key] = result[key][None, ...]
        if "meta_data" in result:
            for key, value in result["meta_data"].items():
                result["meta_data"][key] = value[None, ...]
        result["h_description"] = [[desc] for desc in result["h_description"]]
        result["edge_feature_description"] = [[desc] for desc in result["edge_feature_description"]]
        return result



