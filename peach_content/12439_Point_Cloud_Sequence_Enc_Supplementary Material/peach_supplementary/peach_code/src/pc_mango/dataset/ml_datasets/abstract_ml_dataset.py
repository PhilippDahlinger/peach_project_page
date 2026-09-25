import random
from abc import ABC, abstractmethod
from typing import Tuple

import h5py
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
import fpsample

from pc_mango.dataset.util.util import hdf5_group_to_dict
from pc_mango.util.own_types import ConfigDict


class AbstractMLDataset(Dataset, ABC):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        self.config = config
        self.split = split
        root = config.root
        self.tasks = []

        # load scene encoder if used
        if self.config.pc_preprocessing.output_type == "scene_encoding":
            from pc_mango.network.pc_encoder import get_scene_encoder
            self.scene_encoder = get_scene_encoder().cuda()

        self._my_indices = self.get_splits()

        with h5py.File(root, "r") as hdf:
            try:
                self.global_data = hdf5_group_to_dict(hdf["global_data"])
            except KeyError:
                self.global_data = None  # no global data
            num_tasks_loaded = 0
            for task_key in tqdm(sorted(hdf.keys()), desc=f"Loading {split} dataset"):
                if task_key.startswith("task_"):
                    task_index = int(task_key[-3:])
                    if task_index in self._my_indices:
                        self.tasks.append(hdf5_group_to_dict(hdf[task_key]))
                        # also add a task key to the task dict for easier access later
                        self.tasks[-1]["task_key"] = task_key
                        num_tasks_loaded += 1
                        if self.config.get("dataset_size", None) is not None:
                            if num_tasks_loaded >= self.config.dataset_size:
                                break

        # load the pcd dataset if it is externalized
        if self.config.data_modalities is not None and "pc" in self.config.data_modalities and "external_pcd_root" in self.config and self.config.external_pcd_root is not None:
            print("Loading external point cloud dataset...")
            with h5py.File(self.config.external_pcd_root, "r") as pcd_hdf:
                for task in tqdm(self.tasks, desc="Loading external point clouds"):
                    for traj_key in sorted(task["trajs"].keys()):
                        traj = task["trajs"][traj_key]
                        pcd_data = hdf5_group_to_dict(pcd_hdf[f"{task["task_key"]}/trajs/{traj_key}/generated_pcd"])
                        traj["generated_pcd"] = pcd_data
                        if "generated_pcd_colors" in pcd_data:
                            traj["generated_pcd_colors"] = pcd_data["generated_pcd_colors"]
        # do the postprocessing of the pointclouds here, but only if the pc modality is used
        if "pc" in self.config.data_modalities:
            self.preprocess_pointclouds()
        if "sdf" in self.config.data_modalities:
            self.preprocess_sdf()
        self.random_traj_selection = config.random_traj_selection
        self._task_size = len(self.tasks[0]["trajs"])

        # standard training mode
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        if self.config.get("dataset_size", None) is None:
            return len(self.tasks)
        else:
            return min(self.config.dataset_size, len(self.tasks))

    def preprocess_sdf(self):
        with h5py.File(self.config.external_sdf_root, "r") as pcd_hdf:
            for task in tqdm(self.tasks, desc="Preprocessing sdf"):
                for traj_key in sorted(task["trajs"].keys()):
                    traj = task["trajs"][traj_key]
                    sdf_queries = hdf5_group_to_dict(pcd_hdf[f"{task["task_key"]}/trajs/{traj_key}/queries"])
                    sdf_labels = hdf5_group_to_dict(pcd_hdf[f"{task["task_key"]}/trajs/{traj_key}/sdf"])
                    max_timestep = -1
                    for t in sorted(sdf_queries.keys()):
                        int_t = int(t[-3:])
                        if int_t > max_timestep:
                            max_timestep = int_t
                    output_queries = []
                    output_sdf = []
                    # uses the same temporal stride as for the point clouds
                    for timestep in range(0, max_timestep + 1, self.config.sdf_preprocessing.temporal_stride):
                        queries = sdf_queries[f"ts_{timestep:03d}"]
                        occupancies = sdf_labels[f"ts_{timestep:03d}"]
                        output_queries.append(queries)
                        output_sdf.append(occupancies)

                    # transform the labels to tanh(sdf / scale)
                    output_sdf = torch.stack(output_sdf)
                    output_sdf = torch.tanh(output_sdf / self.config.sdf_preprocessing.scale_factor)
                    traj["queries"] = torch.stack(output_queries)  # shape (traj_length, num_queries, point_dim)
                    traj["sdf"] = output_sdf  # shape

    def preprocess_pointclouds(self):
        for task in tqdm(self.tasks, desc="Preprocessing pointclouds"):
            for _, traj in task["trajs"].items():
                if "generated_pcd" in traj:
                    if self.config.pc_preprocessing.output_type == "tensor":
                        pc, pc_color = self._pc_tensor_output_preproc(traj)
                        traj["processed_pc"] = pc
                        traj["processed_pc_color"] = pc_color
                    elif self.config.pc_preprocessing.output_type == "scene_encoding":
                        raise NotImplementedError("Deprecated Interface")
                    elif self.config.pc_preprocessing.output_type == "graph":
                        raise NotImplementedError("Graph output type not implemented yet")
                    # remove unprocessed pc
                    traj.pop("generated_pcd")

    def _pc_tensor_output_preproc(self, traj) -> torch.Tensor:
        """
        Output shape: (traj_length, num_points, point_dim)
        So for example (52, 2000, 3)
        :param traj:
        :return:
        """
        output_pc = []
        output_color = []
        # get the total number of time steps
        max_timestep = -1
        for t in sorted(traj["generated_pcd"]):
            int_t = int(t[-3:])
            if int_t > max_timestep:
                max_timestep = int_t

        for timestep in range(0, max_timestep + 1, self.config.pc_preprocessing.temporal_stride):
            pc = traj["generated_pcd"][f"pc_{timestep:03d}"]
            if "pc_node_type_name_key" in self.config:
                pc_node_type_name_key = self.config.pc_node_type_name_key
            else:
                pc_node_type_name_key = "pc_colors"
            try:
                color = traj["generated_pcd"][f"{pc_node_type_name_key}_{timestep:03d}"]
            except KeyError:
                # no color presented, use zeros
                color = torch.zeros((pc.shape[0], 1), dtype=torch.float32)
            if self.config.pc_preprocessing.num_subsample_points is not None:
                # FPS sampling
                num_points = self.config.pc_preprocessing.num_subsample_points
                if pc.shape[0] > num_points:
                    # only sample if there are more points than the target number, otherwise keep all points
                    pc_indices = fpsample.bucket_fps_kdtree_sampling(pc, num_points)
                    pc = pc[pc_indices]
                    color = color[pc_indices]
                # for debugging:
                # vis_pcd(pc, camera_eyes=None, mesh_pos=None)
            output_pc.append(pc)
            output_color.append(color)
        output_pc = torch.stack(output_pc)  # shape (traj_length, num_points, point_dim)
        output_color = torch.stack(output_color)  # shape (traj_length, num_points)
        return output_pc, output_color

    def _scene_encoding_output_preproc(self, traj) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_output = []
        feature_output = []
        for t in sorted(traj["generated_pcd"]):
            int_t = int(t[-3:])
            if int_t % self.config.pc_preprocessing.temporal_stride != 0:
                continue
            pc = traj["generated_pcd"][t].cuda()
            batch = torch.zeros(pc.shape[0], dtype=torch.long).cuda()
            x = None
            embeds, positions, batch_idx = self.scene_encoder(x, pc, batch)
            pos_output.append(positions.cpu())
            feature_output.append(embeds.cpu())
        pos_output = torch.stack(pos_output)  # shape (traj_length, num_anchor_points, 3)
        feature_output = torch.stack(feature_output)  # shape (traj_length, num_anchor_points, feature_dim)
        return pos_output, feature_output

    def sdf_subsample(self, sdfs, queries):
        # select num_query_points_per_ts random queries
        num_query_points_per_ts = self.config.sdf_preprocessing.num_query_points_per_ts
        indices = torch.randint(0, sdfs.shape[2], (sdfs.shape[0], sdfs.shape[1], num_query_points_per_ts))
        sdf_indices = indices.unsqueeze(-1).expand(-1, -1, -1, sdfs.size(-1))
        selecected_sdfs = torch.gather(sdfs, dim=2, index=sdf_indices)
        query_indices = indices.unsqueeze(-1).expand(-1, -1, -1, queries.size(-1))
        selected_queries = torch.gather(queries, dim=2, index=query_indices)

        return selecected_sdfs, selected_queries




    def get_splits(self):
        with h5py.File(self.config.root, "r") as hdf:
            split_type = self.config.get("split_type", "standard")
            match split_type:
                case "standard":
                    splits_group = hdf["splits"]
                case "ood":
                    splits_group = hdf["ood_splits"]
                case "small_data":
                    splits_group = hdf["small_data_splits"]
                case _:
                    raise ValueError(f"Unknown split type: {split_type}")

            return splits_group[f"{self.split}_indices"][:]

    @abstractmethod
    def get(self, task_idx):
        raise NotImplementedError

    @abstractmethod
    def get_faces(self):
        raise NotImplementedError

    def get_initial_rel_features(self, x, edge_index):
        row, col = edge_index
        if len(x.shape) == 4:
            r_ij = x[:, 0, row, :] - x[:, 0, col, :]  # relative position
        elif len(x.shape) == 2:
            r_ij = x[row, :] - x[col, :]  # relative position
        else:
            raise ValueError(f"Unknown shape of x: {x.shape}")
        return r_ij

    def get_context_target_trajs(self, task_size):
        if self.random_traj_selection:
            context_size = torch.randint(self.config.min_context_size, self.config.max_context_size + 1, (1,)).item()
            target_size = torch.randint(self.config.min_target_size, self.config.max_target_size + 1, (1,)).item()
        else:
            context_size = self.config.context_size
            target_size = self.config.target_size
        assert context_size <= task_size, "Context size must be smaller or equal to task size"
        assert target_size <= task_size, "Target size must be smaller or equal to task size"

        if self.random_traj_selection:
            # Perform sampling without replacement
            context_trajs = torch.multinomial(torch.ones(task_size), context_size, replacement=False)
            target_trajs = torch.multinomial(torch.ones(task_size), target_size, replacement=False)
        else:
            context_trajs = torch.arange(context_size)
            target_trajs = torch.arange(target_size)
        return context_trajs, target_trajs

    def compute_velocity(self, x):
        v = x[:, 1:, :, :] - x[:, :-1, :, :]
        if self.config.initial_vel == "zero":
            # zero padding for step 0 velocity
            v = torch.cat((torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3])), v), dim=1)
        elif self.config.initial_vel == "constant":
            # constant padding for step 0 velocity: velocity is the same as the velocity from step 1
            v = torch.cat((v[:, 0:1, :, :], v), dim=1)
        else:
            raise ValueError(f"Unknown initial velocity type: {self.config.initial_vel}")
        return v


    def _raw_traj_length(self):
        # length without cut_traj and stride
        raise NotImplementedError


    @property
    def traj_length(self):
        #  length with cut_traj and stride
        if self.config.get("cut_length", None) is not None:
            if self.config.cut_length == -1:
                traj_length = self._raw_traj_length()
            else:
                traj_length = min(self._raw_traj_length(), self.config.cut_length)
        else:
            traj_length = self._raw_traj_length()
        if self.config.get("temporal_stride", None) is not None:
            traj_length = (traj_length - 1) // self.config.temporal_stride + 1
        return traj_length

    @property
    def task_size(self):
        return self._task_size

    @property
    def mat_prop_description(self):
        raise NotImplementedError