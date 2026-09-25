import json
import os
from abc import abstractmethod

import h5py
import numpy as np
import torch
import torch_geometric.data
from torch_geometric.data import Data

from pc_mango.ggns.src.utils.get_connectivity_setting import get_connectivity_setting
from pc_mango.ggns.src.utils.hdf5_dataset_utils import build_dataset_for_split_hdf5
from pc_mango.ggns.src.utils.data_utils import convert_trajectory_to_data_list


_worker_file = None

def worker_init_fn(worker_id):
    global _worker_file
    worker_info = torch.utils.data.get_worker_info()
    cache_path = worker_info.dataset.cache_path
    _worker_file = h5py.File(cache_path, "r", swmr=True)

class AbstractStepGGNSDataset(torch_geometric.data.Dataset):
    """Abstract base class for GGNS step datasets.

    Handles config reading, connectivity setting parsing, HDF5 loading via the
    GGNS utility pipeline, and trajectory flattening. Subclasses only need to
    provide dataset-specific normalization parameters.

    Returns PyG Data objects with .x, .pos, .edge_index, .edge_attr, .y, .y_old,
    .node_type, .edge_type — the format expected by HomoGNN.
    """

    def __init__(self, config, split="train"):
        self.cache_path = None
        cache_path = config.get("cache_path", None)
        if cache_path is not None:
            self.cache_path = os.path.join(cache_path, f"cache_{split}.hdf5")
            # create directory if it does not exist
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            # check if file exists
            if not os.path.isfile(self.cache_path):
                self._len = 0
            else:
                with h5py.File(self.cache_path, "r") as f:
                    self._len = len(f)

        self.config = config

        hdf5_path = config.root
        connectivity_setting = config.connectivity_setting
        use_mesh_coordinates = config.get("use_mesh_coordinates", True)
        use_color = config.get("use_colors", False)
        num_tasks = config.get("dataset_size", None)
        pcd_subsample_points = config.get("pcd_subsample_points", None)
        pcd_hdf5_path = config.get("pcd_hdf5_path", None)
        mesh_key = config.get("mesh_key", "mesh_pos")
        collider_key = config.get("collider_key", "collider_pos")
        mesh_faces_key = config.get("mesh_faces_key", None)
        cut_length = config.get("cut_length", -1)
        temporal_strid = config.get("temporal_stride", 1)
        target_test_trajs = config.get("target_test_trajs", None)
        self.predict_collider = config.get("predict_collider", False)

        edge_radius_dict, input_timestep, self.euclidian_distance, _ = \
            get_connectivity_setting(connectivity_setting)
        self.is_3d = config.get("is_3d", False)

        normalization_params = self.get_normalization_params()

        self.save_to_disk = config.get("save_to_disk", False)

        if self.save_to_disk or self.cache_path is None:
            # only then load the trajectory again
            trajectory_list = build_dataset_for_split_hdf5(
                hdf5_path=hdf5_path,
                path=".",
                split=split,
                edge_radius_dict=edge_radius_dict,
                device="cpu",
                input_timestep=input_timestep,
                use_mesh_coordinates=use_mesh_coordinates,
                hetero=False,
                raw=False,
                use_color=use_color,
                num_tasks=num_tasks,
                normalization_params=normalization_params,
                extract_2d=not self.is_3d,
                pcd_subsample_points=pcd_subsample_points,
                pcd_hdf5_path=pcd_hdf5_path,
                mesh_key=mesh_key,
                collider_key=collider_key,
                mesh_faces_key=mesh_faces_key,
                cut_length=cut_length,
                temporal_stride=temporal_strid,
                postprocess_pc=self.postprocess_pc,
                target_test_trajs=target_test_trajs,
                predict_collider=self.predict_collider,
                connection_node_indices_path=config.get("connection_node_indices_path", None),
                save_to_disk=self.save_to_disk,
                output_path=self.cache_path
            )

            self.data_list = convert_trajectory_to_data_list(trajectory_list)
        super().__init__()


    @abstractmethod
    def get_normalization_params(self):
        """Return dict with keys: min_x, max_x, normalize_mesh, normalize_collider, normalize_pcd.
        Return None to skip all normalization."""
        ...

    def postprocess_pc(self, pc):
        # if needed implement in subclass
        return pc

    def len(self):
        if self.cache_path is None:
            return len(self.data_list)
        else:
            return self._len

    def get(self, idx):
        if self.cache_path is None:
            return self.data_list[idx]
        else:
            f = _worker_file if _worker_file is not None else h5py.File(self.cache_path, "r")
            grp = f[str(idx)]
            tensors = {k: torch.from_numpy(np.array(grp[k])) for k in grp.keys()}
            scalars = {}
            for key, val in grp.attrs.items():
                try:
                    scalars[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    scalars[key] = val
            return Data(**tensors, **scalars)
