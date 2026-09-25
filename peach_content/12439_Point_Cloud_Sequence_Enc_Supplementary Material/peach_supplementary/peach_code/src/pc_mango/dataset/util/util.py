import sys
from typing import Union, List, Optional

import h5py
import numpy as np
import torch.nn



def get_mesh_connectivity(dataset, meta_data=None):
    from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset
    from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset
    from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
    from pc_mango.dataset.ml_datasets.bbv import BBVDataset
    if isinstance(dataset, SheetDeformationDataset):
        return dataset.global_data["cells"]
    elif isinstance(dataset, DeformableBlockDataset):
        return dataset.global_data["cell_indices"]["mesh_faces"]
    elif isinstance(dataset, TrampolineDataset):
        return dataset.global_data["sheet_cells"]
    elif isinstance(dataset, BBVDataset):
        assert meta_data is not None, "Add metadata of current batch to this function call to get the mesh connectivity for BBV"
        return meta_data["faces"].detach().cpu()
    else:
        raise NotImplementedError("Unknown dataset type for exporting mesh connectivity.")


def get_collider_connectivity(dataset, meta_data):
    from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset
    from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset
    from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
    from pc_mango.dataset.ml_datasets.bbv import BBVDataset
    if isinstance(dataset, SheetDeformationDataset):
        return None  # no collider in sheet deformation
    elif isinstance(dataset, DeformableBlockDataset):
        return dataset.global_data["cell_indices"]["collider_faces"]
    elif isinstance(dataset, TrampolineDataset):
        return dataset.faces["ball_faces"][meta_data["diameter"]]
    elif isinstance(dataset, BBVDataset):
        return None  # no collider in BBV
    else:
        raise NotImplementedError("Unknown dataset type for exporting mesh connectivity.")


def hdf5_group_to_dict(group):
    """
    Convert an HDF5 group of datasets to a dictionary of NumPy arrays.

    Parameters:
    - group (h5py.Group): The HDF5 group containing datasets.

    Returns:
    - dict: A dictionary where keys are dataset names and values are NumPy arrays.
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            # ---- STRING DATASETS ----
            if h5py.check_string_dtype(item.dtype) is not None:
                result[key] = item.asstr()[()]  # returns Python str or list[str]

            # ---- NUMERIC DATASETS ----
            else:
                shape = item.shape

                if len(shape) != 0:
                    result[key] = torch.from_numpy(item[:])
                else:
                    result[key] = torch.from_numpy(np.array(item))

                if result[key].dtype == torch.float64:
                    result[key] = result[key].to(torch.float32)
        elif isinstance(item, h5py.Group):
            # If it's a nested group, recursively process it
            result[key] = hdf5_group_to_dict(item)
    return result


def convert_history_pos_to_history_vel_features(history_pos, current_pos):
    """
    Convert a history of positions to a history of velocities by calculating the difference between consecutive positions.

    Parameters:
    - history_pos (torch.Tensor): A tensor of shape (T, N, D) containing the history of positions. (oldest to newest)
    - current_pos (torch.Tensor): A tensor of shape (N, D) containing the current positions.

    Returns:
    - torch.Tensor: A tensor of shape (N, D * T) containing the history of velocities.
    """
    if history_pos.shape[0] == 0:
        return torch.zeros((current_pos.shape[0], 0))
    history_pos = torch.cat([history_pos, current_pos.unsqueeze(0)], dim=0)
    history_vel = history_pos[1:, :] - history_pos[:-1, :]
    # shape (T, N , D)
    # flatten
    history_vel = history_vel.view(history_vel.shape[1], -1)
    return history_vel


def add_noise(noise_scale, pos, history_pos, deformable_mask):
    if noise_scale <= 0:
        return pos, history_pos
    # create  the deformable mask by hand since no batch is created yet
    total_history_length = len(history_pos) + 1  # current position as well
    history_noise_scale = noise_scale / torch.sqrt(torch.tensor(total_history_length, dtype=torch.float32))
    len_deformable_nodes = torch.sum(deformable_mask)
    num_pos_features = pos.shape[1]
    noise = torch.randn(total_history_length, len_deformable_nodes, num_pos_features) * history_noise_scale
    # cumulative sum -> random walk
    noise = noise.cumsum(dim=0)
    history_noise = noise[:-1, :, :]
    current_noise = noise[-1, :, :]
    history_pos[:, deformable_mask, :] += history_noise
    pos[deformable_mask] += current_noise
    return pos, history_pos
