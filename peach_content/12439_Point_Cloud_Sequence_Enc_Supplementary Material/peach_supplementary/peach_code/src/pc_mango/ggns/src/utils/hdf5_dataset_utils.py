"""
HDF5 Data Loader for PC-MANGO Dataset
Adapts pc_mango HDF5 format to GGNS data format
"""

import contextlib
import json
import os

import fpsample
import h5py
import torch
import numpy as np
from sympy.abc import lamda
from tqdm import tqdm
from typing import Dict, Tuple, List

from pc_mango.dataset.edges.edge_indices import get_edge_indices
from pc_mango.dataset.util.util import hdf5_group_to_dict
from pc_mango.ggns.src.utils.data_utils import convert_trajectory_to_data_list
from pc_mango.ggns.util.Types import Data
import pc_mango.ggns.src.utils.graph_utils as graph_utils


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling for point cloud downsampling.

    Args:
        points: Point cloud array of shape (N, 3)
        num_samples: Number of samples to select

    Returns:
        Indices of selected points
    """
    n_points = points.shape[0]

    # Start with a random point
    selected_idx = [np.random.randint(n_points)]

    # Compute distances to all points
    distances = np.full(n_points, np.inf)

    for _ in range(1, num_samples):
        # Update distances to nearest selected point
        last_selected = points[selected_idx[-1]]
        dist_to_last = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Select farthest point
        next_idx = np.argmax(distances)
        selected_idx.append(next_idx)

    return np.array(selected_idx)


def prepare_hdf5_trajectory(hdf5_traj_group,
                            mesh_mesh_indices,
                            use_color: bool = False,
                            normalization_params=None,
                            extract_2d: bool = True,
                            pcd_subsample_points: int = None,
                            mesh_key: str = "mesh_pos",
                            collider_key: str = "collider_pos",
                            pcd_hdf5_traj_group=None,
                            cut_length: int = -1,
                            temporal_stride: int = 1,
                            postprocess_pc=None) -> Tuple:
    """
    Prepares data from HDF5 trajectory group for GGNS format.
    Returns a tuple compatible with prepare_data_for_trajectory.

    Args:
        hdf5_traj_group: HDF5 group containing a single trajectory
        mesh_mesh_indices: Precomputed edge indices for the plate mesh
        use_color: If True, use point cloud colors (if available)
        normalization_params: Dict with min_x, max_x, normalize_mesh, normalize_collider,
            normalize_pcd flags. Or None to skip normalization.
        extract_2d: If True, extract only 2D coordinates from 3D point clouds
        pcd_subsample_points: Max number of point cloud points (FPS downsampling). None = keep all.
        mesh_key: HDF5 key for mesh positions (default 'mesh_pos')
        collider_key: HDF5 key for collider positions (default 'collider_pos')
        pcd_hdf5_traj_group: Optional external HDF5 group for point cloud data

    Returns:
        Tuple: (pcd_points, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, None)
    """
    # Load mesh and collider positions (already all timesteps)
    mesh_pos = hdf5_traj_group[mesh_key][()]
    if collider_key in hdf5_traj_group:
        collider_pos = hdf5_traj_group[collider_key][()]
    else:
        # No collider: empty (T, 0, D) array — downstream code handles 0-node collider gracefully
        D = mesh_pos.shape[2]
        collider_pos = np.zeros((mesh_pos.shape[0], 0, D))

    # cut mesh and collider pos
    if cut_length > 0:
        mesh_pos = mesh_pos[:cut_length, :, :]
        collider_pos = collider_pos[:cut_length, :, :]
    if temporal_stride > 1:
        mesh_pos = mesh_pos[::temporal_stride, :, :]
        collider_pos = collider_pos[::temporal_stride, :, :]

    # Load point clouds per timestep (from external file if provided)
    pcd_source = pcd_hdf5_traj_group if pcd_hdf5_traj_group is not None else hdf5_traj_group
    pcd_group = pcd_source['generated_pcd']

    num_timesteps = mesh_pos.shape[0]

    pcd_points = []
    pcd_colors = []

    for t in range(0, num_timesteps * temporal_stride, temporal_stride):
        pc_key = f'pc_{t:03d}'
        if pc_key in pcd_group:
            # Load point cloud
            pc = pcd_group[pc_key][()]  # (1000, 3) or (1000, D)

            # Extract 2D if all Z coordinates are near zero
            if extract_2d and pc.shape[1] >= 3:
                z_range = pc[:, 2].max() - pc[:, 2].min()
                if z_range < 1e-4:  # Essentially 2D (all Z coords are same)
                    pc = pc[:, :2]  # Keep only X, Y

            # Load colors before subsampling so same indices apply
            pc_color = None
            if use_color:
                pc_color_key = f'pc_colors_{t:03d}'
                if pc_color_key in pcd_group:
                    pc_color = pcd_group[pc_color_key][()]
                else:
                    pc_color = np.ones_like(pc)

            # Subsample if point cloud exceeds target size
            if pcd_subsample_points is not None and pc.shape[0] > pcd_subsample_points:
                indices = fpsample.fps_sampling(pc, pcd_subsample_points)
                pc = pc[indices]
                if pc_color is not None:
                    pc_color = pc_color[indices]

            pcd_points.append(pc)
            pcd_colors.append(pc_color)

    # Normalize positions per data type
    if normalization_params is not None:
        min_x = normalization_params["min_x"]
        max_x = normalization_params["max_x"]
        if normalization_params.get("normalize_mesh", True):
            mesh_pos = (mesh_pos - min_x) / (max_x - min_x)
        if normalization_params.get("normalize_collider", True):
            collider_pos = (collider_pos - min_x) / (max_x - min_x)
        if normalization_params.get("normalize_pcd", False):
            pcd_points = [(pc - min_x) / (max_x - min_x) for pc in pcd_points]

    if postprocess_pc is not None:
        pcd_points = [postprocess_pc(pc) for pc in pcd_points]

    # Create edge index for mesh (precomputed, fixed topology)
    edge_index_grid = mesh_mesh_indices
    if "params" in hdf5_traj_group.keys() and "force1" in hdf5_traj_group["params"].keys():
        # sheet deformation dataset, have to add the force as a node feature, so load it here
        force1 = hdf5_group_to_dict(hdf5_traj_group["params"]["force1"])
        force2 = hdf5_group_to_dict(hdf5_traj_group["params"]["force2"])
        forces = {
            "force1": force1,
            "force2": force2,
        }
    else:
        forces = None


    # Return in same format as prepare_data_from_sofa (poisson_ratio always None)
    data = (pcd_points, collider_pos, mesh_pos, edge_index_grid, pcd_colors, None, forces)
    return data


def create_mesh_edges_plate(h: int, w: int) -> np.ndarray:
    """
    Create edge connectivity for a structured mesh (plate).
    Creates edges between neighboring nodes in a grid.

    Args:
        h: Height (number of nodes vertically)
        w: Width (number of nodes horizontally)

    Returns:
        Edge index array of shape (num_edges, 2)
    """
    edges = []

    # Create a grid mapping: node index = row * w + col
    for row in range(h):
        for col in range(w):
            node_idx = row * w + col

            # Right neighbor
            if col < w - 1:
                neighbor_idx = row * w + (col + 1)
                edges.append([node_idx, neighbor_idx])

            # Bottom neighbor
            if row < h - 1:
                neighbor_idx = (row + 1) * w + col
                edges.append([node_idx, neighbor_idx])

    return np.array(edges)


def prepare_data_for_trajectory_from_hdf5(data: Tuple,
                                          timestep: int,
                                          input_timestep: str = 't+1',
                                          use_color: bool = False,
                                          predict_collider=False) -> Tuple:
    """
    Convert prepared HDF5 data tuple into per-timestep data format.
    This matches the behavior of prepare_data_for_trajectory from dataset_utils.

    Args:
        data: Tuple from prepare_hdf5_trajectory
        timestep: Current timestep index
        input_timestep: 't' or 't+1'
        use_color: Whether to use colors

    Returns:
        Tuple of tensors for a single timestep
    """
    pcd_points, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, _, forces = data

    if input_timestep == 't+1':
        raise NotImplementedError()
        grid_positions = torch.tensor(pcd_points[timestep + 1])
        collider_positions = torch.tensor(nodes_collider[timestep + 1])
        mesh_positions = torch.tensor(nodes_grid[timestep])
        mesh_edge_index = edge_index_grid.detach().clone().long()
        label = torch.tensor(nodes_grid[timestep + 1])
        initial_mesh_positions = torch.tensor(nodes_grid[0])
        if use_color and pcd_colors[timestep + 1] is not None:
            grid_colors = torch.tensor(pcd_colors[timestep + 1])
        else:
            grid_colors = None
    elif input_timestep == 't':
        grid_positions = torch.tensor(pcd_points[timestep], dtype=torch.float64)
        collider_positions = torch.tensor(nodes_collider[timestep], dtype=torch.float64)
        mesh_positions = torch.tensor(nodes_grid[timestep], dtype=torch.float64)
        mesh_edge_index = edge_index_grid.detach().clone().long()
        label = torch.tensor(nodes_grid[timestep + 1], dtype=torch.float64)
        initial_mesh_positions = torch.tensor(nodes_grid[0], dtype=torch.float64)
        prev_mesh = torch.tensor(nodes_grid[timestep - 1], dtype=torch.float64) if timestep > 0 else initial_mesh_positions
        prev_collider = torch.tensor(nodes_collider[timestep - 1], dtype=torch.float64) if timestep > 0 else torch.tensor(nodes_collider[0], dtype=torch.float64)
        mesh_vel = mesh_positions - prev_mesh
        collider_vel = collider_positions - prev_collider
        pcd_vel = torch.zeros_like(grid_positions)
        combined_vel = torch.cat((pcd_vel, collider_vel, mesh_vel), dim=0)

        if predict_collider:
            # combined with collider
            initial_collider_positions = torch.tensor(nodes_collider[0], dtype=torch.float64)
            initial_mesh_positions = torch.cat([initial_collider_positions, initial_mesh_positions], dim=0)
            collider_label = torch.tensor(nodes_collider[timestep + 1], dtype=torch.float64)
            label = torch.cat([collider_label, label], dim=0)
        if use_color and pcd_colors[timestep] is not None:
            grid_colors = torch.tensor(pcd_colors[timestep])
        else:
            grid_colors = None
    else:
        raise ValueError("input_timestep can only be 't' or 't+1'")

    data_timestep = grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions.float(), None, forces, combined_vel

    return data_timestep


def build_dataset_for_split_hdf5(hdf5_path: str,
                                 path: str,
                                 split: str,
                                 edge_radius_dict: Dict,
                                 device,
                                 input_timestep: str,
                                 use_mesh_coordinates: bool,
                                 hetero: bool = False,
                                 raw: bool = False,
                                 use_color: bool = False,
                                 num_tasks: int = None,
                                 normalization_params=None,
                                 extract_2d: bool = True,
                                 pcd_subsample_points: int = None,
                                 pcd_hdf5_path: str = None,
                                 mesh_key: str = "mesh_pos",
                                 collider_key: str = "collider_pos",
                                 cut_length=-1,
                                 temporal_stride=1,
                                 mesh_faces_key: str = None,
                                 postprocess_pc=None,
                                 target_test_trajs=None,
                                 predict_collider=False,
                                 connection_node_indices_path=None,
                                 save_to_disk=False,
                                 output_path=None) -> List:
    """
    Builds dataset from HDF5 file in the format expected by GGNS.

    Args:
        hdf5_path: Path to the HDF5 file
        path: Working directory (for compatibility)
        split: Dataset split ('train', 'val', 'test')
        edge_radius_dict: Dictionary containing edge radii for graph construction
        device: Device to place data on ('cpu' or 'cuda')
        input_timestep: Use data from 't' or 't+1'
        use_mesh_coordinates: Whether to use mesh coordinates in edges
        hetero: Whether to use heterogeneous graphs
        raw: Whether to create raw graphs without edges
        use_color: Whether to use point cloud colors
        num_tasks: Limit number of tasks to load (for testing). If None, load all.
        normalization_params: Dict with per-type normalization flags, or None
        extract_2d: If True, extract only 2D coordinates from 3D point clouds
        pcd_subsample_points: Max number of point cloud points (FPS downsampling). None = keep all.
        pcd_hdf5_path: Optional path to external HDF5 file for point cloud data
        mesh_key: HDF5 key for mesh positions (default 'mesh_pos')
        collider_key: HDF5 key for collider positions (default 'collider_pos')
        mesh_faces_key: Explicit key in global_data for mesh faces. None = auto-detect.
        postprocess_pc: function to call after a pc is loaded, with signature (pc: np.ndarray) -> np.ndarray.
        target_test_trajs: if eval split, load only those trajs. (keep it consistent to the eval from the ml dataset)

    Returns:
        trajectory_list: List of trajectories, each trajectory is a list of PyG Data objects
    """

    print(f"Generating {split} data from HDF5")

    with contextlib.ExitStack() as stack:
        hdf = stack.enter_context(h5py.File(hdf5_path, 'r'))
        pcd_hdf = stack.enter_context(h5py.File(pcd_hdf5_path, 'r')) if pcd_hdf5_path else None

        # Map eval to val (HDF5 uses 'val_indices' instead of 'eval_indices')
        hdf5_split = 'val' if split == 'eval' else split

        # Get split indices
        split_indices = hdf['splits'][f'{hdf5_split}_indices'][()]

        # Determine which tasks and trajectories to use
        task_traj_pairs = []
        counted_tasks = 0

        for task_key in tqdm(sorted(hdf.keys()), desc=f"Loading {split} dataset"):
            if task_key.startswith("task_"):
                task_idx = int(task_key[-3:])
                if task_idx in split_indices:
                    task_group = hdf[task_key]
                    traj_keys = sorted(task_group['trajs'].keys())
                    for traj_idx, traj_key in enumerate(sorted(traj_keys)):
                        # traj_idx = int(traj_key[-3:])
                        if hdf5_split != 'train' and target_test_trajs is not None and traj_idx not in target_test_trajs:
                            continue
                        task_traj_pairs.append((task_idx, traj_key))
                    counted_tasks += 1
                    if num_tasks is not None and counted_tasks >= num_tasks:
                        break

        # Load mesh faces from global_data
        global_data = hdf['global_data']
        if mesh_faces_key is not None:
            mesh_faces = {"mesh": global_data[mesh_faces_key][()]}
            if "ball_30.0_cells" in global_data.keys():
                # this is trampoline, we have to load all ball faces and merge them...
                for key in ['ball_20.0_cells', 'ball_30.0_cells', 'ball_35.0_cells', 'ball_40.0_cells', 'ball_45.0_cells',
                            'ball_50.0_cells', 'ball_55.0_cells', 'ball_60.0_cells']:
                    mesh_faces[key] = global_data[key][()]
        elif 'cell_indices' in global_data and 'mesh_faces' in global_data['cell_indices']:
            mesh_faces = {"mesh": global_data['cell_indices']['mesh_faces'][()]}
        elif 'cells' in global_data:
            mesh_faces = {"mesh": global_data['cells'][()]}
        else:
            raise KeyError("Could not find mesh faces in global_data. "
                           "Expected 'cell_indices/mesh_faces' or 'cells'.")

        mesh_mesh_indices_dict = {
            key: get_edge_indices(mesh_faces[key]) for key in mesh_faces.keys()
        }
        if connection_node_indices_path is not None:
            with open(connection_node_indices_path, "r") as f:
                connection_nodes = json.load(f)
        else:
            connection_nodes = None

        trajectory_list = []

        if save_to_disk:
            print(f"saving to disk to file {output_path}")
            # check if file exists, if yes, delete it to avoid appending to old data
            if os.path.exists(output_path):
                os.remove(output_path)
        for task_idx, traj_key in tqdm(task_traj_pairs):
            task_key = f'task_{task_idx:03d}'

            if task_key not in hdf:
                continue

            task_group = hdf[task_key]
            if traj_key not in task_group['trajs']:
                continue

            hdf5_traj = task_group['trajs'][traj_key]

            # Resolve PCD traj group from external file if provided
            pcd_traj_group = None
            if pcd_hdf is not None:
                if task_key in pcd_hdf and traj_key in pcd_hdf[task_key]['trajs']:
                    pcd_traj_group = pcd_hdf[task_key]['trajs'][traj_key]

            if len(mesh_mesh_indices_dict) == 1:
                # get mesh indices, everything is normal
                mesh_mesh_indices = mesh_mesh_indices_dict["mesh"]
                ball_key = None
            else:
                # collider comes first before mesh in this implementation, so shift mesh nodes up, and combine with correct ball
                ball_diameter = task_group["params"]["ball_diameter"][()]
                ball_key = f"ball_{ball_diameter}_cells"
                ball_indices = mesh_mesh_indices_dict[ball_key]
                mesh_indices = mesh_mesh_indices_dict["mesh"]
                # shift mesh indices by number of collider nodes (which is the same as number of ball nodes)
                num_collider_nodes = ball_indices.max() + 1
                shifted_mesh_indices = mesh_indices + num_collider_nodes
                # need to combine the 2 meshes together
                assert connection_nodes is not None
                mesh_connection_nodes = torch.tensor(connection_nodes["sheet"]) + num_collider_nodes
                collider_connection_nodes = torch.tensor(connection_nodes[str(ball_diameter)])
                mesh_col_indices = torch.cartesian_prod(mesh_connection_nodes, collider_connection_nodes).T
                col_mesh_indices = torch.cartesian_prod(collider_connection_nodes, mesh_connection_nodes).T
                # combine
                mesh_mesh_indices = torch.concatenate(
                    [ball_indices, shifted_mesh_indices, mesh_col_indices, col_mesh_indices], dim=1)

            # Prepare trajectory data
            trajectory = prepare_hdf5_trajectory(hdf5_traj, mesh_mesh_indices,
                                                 use_color=use_color,
                                                 normalization_params=normalization_params,
                                                 extract_2d=extract_2d,
                                                 pcd_subsample_points=pcd_subsample_points,
                                                 mesh_key=mesh_key,
                                                 collider_key=collider_key,
                                                 pcd_hdf5_traj_group=pcd_traj_group,
                                                 cut_length=cut_length,
                                                 temporal_stride=temporal_stride,
                                                 postprocess_pc=postprocess_pc)

            data_list = []
            rollout_length = trajectory[2].shape[0]  # mesh position

            for timestep in range(rollout_length - 1):
                if raw:
                    raise NotImplementedError("Do not use this branch")
                else:
                    # Get trajectory data for current timestep
                    data_timestep = prepare_data_for_trajectory_from_hdf5(trajectory, timestep,
                                                                          input_timestep=input_timestep,
                                                                          use_color=use_color,
                                                                          predict_collider=predict_collider)

                    # Create nearest neighbor graph
                    if hetero:
                        raise NotImplementedError("Do not use this branch")
                        data = graph_utils.create_hetero_graph_from_raw(data_timestep,
                                                                        edge_radius_dict=edge_radius_dict,
                                                                        output_device=device,
                                                                        use_mesh_coordinates=use_mesh_coordinates)
                    else:
                        data = graph_utils.create_graph_from_raw(data_timestep,
                                                                 edge_radius_dict=edge_radius_dict,
                                                                 output_device=device,
                                                                 use_mesh_coordinates=use_mesh_coordinates,
                                                                 predict_collider=predict_collider)
                # add task and traj index
                data.task_idx = task_idx
                data.traj_idx = int(traj_key[-3:])
                data_list.append(data)
                # if val or test: stop now since we only need the first timestep for evaluation
                # however, add the complete position data for all timesteps (collider and deformable)
                # to have a ground truth and to insert collider position if needed
                if split in ['val', 'eval', 'test']:
                    # also add batch dimension so for pyg...
                    data["collider_traj"] = torch.tensor(trajectory[1])[None, :, :, :].float()
                    data["mesh_traj"] = torch.tensor(trajectory[2])[None, :, :, :].float()
                    pcds = torch.stack([torch.tensor(pcd) for pcd in trajectory[0]], dim=0).float()
                    data["pcd_traj"] = pcds[None, :, :, :]
                    data["mesh_mesh_indices"] = mesh_mesh_indices
                    data["vis_mesh_faces"] = torch.tensor(mesh_faces["mesh"])[None, :, :]
                    if ball_key is not None:
                        data["vis_collider_faces"] = torch.tensor(mesh_faces[ball_key])[
                            None, :, :] if ball_key in mesh_faces else None
                    else:
                        # GGNS does not use collider faces but they are in the hdf5
                        # check for deformable block, only here a collider is present
                        if "collider_faces" in global_data.get("cell_indices", {}):
                            data["vis_collider_faces"] = \
                            torch.tensor(global_data["cell_indices"]["collider_faces"][()])[None, :, :]
                        else:
                            data["vis_collider_faces"] = None
                    break

            if save_to_disk:
                output = convert_trajectory_to_data_list([data_list])
                append_graphs_to_hdf5(output_path, output)
            else:
                trajectory_list.append(data_list)

    return trajectory_list


def append_graphs_to_hdf5(path: str, graphs: list):
    with h5py.File(path, "a") as f:
        start_idx = len(f)
        for i, data in enumerate(graphs):
            grp = f.create_group(str(start_idx + i))
            for key in data.keys():
                val = data[key]
                if isinstance(val, torch.Tensor):
                    grp.create_dataset(key, data=val.numpy())
                elif isinstance(val, (int, float)):
                    grp.attrs[key] = val
                elif isinstance(val, list):
                    grp.attrs[key] = json.dumps(val)  # serialize as JSON string