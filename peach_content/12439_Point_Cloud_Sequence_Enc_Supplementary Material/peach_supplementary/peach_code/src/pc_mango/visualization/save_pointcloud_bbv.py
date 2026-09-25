import re
import h5py
import numpy as np
import meshio
from pathlib import Path

import torch
from torch_geometric.nn import fps

from pc_mango.dataset.util.util import hdf5_group_to_dict


def dict_pointclouds_to_array(pc: dict) -> np.ndarray:
    """
    Convert dict { 'pc_000': (D,3), 'pc_001': (D,3), ... } to array (T,D,3).
    """
    # Extract (timestep_index, key) pairs
    pairs = []
    for k in pc.keys():
        m = re.search(r"(\d+)$", k)  # match trailing digits
        if not m:
            raise ValueError(f"Key '{k}' does not end with digits (expected pc_000 etc.)")
        idx = int(m.group(1))
        pairs.append((idx, k))

    # Sort by timestep index
    pairs.sort(key=lambda x: x[0])

    # Collect arrays in sorted order
    arrs = [np.asarray(pc[k]) for _, k in pairs]

    # Sanity checks
    # D = arrs[0].shape[0]
    # if arrs[0].shape[1] != 3:
    #     raise ValueError(f"Expected shape (D,3) but got {arrs[0].shape}")
    #
    # for i, a in enumerate(arrs):
    #     if a.shape != (D, 3):
    #         raise ValueError(f"Shape mismatch at step {i}: expected {(D,3)}, got {a.shape}")

    # Stack into (T,D,3)
    return np.stack(arrs, axis=0)

def write_pointcloud_timeseries_pvd(points_ts, out_dir, name="pc", point_data=None):
    import numpy as np
    import meshio
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_ts = np.asarray(points_ts)
    T, N, D = points_ts.shape
    assert D == 3, "Use shape (T,N,3). If 2D, add z=0."

    timesteps = np.arange(T)
    entries = []
    for k, t in enumerate(timesteps):
        pts = points_ts[k]
        cells = [("vertex", np.arange(N).reshape(-1, 1))]

        pd = None
        if point_data is not None:
            pd = {key: np.asarray(val[k]) for key, val in point_data.items()}

        mesh = meshio.Mesh(points=pts, cells=cells, point_data=pd)

        step_path = out_dir / f"{name}_{k:05d}.vtu"   # ✅ .vtu not .vtp
        meshio.write(step_path, mesh)
        entries.append((float(t), step_path.name))

    pvd_path = out_dir / f"{name}.pvd"
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for t, fname in entries:
            f.write(f'    <DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
    print(f"Wrote: pvd file to {pvd_path} (+ {T} .vtu files)")


    return pvd_path


def normalization(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)


if __name__ == "__main__":
    node_type_id = "node_type"  # or "color" for older datasets, select accordingly
    cut_length = None  # or None for full length, select accordingly
    min_x = 0.0
    max_x = 20.0
    apply_normalization = True  # set to True if the point clouds are not already normalized, otherwise False
    temporal_stride = 1
    task_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # from the test set
    traj_indices = [9]
    if root_pcd is None:
        root_pcd = root
    for task_id in task_ids:
        with h5py.File(root, "r") as hdf:
            num_tasks_loaded = 0
            remapped_task_id = sorted(hdf["splits"]["test_indices"][:])[task_id]
        with h5py.File(root_pcd, "r") as hdf:
            for task_key in sorted(hdf.keys()):
                if task_key.startswith("task_"):
                    task_index = int(task_key[-3:])
                    if task_index == remapped_task_id:
                        task = hdf5_group_to_dict(hdf[task_key])
                        for traj_idx in traj_indices:
                            for traj_key in sorted(task["trajs"].keys()):
                                if int(traj_key[-3:]) == traj_idx:
                                    traj = task["trajs"][traj_key]
                                    pc_and_color = traj["generated_pcd"]  # dict of (N,3) arrays
                                    pc = {key: value for key, value in pc_and_color.items() if node_type_id not in key}  # remove colors
                                    color = {key: value for key, value in pc_and_color.items() if node_type_id in key}  # only colors
                                    if len(color) == 0:
                                        print("No colors found, creating dummy colors.")
                                        # create dummy colors
                                        for k in pc.keys():
                                            num_points = pc[k].shape[0]
                                            color["pc_colors_" + k[-3:]] = np.ones((num_points)) * 0  # first node id
                                    # do fps per timestep
                                    subsampled_pc = {}
                                    subsampled_color = {}
                                    num_points_desired = 512
                                    for k in pc.keys():
                                        points = pc[k]
                                        colors = color[f"pc_{node_type_id}_" + k[-3:]]  # matching color key
                                        if points.shape[0] > num_points_desired:
                                            # FPS
                                            indices = fps(torch.tensor(points), ratio=num_points_desired/points.shape[0], random_start=True)
                                            indices = indices.numpy()
                                            subsampled_pc[k] = points[indices]
                                            subsampled_color[k] = colors[indices]
                                        else:
                                            subsampled_pc[k] = points
                                            subsampled_color[k] = colors

                                    # convert to single array
                                    pc = dict_pointclouds_to_array(subsampled_pc)  # (T,N,3)
                                    color = dict_pointclouds_to_array(subsampled_color)  # (T,N,3)
                                    if cut_length is not None:
                                        pc = pc[:cut_length]
                                        color = color[:cut_length]
                                    # apply temporal stride
                                    pc = pc[::temporal_stride]
                                    color = color[::temporal_stride]
                                    # apply normalization if needed
                                    if apply_normalization:
                                        pc = normalization(pc, min_x=min_x, max_x=max_x)
                                    out_path = f"output/pointcloud_vis/paper_exp/bbv/pc_task_{task_id}_traj_{traj_idx}"
                                    write_pointcloud_timeseries_pvd(
                                        points_ts=pc,
                                        out_dir=out_path,
                                        name="pcd",
                                        point_data={"color": color[:,:, None]}
                                    )
