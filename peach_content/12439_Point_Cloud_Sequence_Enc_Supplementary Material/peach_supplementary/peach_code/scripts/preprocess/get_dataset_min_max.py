import h5py
import numpy as np
from tqdm import tqdm

input_hdf5_path = "../datasets/pc_mango/trampoline_v2.hdf5"

global_min = np.full(3, np.inf)
global_max = np.full(3, -np.inf)

with h5py.File(input_hdf5_path, "r") as f:
    for task_key in tqdm(f.keys()):
        if not task_key.startswith("task_"):
            continue
        trajs_group = f[task_key]["trajs"]
        for traj_key in trajs_group.keys():
            if not traj_key.startswith("traj_"):
                continue
            traj = trajs_group[traj_key]
            # For a different dataset, adapt the keys
            for dataset_name in ["sheet_pos", "sphere_pos"]:
                data = traj[dataset_name][:]          # load one dataset at a time
                data_2d = data.reshape(-1, 3)
                global_min = np.minimum(global_min, data_2d.min(axis=0))
                global_max = np.maximum(global_max, data_2d.max(axis=0))

print(f"Global min per dim (x, y, z): {global_min}")
print(f"Global max per dim (x, y, z): {global_max}")