import h5py
import numpy as np
from tqdm import tqdm

input_a = "../datasets/pc_mangotrampoline_v4_pcd_1cam_a_cropped.hdf5"
input_b = "../datasets/pc_mangotrampoline_v4_pcd_1cam_b_cropped.hdf5"
output = "../datasets/pc_mangotrampoline_v4_pcd_1cam_combined_cropped.hdf5"

np.random.seed(42)  # For reproducibility

with h5py.File(input_a, "r") as file_a:
    with h5py.File(input_b, "r") as file_b:
        with h5py.File(output, "w") as file_out:
            for key in tqdm(file_a.keys()):
                traj_a = file_a[key]["trajs"]
                traj_b = file_b[key]["trajs"]

                group_out = file_out.create_group(key)
                trajs_out = group_out.create_group("trajs")

                for traj_key in traj_a.keys():
                    current_out_traj = trajs_out.create_group(traj_key)
                    current_out_traj = current_out_traj.create_group("generated_pcd")

                    # Randomly choose from file_a or file_b
                    if np.random.rand() < 0.5:
                        source = traj_a[traj_key]["generated_pcd"]
                    else:
                        source = traj_b[traj_key]["generated_pcd"]

                    # Copy all datasets
                    for dataset in source:
                        current_out_traj.create_dataset(dataset, data=source[dataset][()])

