import h5py
import torch
from tqdm import tqdm
from pc_mango.util.mesh_to_pc import mesh_to_pcd

min_x = torch.min(torch.tensor([-168.8057, -100.00])).numpy()
max_x = torch.max(torch.tensor([[173.7962, 215.2644]])).numpy()

def normalization(x):
    return (x - min_x) / (max_x - min_x)

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def dp_high_res_pc_and_fps(num_final_points):
    current_dataset_file = "../datasets/pc_mango/db_v3.hdf5"

    with h5py.File(current_dataset_file, "r+") as f:

        # Load static data once
        mesh_faces = f["global_data"]["cell_indices"]["mesh_faces"][:]
        collider_faces = f["global_data"]["cell_indices"]["collider_faces"][:]

        # Process tasks
        task_names = list(f.keys())
        for task_name in tqdm(task_names, desc="Processing Tasks", total=len(task_names)):
            if not task_name.startswith("task_"):
                continue

            task = f[task_name]
            trajs_group = task["trajs"]

            for traj_name in tqdm(trajs_group.keys(), desc=" Trajs", leave=False):
                if not traj_name.startswith("traj_"):
                    continue

                traj = trajs_group[traj_name]

                # ensure output group exists
                if "generated_pcd" in traj:
                    gen_group = traj["generated_pcd"]
                else:
                    gen_group = traj.create_group("generated_pcd")

                # Fetch data
                mesh = normalization(traj["mesh_pos"][:])
                collider = normalization(traj["collider_pos"][:])
                num_ts = mesh.shape[0]

                for ts in range(num_ts):
                    pc, node_types = mesh_to_pcd(
                        "db",
                        mesh[ts],
                        mesh_faces,
                        collider_pos=collider[ts],
                        collider_faces=collider_faces,
                        num_final_points=num_final_points,
                        visualize=False,
                    )

                    # Save point cloud
                    ds_name = f"pc_{ts:03d}"
                    if ds_name in gen_group:
                        del gen_group[ds_name]
                    gen_group.create_dataset(ds_name, data=pc)
                    # save node types/colors
                    ds_name_types = f"pc_colors_{ts:03d}"
                    if ds_name_types in gen_group:
                        del gen_group[ds_name_types]
                    gen_group.create_dataset(ds_name_types, data=node_types)


if __name__ == "__main__":
    dp_high_res_pc_and_fps(1000)
