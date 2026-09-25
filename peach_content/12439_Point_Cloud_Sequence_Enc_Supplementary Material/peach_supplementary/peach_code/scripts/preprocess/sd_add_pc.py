import h5py
from tqdm import tqdm
from pc_mango.util.mesh_to_pc import mesh_to_pcd


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def dp_high_res_pc_and_fps(num_final_points):
    current_dataset_file = "../datasets/pc_mango/sd_v1.hdf5"

    with h5py.File(current_dataset_file, "r+") as f:

        # Load static data once
        mesh_faces = f["global_data"]["cells"][:]

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
                mesh = traj["mesh_pos"][:] / 100.0
                num_ts = mesh.shape[0]

                for ts in range(num_ts):
                    pc = mesh_to_pcd(
                        dataset_name="sd",
                        mesh_pos=mesh[ts],
                        mesh_faces=mesh_faces,
                        num_final_points=num_final_points,
                        visualize=False,
                    )
                    ds_name = f"pc_{ts:03d}"
                    if ds_name in gen_group:
                        del gen_group[ds_name]
                    gen_group.create_dataset(ds_name, data=pc)


if __name__ == "__main__":
    dp_high_res_pc_and_fps(2000)
