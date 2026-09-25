import h5py
from tqdm import tqdm
from pc_mango.util.mesh_to_pc import mesh_to_pcd


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def dp_high_res_pc_and_fps(num_final_points):
    current_dataset_file = "../datasets/pc_mangobbv_v3.hdf5"
    pc_output_file = "../datasets/pc_mangobbv_v3_pcd.hdf5"

    output = {}

    with h5py.File(current_dataset_file, "r") as f:
        # Process tasks
        task_names = list(f.keys())
        for task_name in tqdm(task_names, desc="Processing Tasks", total=len(task_names)):
            if not task_name.startswith("task_"):
                continue

            task = f[task_name]
            trajs_group = task["trajs"]
            output[task_name] = {}
            for traj_name in trajs_group.keys():
                if not traj_name.startswith("traj_"):
                    continue

                traj = trajs_group[traj_name]
                # Fetch data
                mesh = traj["mesh_pos"][:]
                mesh_faces = traj["faces"][:]
                num_ts = mesh.shape[0]

                output[task_name][traj_name] = {}

                for ts in range(num_ts):
                    pc, pc_node_type = mesh_to_pcd(
                        dataset_name="bbv",
                        mesh_pos=mesh[ts],
                        mesh_faces=mesh_faces,
                        num_final_points=num_final_points,
                        visualize=False,
                    )
                    output[task_name][traj_name][f"pc_{ts:03d}"] = pc
                    output[task_name][traj_name][f"pc_node_type_{ts:03d}"] = pc_node_type

            # break # debug

        # save output to new hdf5 file
        with h5py.File(pc_output_file, "w") as out:
            for task_name, trajs in output.items():
                task_group = out.create_group(task_name)
                task_group = task_group.create_group("trajs")
                for traj_name, pcs in trajs.items():
                    traj_group = task_group.create_group(traj_name)
                    traj_group = traj_group.create_group("generated_pcd")
                    for ts_name, pc in pcs.items():
                        # save pc and pc_node_type in separate datasets
                        traj_group.create_dataset(ts_name, data=pc)
        print("Saved point cloud data to", pc_output_file)


if __name__ == "__main__":
    dp_high_res_pc_and_fps(512)
