import h5py
from tqdm import tqdm
from pc_mango.util.mesh_to_pc import mesh_to_pcd, mesh_to_pcd_with_occlusions


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def trampoline_high_res_pc_and_fps(num_final_points):
    current_dataset_file = "../datasets/pc_mango/trampoline_demo.hdf5"

    with h5py.File(current_dataset_file, "r+") as f:

        # Load static data once
        sheet_face = f["global_data"]["sheet_cells"][:]
        ball_faces = {}
        for k in f["global_data"].keys():
            if k.startswith("ball"):
                # split by _ to find diameter
                k_split = k.split("_")
                diameter = k_split[1]
                ball_faces[diameter] = f["global_data"][k][:]

        # Process tasks
        task_names = list(f.keys())
        for task_name in tqdm(task_names, desc="Processing Tasks", total=len(task_names)):
            if not task_name.startswith("task_"):
                continue
            task = f[task_name]
            # find out ball face
            diameter = task["params"]["ball_diameter"][()]
            ball_face = ball_faces[str(diameter)]
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

                # Fetch data, do not normalize at all, this has to be done by the method this time.
                sheet = traj["sheet_pos"][:]
                sphere = traj["sphere_pos"][:]
                num_ts = sheet.shape[0]

                for ts in range(num_ts):
                    pc, pc_node_types = mesh_to_pcd_with_occlusions(
                        dataset_name="trampoline",
                        mesh_pos=sheet[ts],
                        mesh_faces=sheet_face,
                        collider_pos=sphere[ts],
                        collider_faces=ball_face,
                        num_final_points=num_final_points,
                        visualize=False,
                    )
                    # Save point cloud
                    ds_name = f"pc_{ts:03d}"
                    if ds_name in gen_group:
                        del gen_group[ds_name]
                    gen_group.create_dataset(ds_name, data=pc)
                    # save node types/colors
                    ds_name_types = f"pc_node_types_{ts:03d}"
                    if ds_name_types in gen_group:
                        del gen_group[ds_name_types]
                    gen_group.create_dataset(ds_name_types, data=pc_node_types)


if __name__ == "__main__":
    trampoline_high_res_pc_and_fps(512)
