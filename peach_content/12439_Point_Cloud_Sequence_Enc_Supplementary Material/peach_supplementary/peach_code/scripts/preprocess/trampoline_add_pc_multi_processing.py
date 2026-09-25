"""
Generate point clouds from trampoline mesh trajectories, with multi-process
support. Each process handles a subset of tasks and writes to its own HDF5
file. At the end all temporary files are merged back into the input HDF5.

Usage:
    # Single process (all tasks)
    python generate_pcd.py

    # Multi-process: launch N workers manually
    python generate_pcd.py --num_workers 4 --worker_id 0 &
    python generate_pcd.py --num_workers 4 --worker_id 1 &
    python generate_pcd.py --num_workers 4 --worker_id 2 &
    python generate_pcd.py --num_workers 4 --worker_id 3 &

    # After all workers finish, merge:
    python generate_pcd.py --merge_only
"""

import argparse
import os
import glob
import h5py
import numpy as np
import subprocess
import sys
from tqdm import tqdm

from pc_mango.util.mesh_to_pc import mesh_to_pcd_with_occlusions


DATASET_FILE  = "../datasets/pc_mango/trampoline_v3.hdf5"
TMP_DIR       = "../datasets/pc_mango/pcd_tmp"
OUTPUT_FILE   = "../datasets/pc_mango/trampoline_v3_pcd.hdf5"
NUM_FINAL_PTS = 512


# ---------------------------------------------------------------------------
# Worker: process a slice of tasks, write to a temp HDF5
# ---------------------------------------------------------------------------

def process_worker(worker_id: int, num_workers: int, num_final_points: int):
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TMP_DIR, f"pcd_worker_{worker_id:03d}.hdf5")

    with h5py.File(DATASET_FILE, "r") as src:
        sheet_faces = src["global_data"]["sheet_cells"][:]

        ball_faces_map = {}
        for k in src["global_data"].keys():
            if k.startswith("ball"):
                diameter = k.split("_")[1]
                ball_faces_map[diameter] = src["global_data"][k][:]

        task_names = sorted([k for k in src.keys() if k.startswith("task_")])

    # Distribute tasks across workers
    my_tasks = task_names[worker_id::num_workers]
    print(f"[worker {worker_id}] processing {len(my_tasks)} / {len(task_names)} tasks → {tmp_path}")

    with h5py.File(DATASET_FILE, "r") as src, \
         h5py.File(tmp_path, "w") as dst:

        for task_name in tqdm(my_tasks, desc=f"Worker {worker_id} tasks"):
            task       = src[task_name]
            diameter   = str(task["params"]["ball_diameter"][()])
            b_faces    = ball_faces_map[diameter]
            trajs_grp  = task["trajs"]

            dst_task = dst.require_group(task_name)

            for traj_name in tqdm(trajs_grp.keys(), desc="  trajs", leave=False):
                if not traj_name.startswith("traj_"):
                    continue

                traj       = trajs_grp[traj_name]
                sheet      = traj["sheet_pos"][:]
                sphere     = traj["sphere_pos"][:]
                num_ts     = sheet.shape[0]

                dst_traj   = dst_task.require_group(traj_name)
                gen_group  = dst_traj.require_group("generated_pcd")

                for ts in range(num_ts):
                    pc, pc_node_types = mesh_to_pcd_with_occlusions(
                        dataset_name="trampoline",
                        mesh_pos=sheet[ts],
                        mesh_faces=sheet_faces,
                        collider_pos=sphere[ts],
                        collider_faces=b_faces,
                        num_final_points=num_final_points,
                        visualize=False,
                    )
                    gen_group.create_dataset(f"pc_{ts:03d}",            data=pc,            compression="gzip")
                    gen_group.create_dataset(f"pc_node_types_{ts:03d}", data=pc_node_types, compression="gzip")

    print(f"[worker {worker_id}] done → {tmp_path}")


# ---------------------------------------------------------------------------
# Merge: copy generated_pcd groups from all tmp files into the input HDF5
# ---------------------------------------------------------------------------

def merge_tmp_files(output_file: str = OUTPUT_FILE):
    tmp_files = sorted(glob.glob(os.path.join(TMP_DIR, "pcd_worker_*.hdf5")))
    if not tmp_files:
        print("No temporary files found to merge.")
        return

    print(f"Merging {len(tmp_files)} temporary files into {output_file} ...")

    with h5py.File(output_file, "a") as dst:
        for tmp_path in tqdm(tmp_files, desc="Merging files"):
            with h5py.File(tmp_path, "r") as src:
                for task_name in src.keys():
                    for traj_name in src[task_name].keys():
                        gen_src  = src[task_name][traj_name]["generated_pcd"]
                        dst_path = f"{task_name}/trajs/{traj_name}/generated_pcd"
                        dst_gen  = dst.require_group(dst_path)
                        for ds_name in gen_src.keys():
                            if ds_name in dst_gen:
                                del dst_gen[ds_name]
                            dst_gen.create_dataset(
                                ds_name,
                                data=gen_src[ds_name][:],
                                compression="gzip"
                            )

    print("Merge complete.")
    for f in tmp_files:
        os.remove(f)
    print(f"Removed {len(tmp_files)} temporary files.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",   type=int,  default=0)
    parser.add_argument("--num_workers", type=int,  default=1)
    parser.add_argument("--num_points",  type=int,  default=NUM_FINAL_PTS)
    parser.add_argument("--merge_only",  action="store_true")
    parser.add_argument("--launch",      action="store_true",
                        help="Spawn all workers as subprocesses, then merge when done")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output HDF5 file for merged point clouds")
    args = parser.parse_args()

    if args.merge_only:
        merge_tmp_files(args.output)

    elif args.launch:
        print(f"Launching {args.num_workers} worker processes ...")
        procs = [
            subprocess.Popen([
                sys.executable, __file__,
                "--num_workers", str(args.num_workers),
                "--worker_id",   str(i),
                "--num_points",  str(args.num_points),
            ])
            for i in range(args.num_workers)
        ]
        # Wait for all workers to finish
        for i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0:
                print(f"Worker {i} exited with code {p.returncode}")

        print("All workers finished. Starting merge ...")
        merge_tmp_files(args.output)

    else:
        process_worker(
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            num_final_points=args.num_points,
        )
