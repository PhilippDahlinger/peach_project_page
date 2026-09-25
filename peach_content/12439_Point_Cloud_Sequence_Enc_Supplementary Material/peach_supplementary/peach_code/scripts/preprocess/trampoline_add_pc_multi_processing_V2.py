"""
Generate point clouds from trampoline mesh trajectories, with multi-process
support. All data is pre-loaded into memory and shared via a pickle file
under $TMP. Results are merged in memory and written to disk in a single
HDF5 write.

Usage:
    # Single process (all tasks)
    python generate_pcd.py

    # Debug with limited tasks:
    python generate_pcd.py --num_tasks 2

    # Multi-process (spawns workers, merges in memory, writes once):
    python generate_pcd.py --launch --num_workers 4

    # Multi-process with limited tasks:
    python generate_pcd.py --launch --num_workers 4 --num_tasks 8
"""

import argparse
import os
import pickle
import subprocess
import sys
import h5py
from tqdm import tqdm

from pc_mango.util.mesh_to_pc import mesh_to_pcd_with_occlusions


DATASET_FILE  = "../datasets/pc_mango/trampoline_v3.hdf5"
OUTPUT_FILE   = "../datasets/pc_mango/trampoline_v3_pcd.hdf5"
NUM_FINAL_PTS = 512
PICKLE_FILE   = os.path.join(os.environ.get("TMP", "/tmp"), "pcd_preloaded_data.pkl")


# ---------------------------------------------------------------------------
# Pre-load: read everything from HDF5 into memory once
# ---------------------------------------------------------------------------

def load_all_data(dataset_file: str, num_tasks: int = None) -> dict:
    print("Pre-loading dataset into memory ...", flush=True)
    data = {"tasks": {}}

    with h5py.File(dataset_file, "r") as src:
        data["sheet_faces"] = src["global_data"]["sheet_cells"][:]

        data["ball_faces_map"] = {}
        for k in src["global_data"].keys():
            if k.startswith("ball"):
                diameter = k.split("_")[1]
                data["ball_faces_map"][diameter] = src["global_data"][k][:]

        task_names = sorted([k for k in src.keys() if k.startswith("task_")])
        if num_tasks is not None:
            task_names = task_names[:num_tasks]
            print(f"  (limited to {num_tasks} tasks for debugging)", flush=True)

        for task_name in tqdm(task_names, desc="Loading tasks"):
            task      = src[task_name]
            diameter  = str(task["params"]["ball_diameter"][()])
            trajs_grp = task["trajs"]

            data["tasks"][task_name] = {"diameter": diameter, "trajs": {}}

            for traj_name in trajs_grp.keys():
                if not traj_name.startswith("traj_"):
                    continue
                traj = trajs_grp[traj_name]
                data["tasks"][task_name]["trajs"][traj_name] = {
                    "sheet_pos":  traj["sheet_pos"][:],
                    "sphere_pos": traj["sphere_pos"][:],
                }

    print("Pre-loading done.", flush=True)
    return data


# ---------------------------------------------------------------------------
# Worker: pure compute, no file I/O
# ---------------------------------------------------------------------------

def process_worker(worker_id: int, num_workers: int, num_final_points: int) -> dict:
    print(f"[worker {worker_id}] loading preloaded data from {PICKLE_FILE} ...", flush=True)
    with open(PICKLE_FILE, "rb") as f:
        all_data = pickle.load(f)

    results        = {}
    sheet_faces    = all_data["sheet_faces"]
    ball_faces_map = all_data["ball_faces_map"]
    task_names     = sorted(all_data["tasks"].keys())
    my_tasks       = task_names[worker_id::num_workers]

    print(f"[worker {worker_id}] processing {len(my_tasks)} / {len(task_names)} tasks", flush=True)

    for task_name in tqdm(my_tasks, desc=f"Worker {worker_id} tasks"):
        task_data = all_data["tasks"][task_name]
        b_faces   = ball_faces_map[task_data["diameter"]]
        results[task_name] = {}

        for traj_name, traj_data in task_data["trajs"].items():
            sheet  = traj_data["sheet_pos"]
            sphere = traj_data["sphere_pos"]
            num_ts = sheet.shape[0]

            print(f"  Starting {task_name} {traj_name}", flush=True)
            results[task_name][traj_name] = {}

            for ts in range(num_ts):
                print(f"    Timestep {ts}/{num_ts - 1}", flush=True)
                pc, pc_node_types = mesh_to_pcd_with_occlusions(
                    dataset_name="trampoline",
                    mesh_pos=sheet[ts],
                    mesh_faces=sheet_faces,
                    collider_pos=sphere[ts],
                    collider_faces=b_faces,
                    num_final_points=num_final_points,
                    visualize=False,
                )
                results[task_name][traj_name][f"pc_{ts:03d}"]            = pc
                results[task_name][traj_name][f"pc_node_types_{ts:03d}"] = pc_node_types

    print(f"[worker {worker_id}] done", flush=True)
    return results


# ---------------------------------------------------------------------------
# Worker entry point: called by subprocess, pickles result to $TMP
# ---------------------------------------------------------------------------

def run_worker_subprocess(worker_id: int, num_workers: int, num_final_points: int):
    result      = process_worker(worker_id, num_workers, num_final_points)
    result_path = os.path.join(os.environ.get("TMP", "/tmp"), f"pcd_result_{worker_id:03d}.pkl")
    print(f"[worker {worker_id}] saving result to {result_path} ...", flush=True)
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[worker {worker_id}] result saved.", flush=True)


# ---------------------------------------------------------------------------
# Merge all worker result dicts and write once to disk
# ---------------------------------------------------------------------------

def merge_and_write(all_results: list[dict], output_file: str):
    print(f"Writing merged results to {output_file} ...", flush=True)

    with h5py.File(output_file, "w") as dst:
        for worker_results in tqdm(all_results, desc="Merging workers"):
            for task_name, trajs in worker_results.items():
                for traj_name, datasets in trajs.items():
                    grp = dst.require_group(f"{task_name}/trajs/{traj_name}/generated_pcd")
                    for ds_name, data in datasets.items():
                        grp.create_dataset(ds_name, data=data, compression="gzip")

    print("Write complete.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",   type=int,  default=0)
    parser.add_argument("--num_workers", type=int,  default=1)
    parser.add_argument("--num_points",  type=int,  default=NUM_FINAL_PTS)
    parser.add_argument("--num_tasks",   type=int,  default=None,
                        help="Limit number of tasks to load (for debugging)")
    parser.add_argument("--launch",      action="store_true",
                        help="Preload data, spawn subprocesses, merge in memory, write once")
    parser.add_argument("--worker",      action="store_true",
                        help="Internal flag: run as a subprocess worker")
    parser.add_argument("--output",      type=str,  default=OUTPUT_FILE)
    args = parser.parse_args()

    if args.worker:
        # ---- subprocess entry point ----
        run_worker_subprocess(args.worker_id, args.num_workers, args.num_points)

    elif args.launch:
        # 1. Preload and save shared pickle
        all_data = load_all_data(DATASET_FILE, num_tasks=args.num_tasks)
        print(f"Saving preloaded data to {PICKLE_FILE} ...", flush=True)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(all_data, f)
        del all_data  # free memory before workers load their own copies

        # 2. Spawn workers
        print(f"Launching {args.num_workers} worker subprocesses ...", flush=True)
        procs = [
            subprocess.Popen([
                sys.executable, __file__,
                "--worker",
                "--num_workers", str(args.num_workers),
                "--worker_id",   str(i),
                "--num_points",  str(args.num_points),
            ])
            for i in range(args.num_workers)
        ]
        for i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0:
                print(f"Worker {i} exited with code {p.returncode}", flush=True)

        # 3. Load results from $TMP and merge in memory
        all_results = []
        for i in range(args.num_workers):
            result_path = os.path.join(os.environ.get("TMP", "/tmp"), f"pcd_result_{i:03d}.pkl")
            print(f"Loading result from {result_path} ...", flush=True)
            with open(result_path, "rb") as f:
                all_results.append(pickle.load(f))
            os.remove(result_path)

        os.remove(PICKLE_FILE)

        # 4. Single HDF5 write
        merge_and_write(all_results, args.output)

    else:
        # Single worker
        all_data = load_all_data(DATASET_FILE, num_tasks=args.num_tasks)
        print(f"Saving preloaded data to {PICKLE_FILE} ...", flush=True)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(all_data, f)
        del all_data

        result = process_worker(args.worker_id, args.num_workers, args.num_points)
        os.remove(PICKLE_FILE)
        merge_and_write([result], args.output)