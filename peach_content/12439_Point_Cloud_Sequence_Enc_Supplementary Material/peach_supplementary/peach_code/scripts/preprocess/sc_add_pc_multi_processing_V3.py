"""
Generate point clouds from trampoline mesh trajectories, with multi-process
support. All data is pre-loaded into memory and shared via per-worker pickle
files under $TMP. Each worker writes results directly to an HDF5 shard;
shards are stream-merged into the final output with no in-memory accumulation.

Usage:
    # Single process (all tasks)
    python generate_pcd.py

    # Debug with limited tasks:
    python generate_pcd.py --num_tasks 2

    # Multi-process (spawns workers, stream-merges shards, writes once):
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


DATASET_FILE  = "../datasets/pc_mango/sc_v3.hdf5"
OUTPUT_FILE   = "../datasets/pc_mango/sc_v3_pcd_4cams.hdf5"
NUM_FINAL_PTS = 1024


def _worker_pickle_path(worker_id: int) -> str:
    return os.path.join(os.environ.get("TMP", "/tmp"), f"pcd_data_{worker_id:03d}.pkl")

def _worker_shard_path(worker_id: int) -> str:
    return os.path.join(os.environ.get("TMP", "/tmp"), f"pcd_shard_{worker_id:03d}.hdf5")


# ---------------------------------------------------------------------------
# Pre-load: read everything from HDF5 into memory once, then slice per worker
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


def save_worker_pickles(all_data: dict, num_workers: int) -> list[str]:
    """
    Slice the full dataset by worker and write one small pickle per worker.
    Each worker only loads its own slice, so total pickle RAM across all
    workers is ~1× dataset instead of N× dataset.
    """
    task_names = sorted(all_data["tasks"].keys())
    paths = []

    for i in range(num_workers):
        my_tasks = task_names[i::num_workers]
        worker_data = {
            "sheet_faces":    all_data["sheet_faces"],
            "ball_faces_map": all_data["ball_faces_map"],
            "tasks":          {k: all_data["tasks"][k] for k in my_tasks},
        }
        path = _worker_pickle_path(i)
        print(f"  Saving worker {i} pickle ({len(my_tasks)} tasks) → {path}", flush=True)
        with open(path, "wb") as f:
            pickle.dump(worker_data, f)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Worker: pure compute, writes results directly to an HDF5 shard
# ---------------------------------------------------------------------------

def process_worker(worker_id: int, num_final_points: int):
    pickle_path = _worker_pickle_path(worker_id)
    shard_path  = _worker_shard_path(worker_id)

    print(f"[worker {worker_id}] loading slice from {pickle_path} ...", flush=True)
    with open(pickle_path, "rb") as f:
        worker_data = pickle.load(f)

    sheet_faces    = worker_data["sheet_faces"]
    ball_faces_map = worker_data["ball_faces_map"]
    task_names     = sorted(worker_data["tasks"].keys())

    print(f"[worker {worker_id}] processing {len(task_names)} tasks, writing shard → {shard_path}", flush=True)

    with h5py.File(shard_path, "w") as dst:
        for task_name in tqdm(task_names, desc=f"Worker {worker_id} tasks"):
            task_data = worker_data["tasks"][task_name]
            b_faces   = ball_faces_map[task_data["diameter"]]

            for traj_name, traj_data in task_data["trajs"].items():
                sheet  = traj_data["sheet_pos"]
                sphere = traj_data["sphere_pos"]
                num_ts = sheet.shape[0]

                grp = dst.require_group(f"{task_name}/trajs/{traj_name}/generated_pcd")

                print(f"  Starting {task_name} {traj_name}", flush=True)

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
                    grp.create_dataset(f"pc_{ts:03d}",            data=pc,            compression="gzip")
                    grp.create_dataset(f"pc_node_types_{ts:03d}", data=pc_node_types, compression="gzip")

    print(f"[worker {worker_id}] shard written.", flush=True)


# ---------------------------------------------------------------------------
# Stream-merge HDF5 shards → single output file (no in-memory accumulation)
# ---------------------------------------------------------------------------

def merge_shards(num_workers: int, output_file: str):
    """
    Copy each shard into the output file using h5py's built-in copy, which
    streams data dataset-by-dataset without loading everything into RAM.
    """
    print(f"Stream-merging {num_workers} shards → {output_file} ...", flush=True)

    with h5py.File(output_file, "w") as dst:
        for i in tqdm(range(num_workers), desc="Merging shards"):
            shard_path = _worker_shard_path(i)
            with h5py.File(shard_path, "r") as src:
                for task_name in src:
                    src.copy(task_name, dst)
            os.remove(shard_path)

    print("Merge complete.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",   type=int,  default=0)
    parser.add_argument("--num_workers", type=int,  default=4)
    parser.add_argument("--num_points",  type=int,  default=NUM_FINAL_PTS)
    parser.add_argument("--num_tasks",   type=int,  default=None,
                        help="Limit number of tasks to load (for debugging)")
    parser.add_argument("--launch",      action="store_true",
                        help="Preload data, spawn subprocesses, stream-merge shards, write once")
    parser.add_argument("--worker",      action="store_true",
                        help="Internal flag: run as a subprocess worker")
    parser.add_argument("--output",      type=str,  default=OUTPUT_FILE)
    args = parser.parse_args()

    if args.worker:
        # ---- subprocess entry point ----
        process_worker(args.worker_id, args.num_points)
        # Clean up this worker's pickle slice after use
        pickle_path = _worker_pickle_path(args.worker_id)
        if os.path.exists(pickle_path):
            os.remove(pickle_path)

    elif args.launch:
        # 1. Preload full dataset once
        all_data = load_all_data(DATASET_FILE, num_tasks=args.num_tasks)

        # 2. Write one sliced pickle per worker (~1/N of tasks each)
        print(f"Saving per-worker pickle slices for {args.num_workers} workers ...", flush=True)
        save_worker_pickles(all_data, args.num_workers)
        del all_data  # free the full dataset before workers start

        # 3. Spawn workers; each writes its own HDF5 shard
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

        # 4. Stream-merge shards into final output (no RAM accumulation)
        merge_shards(args.num_workers, args.output)

    else:
        # Single worker: slice just for worker 0, process, write shard, merge
        all_data = load_all_data(DATASET_FILE, num_tasks=args.num_tasks)
        save_worker_pickles(all_data, num_workers=1)
        del all_data

        process_worker(worker_id=0, num_final_points=args.num_points)
        os.remove(_worker_pickle_path(0))

        merge_shards(num_workers=1, output_file=args.output)