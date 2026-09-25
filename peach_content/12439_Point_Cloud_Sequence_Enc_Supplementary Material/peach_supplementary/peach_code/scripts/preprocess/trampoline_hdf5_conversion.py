"""
Convert raw falling-ball simulation folders into a single HDF5 database file.

This script mirrors the output structure used by other conversion scripts in this
repo (see `db_hdf5_conversion.py`) and produces an HDF5 with:

- task_{:03d}/params: global task parameters (tape thickness, elastic params,
  viscoelastic Prony series, ball diameter, ball mass)
- task_{:03d}/trajs/traj_{:03d}/sphere_pos : (T, N_sphere_nodes, 3)
- task_{:03d}/trajs/traj_{:03d}/sheet_pos  : (T, N_sheet_nodes, 3)
- task_{:03d}/trajs/traj_{:03d}/params : Stage2 initial sphere position
- global_data/sheet_cells : triangles for the sheet
- global_data/ball_<n>_cells : triangles for each ball mesh size encountered
- splits/train_indices, val_indices, test_indices (default 600/100/100 when
  enough tasks exist, else proportional split)

Usage example::

    python3 scripts/preprocess/trampoline_hdf5_conversion.py \
        ../datasets/pc_mango/falling_ball_raw \
        /tmp/trampoline_db.hdf5

The script attempts to be robust to whether the HDF5 time-series store absolute
positions or displacements (it inspects magnitudes and adds displacements to
`data0` when needed).

"""

import os
import sys
import h5py
import yaml
import numpy as np
from tqdm import tqdm
import importlib.util
import xml.etree.ElementTree as ET
from downsample_trampoline.remesh import remesh
from downsample_trampoline.create_faces import generate_mesh


def numeric_key(k: str):
    """Return integer suffix of keys like 'data0', 'data102'"""
    return int(k.replace("data", ""))


def read_mesh_h5(mesh_h5_path: str, mesh_xdmf_path: str, discard_first_ts=False):
    """
    Read a mesh H5 file and return (pos_time_stack, faces).

    pos_time_stack: np.ndarray of shape (T, N, 3)
    faces: np.ndarray of shape (F, 3) with integer indices (if any), else None
    """

    def xdmf_key_to_label(xdmf_path: str) -> dict:
        """Returns {'/data0': 'Geometry', '/data1': 'Topology', '/data2': 'U', ...}"""
        tree = ET.parse(xdmf_path)
        root = tree.getroot()

        def strip_ns(tag):
            return tag.split("}")[-1] if "}" in tag else tag

        mapping = {}

        for elem in root.iter():
            tag = strip_ns(elem.tag)
            if tag in ("Attribute", "Geometry", "Topology"):
                label = elem.get("Name", tag)  # Attribute has Name; Geometry/Topology use tag itself
                for child in elem.iter():
                    if strip_ns(child.tag) == "DataItem":
                        text = (child.text or "").strip()
                        if ":" in text:
                            h5_key = text.split(":", 1)[1]
                            mapping[h5_key[1:]] = label  # remove leading '/' from h5_key

        return mapping


    parsed_xdmf = xdmf_key_to_label(mesh_xdmf_path)

    with h5py.File(mesh_h5_path, "r") as f:
        keys = [k for k in f.keys() if k.startswith("data")]
        keys_sorted = sorted(keys, key=numeric_key)

        # find out which corresponds to the faces
        faces_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "Topology"]
        assert len(faces_keys) == 1, f"Expected exactly one faces key in {mesh_h5_path} based on XDMF parsing, found {faces_keys}"
        faces_key = faces_keys[0]
        faces = f[faces_key][()]  # (F, 3) int array
        # find initial geometry
        geom_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "Geometry"]
        assert len(geom_keys) == 1, f"Expected exactly one geometry key in {mesh_h5_path} based on XDMF parsing, found {geom_keys}"
        geom_key = geom_keys[0]
        geom_data = f[geom_key][()]  # (N, 3) float array
        # now get all displacements (key = "U" in XDMF)
        disp_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "U"]
        assert len(disp_keys) > 0, f"Expected at least one displacement key in {mesh_h5_path} based on XDMF parsing, found {disp_keys}"
        disp_keys_sorted = sorted(disp_keys, key=numeric_key)
        if discard_first_ts:
            # discard the first displacement (initial frame to set the sim up)
            disp_keys_sorted = disp_keys_sorted[1:]
        # get the displaced data
        frames = []
        for dk in disp_keys_sorted:
            disp = f[dk][()]  # (N, 3) float array
            frame = geom_data + disp  # absolute positions for this frame
            frames.append(frame)
        stack = np.stack(frames, axis=0)  # (T, N, 3)
        return stack, faces



def make_splits(n_tasks: int, RNG_SEED: int = 42):
    rng = np.random.RandomState(RNG_SEED)
    indices = np.arange(n_tasks)
    rng.shuffle(indices)

    n_test = 100
    n_val = 100
    n_train = n_tasks - n_test - n_val

    test = np.sort(indices[:n_test])
    val = np.sort(indices[n_test:n_test + n_val])
    train = np.sort(indices[n_test + n_val:])
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test

def convert(raw_root: str, output_h5: str, experiments_yaml_rel: str = "meta_data_filtered/experiments_phys_valid.yaml", max_tasks: int = 2, n_even: int = 4, discard_first_ts=False,):
    raw_root = os.path.abspath(raw_root)
    meta_path = os.path.join(raw_root, experiments_yaml_rel)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata YAML not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    experiments = meta.get("experiments", [])
    # Defensive check: ensure experiments is a list of dicts; this avoids
    # static-analysis warnings about using `.get` on unknown types and
    # makes the converter more robust to malformed metadata.
    if not isinstance(experiments, list):
        raise RuntimeError(f"Invalid metadata: 'experiments' must be a list, got {type(experiments)}")
    n_tasks = len(experiments)
    print(f"Found {n_tasks} Stage-1 tasks in metadata")

    # Prepare global cell storage
    ball_cells_map = {}


    train_idx, val_idx, test_idx = make_splits(n_tasks)

    with h5py.File(output_h5, "w") as out:
        # splits
        splits = out.create_group("splits")
        splits.create_dataset("train_indices", data=train_idx)
        splits.create_dataset("val_indices", data=val_idx)
        splits.create_dataset("test_indices", data=test_idx)

        # iterate tasks (limited by max_tasks to keep test runs short)
        for task_idx, exp in enumerate(tqdm(experiments, desc="Tasks")):
            if max_tasks is not None and task_idx >= max_tasks:
                break
            task_group = out.create_group(f"task_{task_idx:03}")
            params = task_group.create_group("params")

            # ensure the experiment entry is a dict (skip otherwise)
            if not isinstance(exp, dict):
                print(f"Warning: skipping malformed experiment entry at index {task_idx}: {type(exp)}")
                continue

            # read tape & material params from stage1. Support both old and
            # new metadata schemas. New schema nests ply/ball keys as shown in
            # the example stage1 dict.
            stage1 = exp.get("stage1", {})

            # PLY / tape thickness
            ply = stage1.get("ply") or stage1.get("tape_metadata") or {}
            if isinstance(ply, dict):
                if "ply_thickness" in ply:
                    params.create_dataset("tape_thickness", data=float(ply["ply_thickness"]))
                elif "tape_thickness" in ply:
                    params.create_dataset("tape_thickness", data=float(ply["tape_thickness"]))

                # Elastic params
                elastic = ply.get("elastic") or ply.get("elastic_parameters") or {}
                if isinstance(elastic, dict):
                    # new schema uses keys 'Young' and 'Poisson'
                    if "Young" in elastic:
                        params.create_dataset("youngs_modulus", data=float(elastic["Young"]))
                    if "young_modulus" in elastic:
                        params.create_dataset("youngs_modulus", data=float(elastic["young_modulus"]))
                    if "Poisson" in elastic:
                        params.create_dataset("poisson_ratio", data=float(elastic["Poisson"]))
                    if "poisson_ratio" in elastic:
                        params.create_dataset("poisson_ratio", data=float(elastic["poisson_ratio"]))

                # Viscous / Prony series: new schema may provide a dict under 'viscous'
                visc = ply.get("viscous") or ply.get("viscoelastic_prony")
                if visc is not None:
                    # If visc is a dict with named keys, store them in a subgroup
                    if isinstance(visc, dict):
                        visc_group = params.create_group("viscous")
                        for k, v in visc.items():
                            try:
                                visc_group.create_dataset(k, data=np.array(v))
                            except Exception:
                                # fallback to scalar
                                visc_group.create_dataset(k, data=float(v))
                    else:
                        # If visc is a list/array, save as 'prony'
                        try:
                            params.create_dataset("prony", data=np.array(visc))
                        except Exception:
                            pass

            # Ball metadata: new schema often contains a `ball` dict plus a
            # `ball_metadata` dict with extra info.
            ball = stage1.get("ball") or {}
            if isinstance(ball, dict):
                if "diameter" in ball:
                    params.create_dataset("ball_diameter", data=float(ball["diameter"]))
                if "mass" in ball:
                    params.create_dataset("ball_mass", data=float(ball["mass"]))

            # Also copy through the `ball_metadata` dict if present
            ball_meta = stage1.get("ball_metadata", {})
            if isinstance(ball_meta, dict) and ball_meta:
                bm = params.create_group("ball_metadata")
                for k, v in ball_meta.items():
                    try:
                        bm.create_dataset(k, data=np.array(v))
                    except Exception:
                        # fallback to string
                        try:
                            bm.create_dataset(k, data=str(v))
                        except Exception:
                            pass

            trajs_group = task_group.create_group("trajs")

            s1_id = stage1.get("stage1_id")
            for s2 in exp.get("stage2_samples", []):
                s2_id = s2.get("stage2_id")
                traj_group = trajs_group.create_group(f"traj_{s2_id:03}")
                # build sim dir path
                sim_dir = os.path.join(raw_root, f"sim_results/S1_{s1_id:03d}_S2_{s2_id:03d}")
                ball_h5 = os.path.join(sim_dir, "BallMeshes.h5")
                ball_xdmf = os.path.join(sim_dir, "BallMeshes.xdmf")
                ply_h5 = os.path.join(sim_dir, "PlyMeshes.h5")
                ply_xdmf = os.path.join(sim_dir, "PlyMeshes.xdmf")

                if not os.path.exists(ball_h5) or not os.path.exists(ply_h5):
                    print(f"Warning: missing files for S1_{s1_id:03d}_S2_{s2_id:03d}; skipping")
                    continue

                try:
                    ball_stack, ball_faces = read_mesh_h5(ball_h5, ball_xdmf, discard_first_ts=discard_first_ts)
                except Exception as e:
                    print(f"Failed reading ball mesh for {sim_dir}: {e}")
                    continue

                # remesh the sheet using the remesh module (works on the original PlyMeshes.h5)
                # remesh expects a path to an H5 file and n_even
                resampled_positions = remesh(ply_h5, n_even)
                if discard_first_ts:
                    # discard first time step (initial condition)
                    resampled_positions = resampled_positions[1:] if resampled_positions is not None else None
                if resampled_positions is not None:
                    # remesh.remesh returns shape (T, M, 3)
                    sheet_stack = resampled_positions

                # Save frames
                traj_group.create_dataset("sphere_pos", data=ball_stack, compression="gzip")
                traj_group.create_dataset("sheet_pos", data=sheet_stack, compression="gzip")

                # stage2 params (starting position) — prefer new schema s2['ball']['translation']
                stage2_params = traj_group.create_group("params")
                start_pos = None
                # new schema: s2 has a 'ball' dict with 'translation'
                if isinstance(s2, dict):
                    ball_info = s2.get('ball') or {}
                    if isinstance(ball_info, dict) and 'translation' in ball_info:
                        start_pos = ball_info.get('translation')


                if start_pos is not None:
                    stage2_params.create_dataset("start_position", data=np.array(start_pos, dtype=np.float64))

                if ball_faces is not None:
                    # use diameter of the ball to determine the key for storing ball cells
                    # different tasks have different balls

                    diameter = float(params["ball_diameter"][()])
                    key = f"ball_{diameter}_cells"
                    if key not in ball_cells_map:
                        ball_cells_map[key] = ball_faces

        # generate sheet cells
        pts_grid, faces_grid = generate_mesh(n_even)
        sheet_cells = np.array(faces_grid, dtype=np.int64)
        # write global_data
        global_data = out.create_group("global_data")
        global_data.create_dataset("sheet_cells", data=sheet_cells, compression="gzip")
        for k, faces in ball_cells_map.items():
            global_data.create_dataset(k, data=faces, compression="gzip")

    print(f"Wrote HDF5 database to {output_h5}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 trampoline_hdf5_conversion.py <raw_root_dir> <output_hdf5> [max_tasks (<=0 means unlimited)] [n_even (for remeshing)] [discard_first_ts (True/False)]")
        sys.exit(1)
    max_tasks_arg = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
    if max_tasks_arg <= 0:
        max_tasks_arg = None

    n_even_arg = int(sys.argv[4]) if len(sys.argv) >= 5 else 4
    discard_first_ts = sys.argv[5].lower() == "true" if len(sys.argv) >= 6 else False

    convert(sys.argv[1], sys.argv[2], max_tasks=max_tasks_arg, n_even=n_even_arg, discard_first_ts=discard_first_ts)