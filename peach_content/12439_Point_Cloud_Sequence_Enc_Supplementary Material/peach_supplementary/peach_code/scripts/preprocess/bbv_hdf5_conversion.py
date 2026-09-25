import os
import sys

import yaml
import numpy as np
import xml.etree.ElementTree as ET
import h5py
from tqdm import tqdm



def read_mesh_h5(mesh_h5_path: str, mesh_xdmf_path: str):
    """
    Read a mesh H5 file and return (pos_time_stack, faces).

    pos_time_stack: np.ndarray of shape (T, N, 3)
    faces: np.ndarray of shape (F, 3) with integer indices (if any), else None
    """

    def numeric_key(k: str):
        """Return integer suffix of keys like 'data0', 'data102'"""
        return int(k.replace("data", ""))

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
        disp_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "displacement"]
        assert len(disp_keys) > 0, f"Expected at least one displacement key in {mesh_h5_path} based on XDMF parsing, found {disp_keys}"
        disp_keys_sorted = sorted(disp_keys, key=numeric_key)

        force_bc_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "force_bc"]
        assert len(force_bc_keys) > 0, f"Expected at least one force_bc key in {mesh_h5_path} based on XDMF parsing, found {force_bc_keys}"
        force_bc_keys_sorted = sorted(force_bc_keys, key=numeric_key)

        mesh_node_type_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "mesh_node_type"]
        assert len(mesh_node_type_keys) > 0, f"Expected at least one mesh node_type key in {mesh_h5_path} based on XDMF parsing, found {mesh_node_type_keys}"
        mesh_node_type_keys_sorted = sorted(mesh_node_type_keys, key=numeric_key)

        boundary_node_type_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "boundary_node_type"]
        assert len(boundary_node_type_keys) > 0, f"Expected at least one boundary_node_type key in {mesh_h5_path} based on XDMF parsing, found {boundary_node_type_keys}"
        boundary_node_type_keys_sorted = sorted(boundary_node_type_keys, key=numeric_key)

        physical_node_type_keys = [k for k in keys_sorted if parsed_xdmf.get(k) == "physical_node_type"]
        assert len(physical_node_type_keys) > 0, f"Expected at least one physical_node_type key in {mesh_h5_path} based on XDMF parsing, found {physical_node_type_keys}"
        physical_node_type_keys_sorted = sorted(physical_node_type_keys, key=numeric_key)

        # get the displaced data
        frames = []
        for dk in disp_keys_sorted:
            disp = f[dk][()]  # (N, 3) float array
            frame = geom_data + disp  # absolute positions for this frame
            frames.append(frame)
        displacements = np.stack(frames, axis=0)  # (T, N, 3)

        all_keys = {
            "force_bc": force_bc_keys_sorted,
            "mesh_node_type": mesh_node_type_keys_sorted,
            "boundary_node_type": boundary_node_type_keys_sorted,
            "physical_node_type": physical_node_type_keys_sorted
        }
        output_fields = {}
        for key_type, keys in all_keys.items():
            output_fields[key_type] = []
            for k in keys:
                data = f[k][()]
                output_fields[key_type].append(data)
            output_fields[key_type] = np.stack(output_fields[key_type], axis=0)
        # boundary force values are very low, so we can scale them up for better numerical stability in the model
        if "force_bc" in output_fields:
            output_fields["force_bc"] *= 1e5
        output_fields["displacement"] = displacements


        return output_fields, faces


def make_splits(n_tasks: int, RNG_SEED: int = 42):
    rng = np.random.RandomState(RNG_SEED)
    indices = np.arange(n_tasks)
    rng.shuffle(indices)

    n_test = 60
    n_val = 60
    n_train = n_tasks - n_test - n_val

    test = np.sort(indices[:n_test])
    val = np.sort(indices[n_test:n_test + n_val])
    train = np.sort(indices[n_test + n_val:])
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test

def convert(raw_root: str, output_h5: str, max_tasks: int = 2):
    raw_root = os.path.abspath(raw_root)


    with h5py.File(output_h5, "w") as out:
        train, val ,test = make_splits(500)
        splits = out.create_group("splits")
        splits.create_dataset("train_indices", data=train)
        splits.create_dataset("val_indices", data=val)
        splits.create_dataset("test_indices", data=test)
        # iterate tasks (limited by max_tasks to keep test runs short)
        task_idx = 0
        for raw_task_idx in tqdm(range(500), "Processing Task..."):
            if max_tasks is not None and raw_task_idx >= max_tasks:
                break
            # check if data is present for this task
            correct_data = True
            for variation in range(10):
                task_folder_path = os.path.join(raw_root, f"material_{raw_task_idx:04d}_variation_{variation:04d}")
                # check that sim data is in the folder
                sim_data = os.path.join(task_folder_path, "sim.xdmf")
                if not os.path.exists(sim_data):
                    correct_data = False
                    break
            if not correct_data:
                print(f"Warning: missing sim.xdmf for task {raw_task_idx:03d}, skipping")
                continue
            # working path looks good, create group for this task
            task_group = out.create_group(f"task_{task_idx:03}")
            params = task_group.create_group("params")
            # read materials for this path, use first variation since materials are the same across variations
            task_folder_path = os.path.join(raw_root, f"material_{raw_task_idx:04d}_variation_0000")
            # load config.yaml for this task
            config_path = os.path.join(task_folder_path, "config.yaml")
            with open(config_path) as f:
                config = yaml.unsafe_load(f)  # unsafe_load needed for numpy tags
            material = config["material"]
            params.create_dataset("poisson", data=float(material["poisson"]))
            params.create_dataset("viscosity", data=float(material["viscosity"]))
            params.create_dataset("youngs_modulus", data=float(material["youngs_modulus"]))

            trajs_group = task_group.create_group("trajs")

            for variation in range(10):
                traj_path = os.path.join(raw_root, f"material_{raw_task_idx:04d}_variation_{variation:04d}")
                # load sim.xdmf for this variation
                output_data, faces = read_mesh_h5(
                    os.path.join(traj_path, "sim.h5"),
                    os.path.join(traj_path, "sim.xdmf")
                )
                traj_group = trajs_group.create_group(f"traj_{variation:03}")
                # subsample time by 4 (401 to 101 steps)
                # Save pos data
                traj_group.create_dataset("mesh_pos", data=output_data["displacement"][::4], compression="gzip")
                # bc
                traj_group.create_dataset("force_bc", data=output_data["force_bc"][::4], compression="gzip")
                # the other data are not time-dependent, so just save the first frame (shape (N, 3) or (F,) depending on the field)
                traj_group.create_dataset("mesh_node_type", data=output_data["mesh_node_type"][0], compression="gzip")
                traj_group.create_dataset("boundary_node_type", data=output_data["boundary_node_type"][0], compression="gzip")
                traj_group.create_dataset("physical_node_type", data=output_data["physical_node_type"][0], compression="gzip")

                # save faces for this variation
                traj_group.create_dataset("faces", data=faces, compression="gzip")
            task_idx += 1
    print(f"Wrote HDF5 database to {output_h5}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 trampoline_hdf5_conversion.py <raw_root_dir> <output_hdf5> [max_tasks (<=0 means unlimited)]")
        sys.exit(1)
    max_tasks_arg = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
    if max_tasks_arg <= 0:
        max_tasks_arg = None


    convert(sys.argv[1], sys.argv[2], max_tasks=max_tasks_arg)



