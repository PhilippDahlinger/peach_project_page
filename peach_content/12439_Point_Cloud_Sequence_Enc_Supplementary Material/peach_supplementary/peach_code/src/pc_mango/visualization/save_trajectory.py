import h5py
import numpy as np
import meshio
from pathlib import Path

from pc_mango.dataset.util.util import hdf5_group_to_dict
from pc_mango.util.xdmf_utils.simulation_data import RawSimulationData

# positions: (T, N, 3)  OR  list of length T with (N,3) arrays
# timesteps: length T list of floats (or ints)
# connectivity: (M, nodes_per_cell) int array
# cell_type: e.g. "triangle", "tetra", "quad", "hexahedron"

def export_deforming_mesh_to_xdmf(
    positions,
    connectivity,
    cell_type,
    out_path,
    features=None,   # optional: list of dicts, length T
):
    positions = np.asarray(positions)
    if positions.shape[2] == 2:
        # extend for z dim if it is missing
        positions = np.concatenate([positions, np.zeros((*positions.shape[:2], 1))], axis=2)

    T, N, D = positions.shape
    assert D == 3, "ParaView works best with (N,3) points. Use z=0 if 2D."

    # fixed mesh geometry at t=0
    points0 = positions[0]

    # fixed connectivity
    cells = [meshio.CellBlock(cell_type, np.asarray(connectivity, dtype=int))]

    # point_data: one dict per timestep
    point_data = []
    for k in range(T):
        disp = positions[k] - points0  # (N,3)

        pd = {
            "displacement": disp,
            # optional: store actual positions too (sometimes useful for debugging)
            # "position": positions[k],
        }

        if features is not None:
            pd.update(features[k])   # merge extra per-node fields

        point_data.append(pd)

    timesteps = np.arange(T)

    sim = RawSimulationData(
        points=points0,
        cells=cells,
        timesteps=list(timesteps),
        point_data=point_data,
        cell_data=[{} for _ in range(T)],
    )

    sim.save(Path(out_path), data_format="HDF")
    print(f"Wrote: {out_path} (+ {Path(out_path).with_suffix('.h5')})")


def merge_meshes(pointsA0, connA, pointsB0, connB, cell_type):
    points0 = np.vstack([pointsA0, pointsB0])

    connA = np.asarray(connA, dtype=int)
    connB = np.asarray(connB, dtype=int) + len(pointsA0)

    connectivity = np.vstack([connA, connB])
    cells = [meshio.CellBlock(cell_type, connectivity)]

    # static per-point labels (0 for A, 1 for B)
    object_id_points = np.concatenate([
        np.zeros(len(pointsA0), dtype=np.int32),
        np.ones(len(pointsB0), dtype=np.int32),
    ])

    # static per-cell labels
    object_id_cells = np.concatenate([
        np.zeros(len(connA), dtype=np.int32),
        np.ones(len(connB), dtype=np.int32),
    ])

    return points0, cells, object_id_points, object_id_cells


def export_two_meshes_to_xdmf_merged(
    positions_A, connectivity_A,
    positions_B, connectivity_B,
    cell_type,
    out_path,
    features_A=None,   # list of dicts length T (optional)
    features_B=None,   # list of dicts length T (optional)
):
    positions_A = np.asarray(positions_A)
    positions_B = np.asarray(positions_B)
    # extend for z dim if it is missing
    if positions_A.shape[2] == 2:
        positions_A = np.concatenate([positions_A, np.zeros((*positions_A.shape[:2], 1))], axis=2)
    if positions_B.shape[2] == 2:
        positions_B = np.concatenate([positions_B, np.zeros((*positions_B.shape[:2], 1))], axis=2)

    TA, NA, DA = positions_A.shape
    TB, NB, DB = positions_B.shape
    assert TA == TB, "Both objects must have same number of timesteps."
    assert DA == 3 and DB == 3, "Use (N,3) points (z=0 for 2D)."

    T = TA

    # fixed mesh geometry at t=0
    pointsA0 = positions_A[0]
    pointsB0 = positions_B[0]

    points0, cells, object_id_points, object_id_cells = merge_meshes(
        pointsA0, connectivity_A,
        pointsB0, connectivity_B,
        cell_type=cell_type
    )

    point_data = []
    cell_data = []

    for k in range(T):
        dispA = positions_A[k] - pointsA0
        dispB = positions_B[k] - pointsB0

        disp = np.vstack([dispA, dispB])

        pd = {
            "displacement": disp,
            "object_id": object_id_points,
        }

        # merge optional extra per-point features
        if features_A is not None:
            pd.update(features_A[k])
        if features_B is not None:
            # Need to merge B features by stacking along axis=0
            for key, arrB in features_B[k].items():
                arrA = features_A[k][key] if (features_A is not None and key in features_A[k]) else None

                if arrA is None:
                    # if only B has it, pad A part with zeros
                    pad_shape = (NA,) + arrB.shape[1:]
                    arrA = np.zeros(pad_shape, dtype=arrB.dtype)

                pd[key] = np.concatenate([arrA, arrB], axis=0)

        point_data.append(pd)

        cd = {
            "object_id": [object_id_cells],
        }
        cell_data.append(cd)

    timesteps = np.arange(T)

    sim = RawSimulationData(
        points=points0,
        cells=cells,
        timesteps=list(timesteps),
        point_data=point_data,
        cell_data=cell_data,
    )

    sim.save(Path(out_path), data_format="HDF")
    print(f"Wrote: {out_path} (+ {Path(out_path).with_suffix('.h5')})")



if __name__ == "__main__":
    root = "..."
    task_id = 0
    traj_idx = 2
    with h5py.File(root, "r") as hdf:
        num_tasks_loaded = 0
        for task_key in sorted(hdf.keys()):
            if task_key.startswith("task_"):
                task_index = int(task_key[-3:])
                if task_index == task_id:
                    task = hdf5_group_to_dict(hdf[task_key])
                    for traj_key in sorted(task["trajs"].keys()):
                        if int(traj_key[-3:]) == traj_idx:
                            traj = task["trajs"][traj_key]
                            positions = traj["mesh_pos"]  # list of (N,3) arrays
                            connectivity = hdf["global_data"]["cells"][:]  # (M, nodes_per_cell)
                            cell_type = "triangle"  # assuming 2D mesh
                            out_path = f"output/debug/example_vis_2.xdmf"
                            export_deforming_mesh_to_xdmf(
                                positions=positions,
                                connectivity=connectivity,
                                cell_type=cell_type,
                                out_path=out_path,
                            )
