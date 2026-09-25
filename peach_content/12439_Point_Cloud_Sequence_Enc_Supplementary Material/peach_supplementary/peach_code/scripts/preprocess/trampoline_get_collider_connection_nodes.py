import h5py
import numpy as np
from fpsample import fps_sampling

np.random.seed(42)

path = "../datasets/pc_mango/trampoline_v5.hdf5"

needed_diameters = [20.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]

output = {}

def sample_lower_half(nodes: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Parameters
    ----------
    nodes : np.ndarray
        Shape (N,3) array of xyz node positions.
    n_samples : int
        Number of points to sample after filtering.

    Returns
    -------
    np.ndarray
        Subsampled nodes of shape (n_samples, 3)
    """

    # 1. Split by z median → lower half
    z_median = np.median(nodes[:, 2])
    lower_nodes = nodes[nodes[:, 2] <= z_median]

    # 2. Farthest point sampling
    idx = fps_sampling(lower_nodes, n_samples)

    return idx

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_sampling(nodes: np.ndarray, sampled_indices: np.ndarray):
    """
    Visualize all nodes and the sampled nodes.

    Parameters
    ----------
    nodes : np.ndarray
        Shape (N,3) array of xyz node positions.
    sampled_indices : np.ndarray
        Indices of sampled nodes within the lower-half subset.
    """

    z_median = np.median(nodes[:, 2])
    lower_nodes = nodes[nodes[:, 2] <= z_median]

    sampled_nodes = lower_nodes[sampled_indices]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # all nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
               s=10, alpha=0.2, label="all nodes")

    # lower half
    ax.scatter(lower_nodes[:, 0], lower_nodes[:, 1], lower_nodes[:, 2],
               s=20, alpha=0.6, label="lower half")

    # sampled nodes
    ax.scatter(sampled_nodes[:, 0], sampled_nodes[:, 1], sampled_nodes[:, 2],
               s=80, label="sampled (FPS)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.tight_layout()
    plt.show()

with h5py.File(path, "r") as f:
    for k in f.keys():
        if k.startswith("task_"):
            task = f[k]
            diameter = task["params"]["ball_diameter"][()]
            if diameter in needed_diameters:
                print(f"Task: {k}, diameter: {diameter}")
                traj = task["trajs"]["traj_001"]
                collider_pos = traj["sphere_pos"][:]
                initial_collider = collider_pos[0]
                connection_indices = sample_lower_half(initial_collider, 16)

                # visualize_sampling(initial_collider, connection_indices)
                # remove from needed diameters to avoid duplicates
                needed_diameters.remove(diameter)
                output[diameter] = connection_indices.tolist()
        if len(needed_diameters) == 0:
            break

    # do the same for the sheet
    task = f["task_000"]
    sheet_traj = task["trajs"]["traj_001"]
    sheet_pos = sheet_traj["sheet_pos"][:]
    initial_sheet = sheet_pos[0]
    sheet_idx = fps_sampling(initial_sheet, 120)
    # visualize_sampling(initial_sheet, sheet_idx)
    output["sheet"] = sheet_idx.tolist()

# write output to json file
import json
with open("config/trampoline_collider_connection_nodes_v5.json", "w") as f:
    json.dump(output, f, indent=4)

