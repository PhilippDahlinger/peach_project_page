import h5py
import numpy as np
import torch
from tqdm import tqdm

min_x = torch.min(torch.tensor([-168.8057, -100.00])).numpy()
max_x = torch.max(torch.tensor([[173.7962, 215.2644]])).numpy()


def normalization(x):
    return (x - min_x) / (max_x - min_x)

# ------------------------------
# Sampling functions
# ------------------------------
def sample_uniform(bbox_min, bbox_max, n):
    return np.random.uniform(bbox_min, bbox_max, size=(n, 2))

def get_boundary_edges(faces):
    """
    faces: [F,3]
    returns: unique boundary edges [E,2]
    """
    from collections import Counter
    edges = []
    for f in faces:
        edges += [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
    edge_count = Counter(tuple(sorted(e)) for e in edges)
    boundary_edges = [np.array(e) for e, c in edge_count.items() if c == 1]
    return np.array(boundary_edges)

def sample_near_boundary(vertices, faces, n, sigma=0.01):
    """
    Sample points along the mesh/collider boundary with small noise
    """
    boundary_edges = get_boundary_edges(faces)
    idx = np.random.randint(0, len(boundary_edges), size=n)
    edges = boundary_edges[idx]
    v0 = vertices[edges[:,0]]
    v1 = vertices[edges[:,1]]
    t = np.random.rand(n,1)
    pts = (1-t)*v0 + t*v1
    # add small noise to spread out points
    pts += np.random.normal(scale=sigma, size=pts.shape)
    return pts

# ------------------------------
# Triangle-based occupancy
# ------------------------------
def point_in_triangle(pt, tri):
    A = tri[:,0,:]
    B = tri[:,1,:]
    C = tri[:,2,:]

    v0 = C - A
    v1 = B - A
    v2 = pt - A

    dot00 = np.einsum('ij,ij->i', v0, v0)
    dot01 = np.einsum('ij,ij->i', v0, v1)
    dot02 = np.einsum('ij,ij->i', v0, v2)
    dot11 = np.einsum('ij,ij->i', v1, v1)
    dot12 = np.einsum('ij,ij->i', v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-12)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) & (v >= 0) & (u + v <= 1)

def occupancy_from_tris(pts, tris):
    Q = pts.shape[0]
    occ = np.zeros(Q, dtype=bool)
    for f in range(tris.shape[0]):
        tri = np.tile(tris[f:f+1], (Q,1,1))  # [Q,3,2]
        occ |= point_in_triangle(pts, tri)
    return occ.astype(np.float32)

def compute_occupancy_triangles(queries, mesh_vertices, mesh_faces, collider_vertices, collider_faces):
    mesh_tris = mesh_vertices[mesh_faces]
    collider_tris = collider_vertices[collider_faces]
    occ_mesh = occupancy_from_tris(queries, mesh_tris)
    occ_collider = occupancy_from_tris(queries, collider_tris)
    return np.stack([occ_mesh, occ_collider], axis=1)

# ------------------------------
# MAIN PIPELINE
# ------------------------------
def dp_queries_occupancy_3d_boundary(num_queries=2000, visualize=False):
    current_dataset_file = "../datasets/pc_mango/db_v3.hdf5"

    with h5py.File(current_dataset_file, "r+") as f:

        # Load static face data once
        mesh_faces = f["global_data"]["cell_indices"]["mesh_faces"][:]
        collider_faces = f["global_data"]["cell_indices"]["collider_faces"][:]

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

                # # Ensure output group exists
                if "generated_queries" in traj:
                    query_group = traj["generated_queries"]
                else:
                    query_group = traj.create_group("generated_queries")

                # Fetch mesh and collider
                mesh = normalization(traj["mesh_pos"][:])
                collider = normalization(traj["collider_pos"][:])
                num_ts = mesh.shape[0]

                for ts in range(num_ts):
                    # ---------------------------
                    # Generate 2D queries
                    # ---------------------------
                    all_pts_2d = np.vstack([mesh[ts], collider[ts]])
                    bbox_min = all_pts_2d.min(axis=0) - 0.1
                    bbox_max = all_pts_2d.max(axis=0) + 0.1

                    queries_2d = np.vstack([
                        sample_uniform(bbox_min, bbox_max, int(0.3*num_queries)),
                        sample_near_boundary(mesh[ts], mesh_faces, int(0.35*num_queries), sigma=0.02),
                        sample_near_boundary(collider[ts], collider_faces, int(0.35*num_queries), sigma=0.02)
                    ])
                    queries_3d = np.hstack([queries_2d, np.zeros((queries_2d.shape[0],1))])

                    # ---------------------------
                    # Compute triangle-based occupancy
                    # ---------------------------
                    occupancy = compute_occupancy_triangles(
                        queries_2d, mesh[ts], mesh_faces, collider[ts], collider_faces
                    )

                    if visualize:
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.set_title(f'Timestep {ts}')

                        # Plot mesh anchor points
                        ax.scatter(mesh[ts][:, 0], mesh[ts][:, 1],
                                   c='blue', s=20, label='Mesh anchors')

                        # Plot collider anchor points
                        ax.scatter(collider[ts][:, 0], collider[ts][:, 1],
                                   c='cyan', s=20, label='Collider anchors')

                        # Plot queries
                        mask_mesh = occupancy[:, 0] > 0
                        mask_collider = occupancy[:, 1] > 0
                        mask_outside = (~mask_mesh) & (~mask_collider)

                        ax.scatter(queries_3d[mask_mesh, 0], queries_3d[mask_mesh, 1],
                                   c='red', s=10, label='Inside mesh')
                        ax.scatter(queries_3d[mask_collider, 0], queries_3d[mask_collider, 1],
                                   c='blue', s=10, label='Inside collider')
                        ax.scatter(queries_3d[mask_outside, 0], queries_3d[mask_outside, 1],
                                   c='green', s=10, alpha=0.3, label='Outside both')

                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_aspect('equal', 'box')  # Keep correct aspect ratio
                        ax.legend()
                        plt.show()

                    # ---------------------------
                    # Save queries and occupancy
                    # ---------------------------
                    q_ds_name = f"queries_{ts:03d}"
                    if q_ds_name in query_group:
                        del query_group[q_ds_name]
                    query_group.create_dataset(q_ds_name, data=queries_3d)

                    occ_ds_name = f"occupancy_{ts:03d}"
                    if occ_ds_name in query_group:
                        del query_group[occ_ds_name]
                    query_group.create_dataset(occ_ds_name, data=occupancy)

if __name__ == "__main__":
    dp_queries_occupancy_3d_boundary(num_queries=2000, visualize=False)
