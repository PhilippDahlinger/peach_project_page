import numpy as np
import h5py
from matplotlib import pyplot as plt
from tqdm import tqdm


def extract_boundary_edges(faces):
    """
    Extract edges that appear in only one triangle = boundary edges.
    Returns (E, 2) array of vertex index pairs.
    """
    edge_count = {}
    for tri in faces:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1
    return np.array([list(e) for e, c in edge_count.items() if c == 1])


def dist_point_to_segments(pts, v0, v1):
    """
    Vectorized distance from (Q, 2) points to (E, 2)->(E, 2) segments.
    Returns (Q, E) distances.
    """
    seg = v1 - v0                                                          # (E, 2)
    seg_len2 = (seg * seg).sum(axis=1)                                     # (E,)
    diff = pts[:, None, :] - v0[None, :, :]                               # (Q, E, 2)
    t = (diff * seg[None]).sum(axis=2) / (seg_len2[None] + 1e-12)         # (Q, E)
    t = np.clip(t, 0.0, 1.0)
    closest = v0[None] + t[:, :, None] * seg[None]                        # (Q, E, 2)
    delta = pts[:, None, :] - closest                                      # (Q, E, 2)
    return np.sqrt((delta * delta).sum(axis=2))                            # (Q, E)


def point_in_mesh_2d(pts, verts, faces):
    """
    Barycentric test per triangle. Returns bool array (Q,): True = inside.
    """
    inside = np.zeros(len(pts), dtype=bool)
    for tri in faces:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        def sign(p, a, b):
            return (p[:,0]-b[0])*(a[1]-b[1]) - (a[0]-b[0])*(p[:,1]-b[1])
        d1 = sign(pts, v0, v1)
        d2 = sign(pts, v1, v2)
        d3 = sign(pts, v2, v0)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        inside |= ~(has_neg & has_pos)
    return inside


def signed_distance_2d(query_pts, verts, faces, boundary_edges):
    """
    Args:
        query_pts:      (Q, 2)
        verts:          (V, 2)
        faces:          (F, 3) int
        boundary_edges: (E, 2) int -- precomputed
    Returns:
        sdf: (Q,) negative inside, positive outside
    """
    v0 = verts[boundary_edges[:, 0]]
    v1 = verts[boundary_edges[:, 1]]

    dists = dist_point_to_segments(query_pts, v0, v1)   # (Q, E)
    unsigned_dist = dists.min(axis=1)                    # (Q,)

    inside = point_in_mesh_2d(query_pts, verts, faces)
    sign = np.where(inside, -1.0, 1.0)

    return (sign * unsigned_dist).astype(np.float32)


def sample_near_boundary_2d(verts, boundary_edges, n):
    """
    Sample points along boundary edges + multi-scale noise.

    Args:
        verts:           (V, 2)
        boundary_edges:  (E, 2)
        n:               number of samples

    Returns:
        samples: (n, 2)
    """
    # ----------------------------------
    # Get edge endpoints
    # ----------------------------------
    v0 = verts[boundary_edges[:, 0]]   # (E, 2)
    v1 = verts[boundary_edges[:, 1]]   # (E, 2)

    # ----------------------------------
    # Compute edge lengths → sampling probs
    # ----------------------------------
    edge_vec = v1 - v0
    edge_lengths = np.linalg.norm(edge_vec, axis=1)  # (E,)

    probs = edge_lengths / (edge_lengths.sum() + 1e-12)

    # ----------------------------------
    # Sample edges
    # ----------------------------------
    edge_indices = np.random.choice(len(boundary_edges), size=n, p=probs)

    v0 = v0[edge_indices]
    v1 = v1[edge_indices]

    # ----------------------------------
    # Sample points along edges
    # ----------------------------------
    t = np.random.rand(n, 1)  # (n, 1)
    samples = (1 - t) * v0 + t * v1

    # ----------------------------------
    # Add multi-scale noise
    # ----------------------------------
    scale = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))

    sigmas = scale * np.array([0.01, 0.05, 0.2])
    sigma_choices = np.random.choice(sigmas, size=n)

    noise = np.random.normal(scale=sigma_choices[:, None], size=samples.shape)

    return samples + noise


def sample_uniform_2d(bbox_min, bbox_max, n):
    return np.random.uniform(bbox_min, bbox_max, size=(n, 2))


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def dp_queries_sdf_2d(num_queries=512):
    mesh_dataset_file = "../datasets/pc_mangobbv_v3.hdf5"
    output_dataset_file = "../datasets/pc_mangobbv_v3_sdf.hdf5"

    with h5py.File(mesh_dataset_file, "r") as f_in, \
         h5py.File(output_dataset_file, "w") as f_out:

        for task_name in tqdm(f_in.keys(), desc="Tasks"):
            if not task_name.startswith("task_"):
                continue
            task_out = f_out.create_group(task_name)
            trajs_in = f_in[task_name]["trajs"]
            trajs_out = task_out.create_group("trajs")
            faces_mesh = trajs_in["traj_000"]["faces"][:]

            # Precompute once -- faces never change across trajs/timesteps
            boundary_edges_mesh = extract_boundary_edges(faces_mesh)

            for traj_name in tqdm(trajs_in.keys(), desc=" Trajs", leave=False):
                if not traj_name.startswith("traj_"):
                    continue
                traj_in  = trajs_in[traj_name]
                traj_out = trajs_out.create_group(traj_name)

                mesh     = traj_in["mesh_pos"][:, :, :2]      # (T, V_a, 2)
                num_ts   = mesh.shape[0]

                queries_grp = traj_out.create_group("queries")
                sdf_grp     = traj_out.create_group("sdf")

                bbox_min = np.array([mesh[:, :, 0].min(), mesh[:, :, 1].min()])
                bbox_max = np.array([mesh[:, :, 0].max(), mesh[:, :, 1].max()])

                for ts in range(num_ts):
                    verts_a = mesh[ts]      # (V_a, 2)
                    n_surface = int(num_queries * 0.7)
                    n_uniform = num_queries - n_surface

                    pts_a     = sample_near_boundary_2d(verts_a, boundary_edges_mesh,     n_surface)
                    pts_uniform = sample_uniform_2d(bbox_min - 1, bbox_max + 1, n_uniform)

                    query_pts = np.concatenate([pts_a, pts_uniform], axis=0)
                    np.random.shuffle(query_pts)

                    sdf_a = signed_distance_2d(query_pts, verts_a, faces_mesh,     boundary_edges_mesh)
                    sdf   = np.stack([sdf_a], axis=1)  # (Q, 2)
                    # take the sign for debug
                    # sdf = np.where(sdf < 0, -1, 1)

                    # visualize_sdf_debug(
                    #     verts_a, faces_mesh,
                    #     query_pts, sdf,
                    #     sdf_index=0,  # visualize mesh SDF
                    #     cmap="coolwarm",
                    #     point_size=20,
                    #     show_zero_level=True
                    # )


                    queries_grp.create_dataset(f"ts_{ts:03d}", data=query_pts, compression="gzip")
                    sdf_grp.create_dataset(    f"ts_{ts:03d}", data=sdf,       compression="gzip")



def plot_mesh(ax, verts, faces, color='black', alpha=0.6, linewidth=1.0):
    """Draw triangle mesh."""
    for tri in faces:
        pts = verts[tri]
        pts = np.vstack([pts, pts[0]])  # close triangle
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=linewidth)


def visualize_sdf_debug(
    mesh_verts,
    mesh_faces,
    query_pts,
    sdf,
    sdf_index=0,         # 0 = mesh, 1 = collider
    cmap="coolwarm",
    point_size=10,
    show_zero_level=True
):
    """
    Debug visualization for SDF.

    Args:
        mesh_verts:     (V_a, 2)
        mesh_faces:     (F_a, 3)
        collider_verts: (V_b, 2)
        collider_faces: (F_b, 3)
        query_pts:      (Q, 2)
        sdf:            (Q, 2)
        sdf_index:      which SDF to visualize
    """
    sdf_vals = sdf[:, sdf_index]

    fig, ax = plt.subplots(figsize=(8, 8))

    # --------------------------
    # Meshes
    # --------------------------
    plot_mesh(ax, mesh_verts, mesh_faces, color='blue', alpha=0.8)

    # --------------------------
    # Query points colored by SDF
    # --------------------------
    sc = ax.scatter(
        query_pts[:, 0],
        query_pts[:, 1],
        c=sdf_vals,
        cmap=cmap,
        s=point_size
    )

    # --------------------------
    # Zero level set (SDF = 0)
    # --------------------------
    if show_zero_level:
        try:
            ax.tricontour(
                query_pts[:, 0],
                query_pts[:, 1],
                sdf_vals,
                levels=[0],
                colors='black',
                linewidths=1.5
            )
        except Exception:
            # Happens if triangulation fails (e.g. degenerate points)
            pass

    # --------------------------
    # Colorbar
    # --------------------------
    cbar = plt.colorbar(sc, ax=ax)
    label = "mesh SDF" if sdf_index == 0 else "collider SDF"
    cbar.set_label(label)

    # --------------------------
    # Layout
    # --------------------------
    ax.set_title(f"SDF Debug View ({label})")
    ax.set_aspect('equal')
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    dp_queries_sdf_2d(num_queries=512)