import h5py
import numpy as np
from tqdm import tqdm
import igl


# ------------------------------
# Sampling functions
# ------------------------------
def sample_uniform(bbox_min, bbox_max, n):
    return np.random.uniform(bbox_min, bbox_max, size=(n, 3))


def sample_near_boundary(vertices, faces, n):
    """
    Sample points on triangle surfaces + multi-scale Gaussian noise.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Triangle areas (for importance sampling)
    tri_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    tri_probs = tri_areas / (tri_areas.sum() + 1e-12)

    # Sample triangle indices
    tri_indices = np.random.choice(len(faces), size=n, p=tri_probs)

    v0 = v0[tri_indices]
    v1 = v1[tri_indices]
    v2 = v2[tri_indices]

    # Barycentric sampling
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)

    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]

    w = 1 - (u + v)
    samples = u * v0 + v * v1 + w * v2

    # Multi-scale noise
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    scale = np.linalg.norm(bbox_size)  # diagonal

    sigmas = scale * np.array([0.001, 0.005, 0.02])
    sigma_choices = np.random.choice(sigmas, size=n)
    noise = np.random.normal(scale=sigma_choices[:, None], size=samples.shape)

    return samples + noise


# ------------------------------
# SDF computation
# ------------------------------
def get_sdf_labels(verts, faces, query_pts):
    result = igl.signed_distance(
        query_pts.astype(np.float64),
        verts.astype(np.float64),
        faces,
        sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    return result[0].astype(np.float32)


# ------------------------------
# MAIN PIPELINE
# ------------------------------
def dp_queries_occupancy_3d_boundary(num_queries=512, visualize=False):
    mesh_dataset_file = "../datasets/pc_mango/trampoline_v4.hdf5"
    output_dataset_file = "../datasets/pc_mangotrampoline_v4_sdf.hdf5"

    with h5py.File(mesh_dataset_file, "r") as f_in, \
         h5py.File(output_dataset_file, "w") as f_out:

        sheet_face = f_in["global_data"]["sheet_cells"][:]
        ball_faces = {}
        for k in f_in["global_data"].keys():
            if k.startswith("ball"):
                # split by _ to find diameter
                k_split = k.split("_")
                diameter = k_split[1]
                ball_faces[diameter] = f_in["global_data"][k][:]

        for task_name in tqdm(f_in.keys(), desc="Processing Tasks"):
            if not task_name.startswith("task_"):
                continue

            task_in = f_in[task_name]
            task_out = f_out.create_group(task_name)

            trajs_in = task_in["trajs"]
            trajs_out = task_out.create_group("trajs")

            for traj_name in tqdm(trajs_in.keys(), desc=" Trajs", leave=False):
                if not traj_name.startswith("traj_"):
                    continue

                traj_in = trajs_in[traj_name]
                traj_out = trajs_out.create_group(traj_name)

                sheet = traj_in["sheet_pos"][:]
                sphere = traj_in["sphere_pos"][:]
                num_ts = sheet.shape[0]
                # vis debug
                # plot_mesh_with_normals(mesh[0], mesh_faces)

                queries_grp = traj_out.create_group("queries")
                sdf_grp = traj_out.create_group("sdf")

                # Bounding box
                sphere_bbox_min = sphere[0].min(axis=0)
                sphere_bbox_max = sphere[1].max(axis=0)
                sheet_bbox_min = sheet[0].min(axis=0)
                sheet_bbox_max = sheet[0].max(axis=0)
                bbox_min = np.minimum(sphere_bbox_min, sheet_bbox_min)
                bbox_max = np.maximum(sphere_bbox_max, sheet_bbox_max)

                sphere_face = ball_faces[str(task_in["params"]["ball_diameter"][()])]

                for ts in range(num_ts):
                    verts_a = sheet[ts]  # (V_a, 2)
                    verts_b = sphere[ts]  # (V_b, 2)

                    # 70% near surface, 30% uniform
                    n_surface = int(num_queries * 0.7)
                    n_uniform = num_queries - n_surface

                    pts_a = sample_near_boundary(
                        verts_a, sheet_face, n_surface // 2
                    )

                    pts_b = sample_near_boundary(
                        verts_b, sphere_face, n_surface // 2
                    )

                    pts_uniform = sample_uniform(
                        bbox_min - 30, bbox_max + 30, n_uniform
                    )

                    query_pts = np.concatenate(
                        [pts_a, pts_b, pts_uniform], axis=0
                    )


                    # Shuffle to avoid ordering bias
                    np.random.shuffle(query_pts)

                    # Compute SDF
                    sdf_a = get_sdf_labels(verts_a, sheet_face, query_pts)
                    sdf_b = get_sdf_labels(verts_b, sphere_face, query_pts)

                    # in other tasks there are multiple sdf so reshape to add a channel dimension for consistency
                    sdf = np.stack([sdf_a, sdf_b], axis=-1)  # (N, 2)

                    # debug: just get the sign
                    # sdf = np.sign(sdf)

                    # debug vis
                    # plot_mesh_with_query_points(
                    #     verts_a, sheet_face, query_pts, sdf=sdf[:, 0], max_points=5000
                    # )

                    # Save
                    queries_grp.create_dataset(
                        f"ts_{ts:03d}", data=query_pts, compression="gzip"
                    )
                    sdf_grp.create_dataset(
                        f"ts_{ts:03d}", data=sdf, compression="gzip"
                    )

# VIS
import numpy as np
import matplotlib.pyplot as plt


def compute_face_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms

    centers = (v0 + v1 + v2) / 3.0
    return centers, normals


def plot_mesh_with_normals(verts, faces, normal_scale=0.05, max_normals=2000):
    centers, normals = compute_face_normals(verts, faces)

    # Optional: subsample normals (important for large meshes)
    if len(centers) > max_normals:
        idx = np.random.choice(len(centers), max_normals, replace=False)
        centers = centers[idx]
        normals = normals[idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh (wireframe-ish)
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color='lightblue',
        alpha=0.5,
        edgecolor='gray',
        linewidth=0.2
    )

    # Plot normals
    ax.quiver(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
        length=normal_scale,
        normalize=True
    )

    ax.set_box_aspect([1, 1, 1])
    plt.title("Mesh with Face Normals")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_mesh_with_query_points(
    verts,
    faces,
    query_pts,
    sdf=None,
    max_points=5000,
    point_size=2
):
    """
    Visualize mesh + sampled query points.

    Args:
        verts: (V, 3)
        faces: (F, 3)
        query_pts: (N, 3)
        sdf: (N,) optional, colors points by signed distance
    """

    # Subsample points if too many (important for speed)
    if len(query_pts) > max_points:
        idx = np.random.choice(len(query_pts), max_points, replace=False)
        query_pts = query_pts[idx]
        if sdf is not None:
            sdf = sdf[idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot mesh ---
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color='lightgray',
        alpha=0.3,
        edgecolor='none'
    )

    # --- Plot query points ---
    if sdf is not None:
        sc = ax.scatter(
            query_pts[:, 0],
            query_pts[:, 1],
            query_pts[:, 2],
            c=sdf,
            cmap='coolwarm',
            s=point_size
        )
        plt.colorbar(sc, ax=ax, shrink=0.5, label="SDF")
    else:
        ax.scatter(
            query_pts[:, 0],
            query_pts[:, 1],
            query_pts[:, 2],
            color='red',
            s=point_size
        )

    # Equal aspect ratio
    max_range = (verts.max(axis=0) - verts.min(axis=0)).max()
    mid = verts.mean(axis=0)

    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    ax.set_title("Mesh + Query Points")
    plt.show()


if __name__ == "__main__":
    dp_queries_occupancy_3d_boundary(num_queries=512, visualize=False)