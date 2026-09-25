import h5py
import numpy as np
import torch

import dash
from dash import dcc, html, Input, Output
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

    sigmas = scale * np.array([0.01, 0.03])
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
def dp_queries_occupancy_3d_boundary(num_queries=50, visualize=False):
    mesh_dataset_file = "../datasets/pc_mango/trampoline_v4.hdf5"
    output_dataset_file = "../datasets/pc_mangotrampoline_v4_sdf_debug.hdf5"

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
                bbox_min[2] -= 40  # add some padding in z to capture above/below sheet
                bbox_max[2] -= 30
                bbox_min[0:2] += 30
                bbox_max[0:2] -= 30


                sphere_face = ball_faces[str(task_in["params"]["ball_diameter"][()])]

                for ts in range(0, 50, 2):
                    verts_a = sheet[ts]  # (V_a, 2)
                    verts_b = sphere[ts]  # (V_b, 2)

                    # 70% near surface, 30% uniform
                    n_surface = int(num_queries * 0.1)
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

                    # vis
                    sdf = np.min(sdf, axis=-1)  # take min across objects for coloring
                    fig = plot_mesh_with_query_points(
                        [verts_a, verts_b],
                        [sheet_face, sphere_face],
                        query_pts,
                        sdf=sdf,
                        max_points=5000
                    )
                    # fig.show()

                    fig.write_image(f"output/sdf_data/sdf_{task_name}_{traj_name}_ts_{ts:03d}.png", width=1080, height=1080, scale=3)

                    # app = dash.Dash(__name__)
                    # app.layout = html.Div([
                    #     dcc.Graph(id='graph', figure=fig, style={'height': '90vh'}),
                    #     html.Pre(id='camera-out')
                    # ])
                    #
                    # @app.callback(
                    #     Output('camera-out', 'children'),
                    #     Input('graph', 'relayoutData')
                    # )
                    # def print_camera(relayout):
                    #     if relayout and 'scene.camera' in relayout:
                    #         cam = relayout['scene.camera']
                    #         print(cam)
                    #         return str(cam)
                    #     return ''
                    #
                    # app.run(debug=False)
                    #
                    # exit()

                    # Save


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



import plotly.graph_objects as go
import numpy as np


def plot_mesh_with_query_points(
    verts_list,
    faces_list,
    query_pts,
    sdf=None,
    max_points=5000,
    point_size=14,
):
    fig = go.Figure()

    # --- Plot meshes ---
    for verts, faces in zip(verts_list, faces_list):
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=0.6,
            flatshading=False,
            lighting=dict(ambient=0.6),
        ))
        # Edges overlay
        edge_x, edge_y, edge_z = [], [], []
        for tri in faces:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                edge_x += [verts[tri[a], 0], verts[tri[b], 0], None]
                edge_y += [verts[tri[a], 1], verts[tri[b], 1], None]
                edge_z += [verts[tri[a], 2], verts[tri[b], 2], None]
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none'
        ))

    # --- Subsample query points ---
    if len(query_pts) > max_points:
        idx = np.random.choice(len(query_pts), max_points, replace=False)
        query_pts = query_pts[idx]
        if sdf is not None:
            sdf = sdf[idx]

    if sdf is not None:
        sdf = np.tanh(sdf / 80)
        abs_max = np.abs(sdf).max()

    # --- Plot query points ---
    fig.add_trace(go.Scatter3d(
        x=query_pts[:, 0],
        y=query_pts[:, 1],
        z=query_pts[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=sdf if sdf is not None else 'red',
            colorscale='Plasma' if sdf is not None else None,
            cmin=-abs_max * 1.3,
            cmax=abs_max * 1.3,
            showscale=False,
        ),
        hoverinfo='none'
    ))

    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.3, y=1.0, z=-0.1),
            projection=dict(type='perspective')
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='white',
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig


if __name__ == "__main__":
    # dp_queries_occupancy_3d_boundary(num_queries=80, visualize=False)

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(1.2, 4))
    fig.subplots_adjust(left=0.3, right=0.55, top=0.92, bottom=0.08)

    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation='vertical'
    )

    cb.set_ticks([])
    cb.outline.set_visible(True)

    ax.text(0.5, 1.04, 'Positive', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=12)
    ax.text(0.5, -0.04, 'Negative', transform=ax.transAxes,
            ha='center', va='top', fontsize=12)

    # Vertical label to the right, like "Poisson's Ratio"
    ax.text(2.2, 0.5, 'Signed Distance Field (SDF)', transform=ax.transAxes,
            ha='center', va='center', fontsize=14,
            rotation=-90)
    plt.tight_layout()
    plt.savefig('output/sdf_data/sdf_colorbar.png', bbox_inches='tight', dpi=300, transparent=True)
    plt.close()