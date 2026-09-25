import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualize_prediction_interactive(vis_results, show_pc=True):
    """
    Interactive visualization of predicted mesh, collider, and point cloud.
    Use Forward/Back buttons to advance through time.

    Args:
        vis_results: dict with keys:
            - predicted_trajectory: (50, 1201, 3) - mesh vertices over time
            - mesh_connectivity: (2304, 3) - mesh face indices
            - predicted_collider: (50, 516, 3) - collider vertices over time
            - collider_connectivity: (1028, 3) - collider face indices
            - pointcloud: (14, 1024, 3) - point cloud over time
            - pc_time_stamps: (14,) - time indices for point cloud
        show_pc: whether to show point cloud
    """

    # Convert to numpy if needed
    mesh_traj = vis_results["predicted_trajectory"]
    if hasattr(mesh_traj, "cpu"):
        mesh_traj = mesh_traj.cpu().numpy()

    collider_traj = vis_results["predicted_collider"]
    if hasattr(collider_traj, "cpu"):
        collider_traj = collider_traj.cpu().numpy()

    mesh_faces = vis_results["mesh_connectivity"]
    if hasattr(mesh_faces, "cpu"):
        mesh_faces = mesh_faces.cpu().numpy()

    collider_faces = vis_results["collider_connectivity"]
    if hasattr(collider_faces, "cpu"):
        collider_faces = collider_faces.cpu().numpy()

    pointcloud = vis_results["pointcloud"]
    if hasattr(pointcloud, "cpu"):
        pointcloud = pointcloud.cpu().numpy()

    pc_time_stamps = vis_results["pc_time_stamps"] * 100  # convert to same time scale as mesh/collider
    if hasattr(pc_time_stamps, "cpu"):
        pc_time_stamps = pc_time_stamps.cpu().numpy()



    T = mesh_traj.shape[0]
    state = {"t": 0}

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.15)

    # Create buttons
    ax_back = plt.axes([0.3, 0.03, 0.15, 0.05])
    ax_fwd = plt.axes([0.55, 0.03, 0.15, 0.05])
    btn_back = Button(ax_back, "< Back")
    btn_fwd = Button(ax_fwd, "Forward >")

    def draw(t):
        elev, azim = ax.elev, ax.azim
        ax.cla()

        # Get mesh for this frame
        mesh_pos = mesh_traj[t]  # (1201, 3)
        mesh_triangles = [mesh_pos[mesh_faces[i]] for i in range(len(mesh_faces))]
        mesh_collection = Poly3DCollection(
            mesh_triangles,
            alpha=0.7,
            facecolor="lightblue",
            edgecolor="none"
        )
        ax.add_collection3d(mesh_collection)

        # Get collider for this frame
        collider_pos = collider_traj[t]  # (516, 3)
        collider_triangles = [collider_pos[collider_faces[i]] for i in range(len(collider_faces))]
        collider_collection = Poly3DCollection(
            collider_triangles,
            alpha=0.5,
            facecolor="tomato",
            edgecolor="none"
        )
        ax.add_collection3d(collider_collection)

        # Add point cloud if available
        if show_pc:
            closest_pc_idx = (abs(pc_time_stamps - t)).argmin()
            pc = pointcloud[closest_pc_idx]
            ax.scatter(
                pc[:, 0], pc[:, 1], pc[:, 2],
                c="green", s=5, alpha=0.6, label="Point Cloud"
            )

        # Set equal aspect ratio
        all_pts = np.concatenate([mesh_pos, collider_pos], axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        mid = (mins + maxs) / 2
        half = (maxs - mins).max() / 2

        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Timestep {t + 1} / {T}")

        if show_pc:
            ax.legend()

        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    def on_forward(_):
        if state["t"] < T - 1:
            state["t"] += 1
            draw(state["t"])

    def on_back(_):
        if state["t"] > 0:
            state["t"] -= 1
            draw(state["t"])

    btn_fwd.on_clicked(on_forward)
    btn_back.on_clicked(on_back)

    draw(0)
    plt.show()
    print("Done.")


if __name__ == "__main__":
    # Example usage:
    # vis_results = {...}  # your data dict
    # visualize_prediction_interactive(vis_results, show_pc=True)
    pass