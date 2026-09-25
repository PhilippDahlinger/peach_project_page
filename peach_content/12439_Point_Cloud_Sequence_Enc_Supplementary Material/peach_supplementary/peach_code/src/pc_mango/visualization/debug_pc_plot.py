import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
def visualize_pointcloud_sequence(pointcloud, types, center=None, radius=None):
    """
    Args:
        pointcloud: np.ndarray of shape (T, N, 3)
        types:      np.ndarray of shape (T, N, 1), values 0 or 1
        center:     (3,) array-like or None
        radius:     float or None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from matplotlib.colors import ListedColormap

    if hasattr(pointcloud, "cpu"):
        pointcloud = pointcloud.cpu().numpy()
    if hasattr(types, "cpu"):
        types = types.cpu().numpy()

    T = pointcloud.shape[0]
    cmap = ListedColormap(["steelblue", "tomato"])

    state = {"t": 0}

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.15)

    ax_back = plt.axes([0.3, 0.03, 0.15, 0.05])
    ax_fwd  = plt.axes([0.55, 0.03, 0.15, 0.05])
    btn_back = Button(ax_back, "< Back")
    btn_fwd  = Button(ax_fwd,  "Forward >")

    def draw(t):
        elev, azim = ax.elev, ax.azim
        ax.cla()

        pts = pointcloud[t]
        c   = types[t].squeeze(-1)

        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=c, cmap=cmap, vmin=0, vmax=1,
            s=5, alpha=0.8
        )

        # --- draw sphere if provided ---
        if center is not None and radius is not None:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)

            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

            ax.plot_surface(
                x, y, z,
                color="blue",
                alpha=0.2,   # transparency
                linewidth=0,
                shade=True
            )

        ax.set_title(f"Timestep {t + 1} / {T}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        mid        = (pts.max(axis=0) + pts.min(axis=0)) / 2
        half_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
        ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
        ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
        ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

        # stuff for real pointcloud, delete if it is disturbing
        # ax.set_xlim(50, 240)
        # ax.set_ylim(50, 240)
        # ax.set_zlim(-30, 160)

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