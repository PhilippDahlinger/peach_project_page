from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from pc_mango.util.pyg import fps, knn, ptr2batch

def debug_patch_assignments(spacetime, center_spacetime, patch_idxs, center_idxs, n_patches, patch_size, batch_idx=0, L=None):
    """
    Args:
        spacetime:        (B*T*N, 4) - x,y,z,t
        center_spacetime: (B*C, 4)
        patch_idxs:       (B*C*G,)
        center_idxs:      (B*C,)
        n_patches:        total number of centers across all batches
        patch_size:       G
        batch_idx:        which batch element to visualize
        L:                T*N points per batch element
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from matplotlib.widgets import Button, TextBox

    pts_mask = torch.zeros(spacetime.shape[0], dtype=torch.bool)
    pts_mask[batch_idx * L : (batch_idx + 1) * L] = True

    local_center_mask = pts_mask[center_idxs]
    local_center_global = torch.where(local_center_mask)[0]
    C_local = local_center_global.shape[0]

    patch_idxs_2d = patch_idxs.view(n_patches, patch_size)

    st = spacetime.detach().cpu().numpy()
    all_pts = st[pts_mask.cpu().numpy()]
    all_centers = center_spacetime.detach().cpu().numpy()[local_center_mask.cpu().numpy()]

    t_vals_global = all_pts[:, 3]
    t_min, t_max = t_vals_global.min(), t_vals_global.max()

    grey   = np.array([0.5, 0.5, 0.5])
    viridis = plt.cm.get_cmap("viridis")

    def time_to_cluster_color(t_norm):
        """Active patch: viridis colormap directly encodes time."""
        return viridis(t_norm)[:, :3]  # (G, 3), drop alpha

    def time_to_bg_color(t_norm):
        """Background: black -> grey over time."""
        t_norm = np.asarray(t_norm)
        return (grey * t_norm[:, None]).clip(0, 1)

    state      = {"ci": 0}
    anim_state = {"running": False, "anim": None}

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.15)

    ax_back    = plt.axes([0.10, 0.03, 0.10, 0.05])
    ax_fwd     = plt.axes([0.22, 0.03, 0.10, 0.05])
    ax_anim    = plt.axes([0.34, 0.03, 0.10, 0.05])
    ax_textbox = plt.axes([0.58, 0.03, 0.10, 0.05])
    ax_go      = plt.axes([0.70, 0.03, 0.08, 0.05])

    btn_back = Button(ax_back, "< Back")
    btn_fwd  = Button(ax_fwd,  "Forward >")
    btn_anim = Button(ax_anim, "Animate")
    textbox  = TextBox(ax_textbox, "ID: ", initial="0")
    btn_go   = Button(ax_go, "Go")

    # precompute background colors — they never change
    t_norm_all = (t_vals_global - t_min) / (t_max - t_min + 1e-8)
    bg_colors  = time_to_bg_color(t_norm_all)

    def draw(ci):
        elev, azim = ax.elev, ax.azim
        ax.cla()

        # background: black -> grey
        ax.scatter(
            all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
            c=bg_colors, s=2, alpha=0.2, zorder=1
        )

        # active patch: viridis encodes time
        patch_global_idxs = patch_idxs_2d[local_center_global[ci]].cpu().numpy()
        patch_pts  = st[patch_global_idxs]  # (G, 4)
        t_norm_patch = (patch_pts[:, 3] - t_min) / (t_max - t_min + 1e-8)
        patch_colors = time_to_cluster_color(t_norm_patch)
        ax.scatter(
            patch_pts[:, 0], patch_pts[:, 1], patch_pts[:, 2],
            c=patch_colors, s=15, alpha=0.9, zorder=2
        )

        # center marker: viridis color at its own timestep
        t_norm_center = (all_centers[ci, 3] - t_min) / (t_max - t_min + 1e-8)
        center_color  = viridis(t_norm_center)[:3]
        ax.scatter(
            all_centers[ci, 0], all_centers[ci, 1], all_centers[ci, 2],
            color=center_color, marker="x", s=120, linewidths=2.5, zorder=3
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Center {ci + 1} / {C_local}  |  t={all_centers[ci, 3]:.2f}  |  patch size {patch_size}")
        ax.view_init(elev=elev, azim=azim)
        textbox.set_val(str(ci))
        fig.canvas.draw_idle()

    def on_forward(_):
        if state["ci"] < C_local - 1:
            state["ci"] += 1
            draw(state["ci"])

    def on_back(_):
        if state["ci"] > 0:
            state["ci"] -= 1
            draw(state["ci"])

    def on_go(_):
        try:
            ci = int(textbox.text)
            if 0 <= ci < C_local:
                state["ci"] = ci
                draw(ci)
            else:
                ax.set_title(f"ID must be in [0, {C_local - 1}]")
                fig.canvas.draw_idle()
        except ValueError:
            ax.set_title("Invalid ID — enter an integer")
            fig.canvas.draw_idle()

    def animate_step(_):
        ci = np.random.randint(0, C_local)
        state["ci"] = ci
        draw(ci)
        return []

    def on_animate(_):
        if anim_state["running"]:
            anim_state["anim"].event_source.stop()
            anim_state["running"] = False
            btn_anim.label.set_text("Animate")
        else:
            anim_state["anim"] = animation.FuncAnimation(
                fig, animate_step, interval=700, blit=False, cache_frame_data=False
            )
            anim_state["running"] = True
            btn_anim.label.set_text("Stop")
        fig.canvas.draw_idle()

    btn_fwd.on_clicked(on_forward)
    btn_back.on_clicked(on_back)
    btn_go.on_clicked(on_go)
    btn_anim.on_clicked(on_animate)
    textbox.on_submit(lambda val: on_go(None))

    draw(0)
    plt.show()

class PointPatchEncoder(nn.Module):
    """
    Graph Neural Network encoder for point cloud trajectories.

    Takes multiple streams of point clouds over time and produces
    one embedding vector per trajectory.

    Pipeline:
    1. Tokenize each timestep's point cloud using PPRL tokenizer
    2. Build a spatio-temporal graph from tokens
    3. Batch all trajectory graphs together
    4. Process through message passing network
    5. Readout to get trajectory-level embeddings

    Input shape: (num_trajectories, num_timesteps, num_points, 3)
    Output shape: (num_trajectories, output_dim)
    """

    def __init__(
        self,
        output_dim: int,
        mlp_1: Callable[[int], nn.Linear],
        mlp_2: Callable[[int], nn.Linear],
        token_pos_encoder: Callable[[int, int], nn.Linear],
        pooling: Callable[[int, int], nn.Module],
        patch_size: int,
        oversampling_ratio: float,
        fps_random_start: bool = True,
        time_scale: float | None = None,
        patch_spatial_encoder: Callable[[int], nn.Module] | None = None,
        center_spatial_encoder: Callable[[int], nn.Module] | None = None,
        color_based_sampling: bool = False,
        color_based_collider_ratio: float = 0.5,
        color_based_num_centers: int = 256,
        example_batch=None,
        name: str = "pointpatch_encoder",
    ):
        """
        Initialize the GNN Encoder.

        Args:
            config: Configuration object with tokenizer, graph, mpn, and readout settings
            example_batch: Optional example batch for shape inference (unused, for API compatibility)
        """
        super().__init__()

        self.patch_size = patch_size
        self.oversampling_ratio = oversampling_ratio
        self.fps_random_start = fps_random_start
        self.time_scale = time_scale if time_scale is not None else float("inf")
        self.color_based_sampling = color_based_sampling
        self.color_based_collider_ratio = color_based_collider_ratio
        self.color_based_num_centers = color_based_num_centers

        # N_patches * patch_size = N_points * oversampling_ratio
        # fps_sampling_ratio = N_patches / N_points
        # -> fps_sampling_ratio = oversampling_ratio / patch_size
        self.fps_sampling_ratio = oversampling_ratio / patch_size

        # TODO: check if there is a color feature in example_batch to adjust input dims
        point_dim = 3
        spacetime_dim = point_dim + 1  # +1 for time dimension

        if patch_spatial_encoder is not None:
            patch_spatial_encoder = patch_spatial_encoder(point_dim)
            point_dim = patch_spatial_encoder.out_features
        self.patch_spatial_encoder = patch_spatial_encoder

        if center_spatial_encoder is not None:
            center_spatial_encoder = center_spatial_encoder(spacetime_dim)
            spacetime_dim = center_spatial_encoder.out_features
        self.center_spatial_encoder = center_spatial_encoder

        self.token_pos_encoder = token_pos_encoder(spacetime_dim, output_dim)

        self.mlp_1 = mlp_1(point_dim + example_batch["pc_color"].shape[-1])  # +dim for color feature
        self.mlp_2 = mlp_2(output_dim)

        if self.mlp_1.out_features * 2 != self.mlp_2.in_features:
            raise ValueError(
                f"The last layer of mlp_1 (size {self.mlp_1.out_features}) must be half the size of the first layer of mlp_2 (size {self.mlp_2.in_features})"
            )

        self.pooling = pooling(output_dim, output_dim)

        self._output_dim = output_dim

    def forward(self, pos: Tensor, color: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the GNN encoder.

        Args:
            xyzs: [num_trajectories, num_timesteps, num_points, 3]
                  Multiple point cloud trajectories

        Returns:
            embeddings: [num_trajectories, output_dim]
                       One embedding per trajectory
        """

        B, batch, pos, ptr, scaled_spacetime, spacetime = self.compute_spacetime(pos)

        if color is not None:
            color = color.flatten(0, -2)

        # ---------------------------
        # SAMPLING
        # ---------------------------

        if self.color_based_sampling and color is not None:
            k_total = self.color_based_num_centers
            center_idxs_list = []

            for b in range(B):
                start = ptr[b]
                end = ptr[b + 1]

                x_b = scaled_spacetime[start:end]
                color_b = color[start:end]

                collider_mask = color_b[:, 0] == 1
                mesh_mask = color_b[:, 0] == 0

                collider_idx_local = collider_mask.nonzero(as_tuple=False).view(-1)
                mesh_idx_local = mesh_mask.nonzero(as_tuple=False).view(-1)

                k_collider = int(k_total * self.color_based_collider_ratio)
                k_mesh = k_total - k_collider

                def sample_subset(idx_subset, k_subset):
                    # EMPTY CASE
                    if k_subset == 0:
                        return torch.empty((0,), dtype=torch.long, device=pos.device)
                    if idx_subset.numel() == 0:
                        # sample directly from full batch (already in correct index space)
                        return torch.randint(0, x_b.size(0), (k_subset,), device=pos.device)

                    x_subset = x_b[idx_subset]
                    ratio = min(1.0, k_subset / x_subset.size(0))

                    idx_local = fps(
                        x_subset,
                        ratio=ratio,
                        random_start=self.fps_random_start,
                    )

                    # enforce exact k
                    if idx_local.size(0) > k_subset:
                        idx_local = idx_local[:k_subset]
                    elif idx_local.size(0) < k_subset:
                        pad = idx_local[
                            torch.randint(0, idx_local.size(0), (k_subset - idx_local.size(0),))
                        ]
                        idx_local = torch.cat([idx_local, pad], dim=0)

                    # map back to batch indices
                    return idx_subset[idx_local]

                collider_samples = sample_subset(collider_idx_local, k_collider)
                mesh_samples = sample_subset(mesh_idx_local, k_mesh)
                idx_b = torch.cat([collider_samples, mesh_samples], dim=0)
                # optional shuffle to avoid ordering bias
                perm = torch.randperm(idx_b.size(0), device=idx_b.device)
                idx_b = idx_b[perm]

                center_idxs_list.append(idx_b + start)

            center_idxs = torch.cat(center_idxs_list, dim=0)

        else:
            # ratio-based FPS (original fast path)
            center_idxs = fps(
                scaled_spacetime,
                ptr=ptr,
                ratio=self.fps_sampling_ratio,
                random_start=self.fps_random_start,
            )

        # ---------------------------
        # CONTINUE PIPELINE
        # ---------------------------

        center_spacetime = spacetime[center_idxs]
        center_scaled_spacetime = scaled_spacetime[center_idxs]
        center_batch = batch[center_idxs]

        n_patches = center_idxs.size(0)

        _, patch_idxs = knn(
            x=scaled_spacetime,
            y=center_scaled_spacetime,
            k=self.patch_size,
            ptr_x=ptr,
            batch_y=center_batch,
            batch_size=B,
        )

        # debug_patch_assignments(
        #     spacetime,
        #     center_spacetime,
        #     patch_idxs,
        #     center_idxs.to("cpu"),
        #     n_patches,
        #     self.patch_size,
        #     batch_idx=0,
        #     L=L,
        # )

        patch_pos = pos[patch_idxs]
        patch_pos = patch_pos.view(n_patches, self.patch_size, 3)

        if color is not None:
            color = color[patch_idxs]
            color = color.view(n_patches, self.patch_size, -1)

        features = patch_pos

        if self.patch_spatial_encoder is not None:
            features = self.patch_spatial_encoder(features)

        if self.center_spatial_encoder is not None:
            processed_center_spacetime = self.center_spatial_encoder(center_spacetime)
        else:
            processed_center_spacetime = center_spacetime

        if color is not None:
            color = color.to(dtype=features.dtype)
            features = torch.cat([features, color], dim=-1)

        features = self.mlp_1(features)

        aggr_features = torch.max(features, dim=1, keepdim=True).values
        aggr_features = aggr_features.expand(-1, features.shape[1], -1)

        features = torch.cat([aggr_features, features], dim=-1)
        features = self.mlp_2(features)

        features = torch.max(features, dim=1).values  # (num_centers_total, D)

        processed_center_spacetime = self.token_pos_encoder(processed_center_spacetime)
        features += processed_center_spacetime

        # ---------------------------
        # SAFE UNFLATTEN
        # ---------------------------

        if self.color_based_sampling and color is not None:
            C = self.color_based_num_centers
        else:
            C = center_idxs.size(0) // B

        tokens = features.unflatten(0, (B, C))  # (B, C, D)
        center_scaled_spacetime = center_scaled_spacetime.unflatten(0, (B, C))  # (B, C, 4)

        embedding = self.pooling(tokens)
        embedding = embedding.squeeze(dim=1)

        return embedding, tokens, center_scaled_spacetime

    def compute_spacetime(self, pos: Tensor) -> tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, num_timesteps, num_points, _ = pos.shape
        L = num_timesteps * num_points

        pos = pos.flatten(0, -2)  # (B*T*N, 3)

        ptr = torch.arange(0, (B + 1) * L, step=L, device=pos.device)
        batch = ptr2batch(ptr)

        # time encoding
        time = torch.arange(0, num_timesteps, device=pos.device) / num_timesteps
        time = time.view(1, num_timesteps, 1, 1).repeat(B, 1, num_points, 1)
        time = time.flatten(0, -2)  # (B*T*N, 1)

        # spacetime
        spacetime = torch.cat([pos, time], dim=-1)
        scaled_spacetime = torch.cat([pos, time * self.time_scale], dim=-1)
        return B, batch, pos, ptr, scaled_spacetime, spacetime

    @property
    def output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self._output_dim


if __name__ == "__main__":
    import hydra

    with hydra.initialize(
        config_path="../../../../config/network/pc_encoder", version_base="1.3.2"
    ):
        config = hydra.compose(config_name="pointpatch_encoder")

    config = hydra.utils.instantiate(config)

    # Create example input
    # 2 trajectories, 10 timesteps, 512 points per timestep
    xyzs = torch.randn(2, 10, 512, 3)

    # Create model
    model = PointPatchEncoder(**config)

    # Forward pass
    output = model(xyzs)
    print(f"Input shape: {xyzs.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({xyzs.shape[0]}, {config.output_dim})")
