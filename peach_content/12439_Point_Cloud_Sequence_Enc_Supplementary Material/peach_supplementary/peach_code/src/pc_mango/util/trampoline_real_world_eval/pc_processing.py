import fpsample
import torch
from scipy.spatial import cKDTree

from pc_mango.util.trampoline_real_world_eval.sphere_fitting import fit_sphere_algebraic, fit_sphere_known_radius
from pc_mango.visualization.debug_pc_plot import visualize_pointcloud_sequence


def rgb_to_hsv(rgb):
    """
    Convert RGB to HSV.

    Args:
        rgb: tensor of shape (..., 3) with values in [0, 1]

    Returns:
        hsv: tensor of shape (..., 3) with H in [0, 1], S in [0, 1], V in [0, 1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_val = torch.max(rgb, dim=-1)[0]
    min_val = torch.min(rgb, dim=-1)[0]
    delta = max_val - min_val

    # Value
    v = max_val

    # Saturation
    s = torch.where(max_val != 0, delta / max_val, torch.zeros_like(delta))

    # Hue
    h = torch.zeros_like(delta)

    mask_r = (max_val == r) & (delta != 0)
    mask_g = (max_val == g) & (delta != 0)
    mask_b = (max_val == b) & (delta != 0)

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    # Normalize H to [0, 1]
    h = h / 360

    return torch.stack([h, s, v], dim=-1)


def classify_pink_hsv(colors, hue_range=(300, 60), saturation_min=0.2, value_min=0.2):
    """
    Classify points as pink using HSV color space.

    Args:
        colors: RGB tensor in [0, 1], shape (batch_dim, num_points, 3)
        hue_range: tuple (min_hue, max_hue) in degrees. Pink wraps: (330, 30)
        saturation_min: minimum saturation for pink
        value_min: minimum brightness

    Returns:
        type_tensor: shape (batch_dim, num_points) with 0 or 1
    """
    batch_dim, num_points, _ = colors.shape

    # Reshape to (batch_dim * num_points, 3) for conversion
    colors_flat = colors.reshape(-1, 3)

    # Convert RGB to HSV using torchvision
    # Input shape: (N, 3), output shape: (N, 3)
    hsv_flat = rgb_to_hsv(colors_flat)

    # Reshape back to (batch_dim, num_points, 3)
    hsv = hsv_flat.reshape(batch_dim, num_points, 3)

    # Extract H, S, V
    # H is in [0, 1] range, convert to degrees [0, 360]
    h_deg = hsv[..., 0] * 360
    s = hsv[..., 1]
    v = hsv[..., 2]

    # Pink has hue wrapping around 0°/360°
    # Check if hue is in (330°, 360°) OR (0°, 30°)
    h_min, h_max = hue_range
    in_hue_range = (h_deg >= h_min) | (h_deg <= h_max)

    # Check saturation and value thresholds
    in_sat_range = (s >= saturation_min)
    in_val_range = (v >= value_min)

    # Combine all conditions
    type_tensor = (in_hue_range & in_sat_range & in_val_range).long()

    return type_tensor


def estimate_pink_reference(colors, initial_mask, verbose=False):
    """
    Estimate pink reference color from initial pink points.

    Args:
        colors: (batch_dim, num_points, 3) RGB tensor
        initial_mask: (batch_dim, num_points) binary tensor

    Returns:
        pink_ref_rgb: (3,) reference color in [0, 1]
        pink_ref_hsv: (3,) reference color in HSV
    """
    # Get all pink points across batch
    pink_points = colors[initial_mask > 0]

    if pink_points.shape[0] == 0:
        raise ValueError("No initial pink points found!")

    # Mean color of pink points
    pink_ref_rgb = pink_points.mean(dim=0)
    pink_ref_hsv = rgb_to_hsv(pink_ref_rgb.unsqueeze(0)).squeeze(0)

    if verbose:
        print(f"Estimated pink reference RGB: {pink_ref_rgb.cpu().numpy()}")
        print(f"Estimated pink reference HSV: {pink_ref_hsv.cpu().numpy()}")

    return pink_ref_rgb, pink_ref_hsv


def color_distance_rgb(colors, pink_ref_rgb):
    """Euclidean distance in RGB space"""
    return torch.norm(colors - pink_ref_rgb, dim=-1)


def color_distance_hsv(colors, pink_ref_hsv):
    """Euclidean distance in HSV space (with hue weighting)"""
    hsv = rgb_to_hsv(colors)

    # Weight components: hue is perceptually important
    h_dist = torch.abs(hsv[..., 0] - pink_ref_hsv[0])
    # Handle hue wrapping at 0/360
    h_dist = torch.min(h_dist, 1 - h_dist)

    s_dist = torch.abs(hsv[..., 1] - pink_ref_hsv[1])
    v_dist = torch.abs(hsv[..., 2] - pink_ref_hsv[2])

    # Combined distance with hue weighted higher
    dist = torch.sqrt((2.0 * h_dist) ** 2 + s_dist ** 2 + v_dist ** 2)
    return dist


def iterative_pink_segmentation_knn(
        colors,
        points,
        pink_ref_rgb,
        pink_ref_hsv,
        n_neighbors=10,
        max_iterations=10,
        init_threshold=0.15,
        iter_threshold=0.7,
        color_weight=0.5,
        neighbor_weight=0.5,
        verbose=True
):
    """
    Iteratively segment pink points using KNN-based neighbor analysis.

    Args:
        colors: (batch_dim, num_points, 3) RGB tensor
        points: (batch_dim, num_points, 3) XYZ coordinates
        pink_ref_rgb: (3,) reference pink color RGB
        pink_ref_hsv: (3,) reference pink color HSV
        n_neighbors: number of KNN neighbors
        max_iterations: maximum iterations
        init_threshold: initial color distance threshold
        iter_threshold: threshold for subsequent iterations
        color_weight: weight for color distance [0, 1]
        neighbor_weight: weight for neighbor density [0, 1]
        verbose: print progress

    Returns:
        pink_mask: (batch_dim, num_points) binary tensor
    """
    batch_dim, num_points, _ = colors.shape
    device = colors.device

    # Compute color distances
    metrics = []
    if pink_ref_rgb is not None:
        rgb_dist = color_distance_rgb(colors, pink_ref_rgb)
        metrics.append(rgb_dist)
    if pink_ref_hsv is not None:
        hsv_dist = color_distance_hsv(colors, pink_ref_hsv)
        metrics.append(hsv_dist)
    color_dist = torch.stack(metrics, dim=-1).mean(dim=-1)  # Average of available metrics

    # Normalize color distance to [0, 1]
    color_dist_norm = color_dist / (color_dist.max() + 1e-6)
    color_score = 1 - color_dist_norm  # Invert: high distance = low score

    # Initialize: loose threshold
    pink_mask = (color_dist < init_threshold).float()

    if verbose:
        print(f"[KNN] Initial pink points: {pink_mask.sum().item()}")

    # visualize_pointcloud_sequence(points, pink_mask[:, :, None])
    # Iterative refinement
    for iteration in range(max_iterations):
        # visualize_pointcloud_sequence(points, pink_mask[:, :, None])

        new_mask = pink_mask.clone()

        for batch_idx in range(batch_dim):
            pts = points[batch_idx].cpu().numpy()

            # Build KDTree for this batch
            tree = cKDTree(pts)

            # For each point, find neighbors
            _, neighbor_indices = tree.query(pts, k=min(n_neighbors + 1, num_points))
            neighbor_indices = neighbor_indices[:, 1:]  # Remove self

            # Count pink neighbors
            pink_neighbor_count = pink_mask[batch_idx, neighbor_indices].sum(dim=1)
            neighbor_density = pink_neighbor_count / neighbor_indices.shape[1]

            # Combine color score + neighbor density
            combined_score = (
                    color_weight * color_score[batch_idx] +
                    neighbor_weight * neighbor_density
            )

            # Update mask
            new_mask[batch_idx] = (combined_score > iter_threshold).float()

        # Check convergence
        n_added = (new_mask - pink_mask).sum().item()
        n_pink = new_mask.sum().item()

        if verbose:
            print(f"[KNN] Iteration {iteration + 1}: {n_pink:.0f} pink points, +{n_added:.0f} new")

        if n_added == 0:
            if verbose:
                print(f"[KNN] Converged at iteration {iteration + 1}")
            break

        pink_mask = new_mask
    # visualize_pointcloud_sequence(points, pink_mask[:, :, None])


    return pink_mask.long()


def crop_robot_out_and_subsample(points, types, sphere_id, margin=0.05, z_threshold=0.5, final_num_points=512):
    output_points = []
    output_types = []
    output_center = []
    try:
        from pc_mango.util.trampoline_real_world_eval.real_world_evaluator import RealWorldEvaluator
        radius = torch.tensor(RealWorldEvaluator.diameter_dict[sphere_id]) / 2.0
    except KeyError:
        raise KeyError(f"Unknown sphere_id: {sphere_id}")
    for pc, type in zip(points, types):
        # Fit sphere to pink points
        pink_points = pc[type == 1]
        center = fit_sphere_known_radius(pink_points, radius, iters=500, lr=1.5, tol=1e-5, verbose=False)
        # visualize sphere
        # crop points outside the sphere + margin and above z threshold
        dists = torch.norm(pc - center, dim=-1)
        mask = (dists < radius + margin) | (pc[:, 2] < z_threshold)
        cropped_points = pc[mask]
        new_pink_mask = (dists < radius + margin)
        # create type tensor out of it
        cropped_types = new_pink_mask.long()
        cropped_types = cropped_types[mask]

        # visualize_pointcloud_sequence(
        #     pointcloud=cropped_points[None, :, :],
        #     types=cropped_types[None, :, None],
        #     center=center.cpu().numpy(),
        #     radius=radius.cpu().numpy(),
        # )

        # do fpsubsample
        assert cropped_points.shape[0] > final_num_points, f"Not enough points after cropping: {cropped_points.shape[0]}"
        indices = fpsample.fps_sampling(cropped_points, final_num_points)
        cropped_points = cropped_points[indices]
        cropped_types = cropped_types[indices]
        output_points.append(cropped_points)
        output_types.append(cropped_types)
        output_center.append(center)
    return torch.stack(output_points), torch.stack(output_types), torch.stack(output_center)

