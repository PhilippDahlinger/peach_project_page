import torch
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt


def fit_sphere_algebraic(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a sphere to points using algebraic least squares.

    Solves the reparameterized sphere equation:
    x^2 + y^2 + z^2 = 2*cx*x + 2*cy*y + 2*cz*z + (r^2 - cx^2 - cy^2 - cz^2)

    This is linear in unknowns [cx, cy, cz, c0] where c0 = r^2 - cx^2 - cy^2 - cz^2
    Then recover r = sqrt(cx^2 + cy^2 + cz^2 + c0)

    Args:
        points: (num_points, 3) tensor of XYZ coordinates

    Returns:
        center: (3,) center point [cx, cy, cz]
        radius: scalar radius
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Build the system: A * params = b
    # params = [cx, cy, cz, c0]
    A = torch.stack([
        2 * x,
        2 * y,
        2 * z,
        torch.ones_like(x)
    ], dim=1)  # (num_points, 4)

    b = (x ** 2 + y ** 2 + z ** 2).unsqueeze(1)  # (num_points, 1)

    # Solve least squares: (A^T A)^-1 A^T b
    params = torch.linalg.lstsq(A, b).solution.squeeze()  # (4,)

    center = params[:3]
    c0 = params[3]

    # Recover radius
    radius = torch.sqrt(center[0] ** 2 + center[1] ** 2 + center[2] ** 2 + c0)

    return center, radius


def fit_sphere_known_radius(points: torch.Tensor, radius: torch.Tensor,
                            iters: int = 200, lr: float = 1e-2,
                            tol: float = 1e-6, verbose: bool = False) -> torch.Tensor:
    """
    Fit sphere center with known radius using nonlinear least squares
    with early stopping.

    Args:
        points: (N, 3)
        radius: known radius
        iters: max optimization steps
        lr: learning rate
        tol: convergence tolerance
        verbose: print progress

    Returns:
        center: (3,)
    """

    center = points.mean(dim=0).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([center], lr=lr)

    prev_loss = float("inf")

    for i in range(iters):
        optimizer.zero_grad()

        dists = torch.norm(points - center, dim=1)
        loss = ((dists - radius) ** 2).mean()

        loss.backward()

        prev_center = center.detach().clone()
        optimizer.step()

        # --- Early stopping checks ---
        loss_diff = abs(prev_loss - loss.item())
        center_shift = torch.norm(center.detach() - prev_center)

        if verbose:
            print(f"Iter {i:03d} | Loss: {loss.item():.6f} | Δloss: {loss_diff:.2e} | Δcenter: {center_shift:.2e}")

        if loss_diff < tol and center_shift < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break

        prev_loss = loss.item()

    return center.detach()


def fit_sphere_algebraic_robust(
        points: torch.Tensor,
        outlier_threshold: float = 2.0,
        max_iterations: int = 5,
        verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit sphere with iterative outlier rejection.

    Args:
        points: (num_points, 3) tensor
        outlier_threshold: reject points with residual > outlier_threshold * median_residual
        max_iterations: max iterations for outlier rejection
        verbose: print progress

    Returns:
        center: (3,) center point
        radius: scalar radius
    """
    mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    for iteration in range(max_iterations):
        # Fit on inliers
        center, radius = fit_sphere_algebraic(points[mask])

        # Compute residuals
        distances = torch.norm(points - center.unsqueeze(0), dim=1)
        residuals = torch.abs(distances - radius)

        # Reject outliers
        median_residual = torch.median(residuals[mask])
        new_mask = residuals < outlier_threshold * median_residual

        n_inliers = new_mask.sum().item()

        if verbose:
            print(f"[Robust fit] Iteration {iteration + 1}: {n_inliers} inliers, "
                  f"median residual: {median_residual.item():.6f}")

        # Check convergence
        if torch.equal(mask, new_mask):
            break

        mask = new_mask

    return center, radius


def fit_sphere_gradient_descent(
        points: torch.Tensor,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        verbose: bool = False,
        loss_type: str = "mse"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit sphere using gradient descent.

    Args:
        points: (num_points, 3) tensor
        learning_rate: GD step size
        max_iterations: max iterations
        convergence_threshold: stop if loss change < this
        verbose: print progress
        loss_type: "mse" (mean squared error) or "mae" (mean absolute error)

    Returns:
        center: (3,) optimized center
        radius: scalar optimized radius
    """
    device = points.device

    # Initialize: center as mean, radius as mean distance
    center = points.mean(dim=0).clone().requires_grad_(True)
    radius = torch.tensor(
        torch.norm(points - center.detach().unsqueeze(0), dim=1).mean().item(),
        device=device,
        requires_grad=True
    )

    # Optimizer
    optimizer = torch.optim.Adam([center, radius], lr=learning_rate)

    prev_loss = float('inf')

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Compute distances from center to all points
        distances = torch.norm(points - center.unsqueeze(0), dim=1)

        # Loss: how far are distances from radius?
        if loss_type == "mse":
            loss = torch.mean((distances - radius) ** 2)
        elif loss_type == "mae":
            loss = torch.mean(torch.abs(distances - radius))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        loss.backward()
        optimizer.step()

        # Ensure radius stays positive
        with torch.no_grad():
            radius.clamp_(min=1e-6)

        # Check convergence
        loss_change = abs(prev_loss - loss.item())

        if verbose and (iteration + 1) % 100 == 0:
            print(f"[GD] Iteration {iteration + 1}: loss={loss.item():.8f}, "
                  f"center={center.detach().cpu().numpy()}, radius={radius.item():.6f}")

        if loss_change < convergence_threshold:
            if verbose:
                print(f"[GD] Converged at iteration {iteration + 1}")
            break

        prev_loss = loss.item()

    return center.detach(), radius.detach()


def fit_sphere_ransac(
        points: torch.Tensor,
        n_iterations: int = 1000,
        inlier_threshold: float = 0.02,
        verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit sphere using RANSAC for robustness to outliers.

    Args:
        points: (num_points, 3) tensor
        n_iterations: number of RANSAC iterations
        inlier_threshold: residual threshold for inlier classification
        verbose: print progress

    Returns:
        center: (3,) best center
        radius: scalar best radius
    """
    device = points.device
    num_points = points.shape[0]

    best_center = None
    best_radius = None
    best_inlier_count = 0

    for iteration in range(n_iterations):
        # Random sample: 3 points define a sphere (under-constrained, but we'll fit anyway)
        # Better: sample random subset and fit
        sample_indices = torch.randperm(num_points, device=device)[:max(10, num_points // 10)]
        sample_points = points[sample_indices]

        # Fit sphere on sample
        try:
            center, radius = fit_sphere_algebraic(sample_points)
        except:
            continue

        # Ensure radius is positive
        if radius <= 0:
            continue

        # Count inliers
        distances = torch.norm(points - center.unsqueeze(0), dim=1)
        residuals = torch.abs(distances - radius)
        inlier_mask = residuals < inlier_threshold
        inlier_count = inlier_mask.sum().item()

        # Update best fit
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_center = center.clone()
            best_radius = radius.clone()

        if verbose and (iteration + 1) % 200 == 0:
            print(f"[RANSAC] Iteration {iteration + 1}: best inliers = {best_inlier_count}")

    if verbose:
        print(f"[RANSAC] Final: {best_inlier_count} inliers ({100 * best_inlier_count / num_points:.1f}%)")

    return best_center, best_radius


def plot_sphere_fit(
        points: torch.Tensor,
        center: torch.Tensor,
        radius: torch.Tensor,
        title: str = "Sphere Fitting",
        figsize: tuple = (10, 8),
        point_size: int = 20,
        alpha: float = 0.3,
        show_grid: bool = True
):
    """
    Visualize points, fitted center, and sphere surface.

    Args:
        points: (num_points, 3) tensor of XYZ coordinates
        center: (3,) tensor of sphere center
        radius: scalar tensor of sphere radius
        title: plot title
        figsize: figure size
        point_size: marker size for points
        alpha: transparency of sphere surface [0, 1]
        show_grid: show grid
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy
    points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
    center_np = center.cpu().numpy() if isinstance(center, torch.Tensor) else center
    radius_np = radius.item() if isinstance(radius, torch.Tensor) else radius

    # Plot points
    ax.scatter(
        points_np[:, 0], points_np[:, 1], points_np[:, 2],
        c='steelblue', s=point_size, alpha=0.6, label='Points'
    )

    # Plot center
    ax.scatter(
        center_np[0], center_np[1], center_np[2],
        c='red', s=200, marker='*', label='Center', edgecolors='darkred', linewidth=2
    )

    # Create sphere surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = radius_np * np.outer(np.cos(u), np.sin(v)) + center_np[0]
    y_sphere = radius_np * np.outer(np.sin(u), np.sin(v)) + center_np[1]
    z_sphere = radius_np * np.outer(np.ones(np.size(u)), np.cos(v)) + center_np[2]

    # Plot sphere surface
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere,
        alpha=alpha, color='cyan', edgecolor='none'
    )

    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')

    if show_grid:
        ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    # Get current limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Set equal ranges
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.set_aspect('equal')

    plt.show()
    return fig, ax


# Example usage and comparison
if __name__ == "__main__":
    # Create synthetic sphere with occlusion and noise
    num_points = 1000

    # True sphere parameters
    true_center = torch.tensor([1.0, 2.0, 3.0])
    true_radius = 5.0

    # Generate points on sphere
    theta = torch.rand(num_points) * 2 * np.pi
    phi = torch.rand(num_points) * np.pi * 0.7  # Occluded (only 0.7*pi range)

    x = true_radius * torch.sin(phi) * torch.cos(theta) + true_center[0]
    y = true_radius * torch.sin(phi) * torch.sin(theta) + true_center[1]
    z = true_radius * torch.cos(phi) + true_center[2]

    points = torch.stack([x, y, z], dim=1)

    # Add noise
    noise = torch.randn_like(points) * 0.1
    points += noise

    print(f"True center: {true_center}")
    print(f"True radius: {true_radius}")
    print(f"Num points: {num_points}\n")

    # Test different methods
    print("=" * 60)
    print("METHOD 1: Algebraic Least Squares")
    print("=" * 60)
    center_alg, radius_alg = fit_sphere_algebraic(points)
    print(f"Fitted center: {center_alg}")
    print(f"Fitted radius: {radius_alg.item():.6f}")
    error = torch.norm(center_alg - true_center).item()
    print(f"Center error: {error:.6f}\n")

    print("=" * 60)
    print("METHOD 2: Algebraic + Robust (Outlier Rejection)")
    print("=" * 60)
    center_robust, radius_robust = fit_sphere_algebraic_robust(points, verbose=True)
    print(f"Fitted center: {center_robust}")
    print(f"Fitted radius: {radius_robust.item():.6f}")
    error = torch.norm(center_robust - true_center).item()
    print(f"Center error: {error:.6f}\n")

    print("=" * 60)
    print("METHOD 3: Gradient Descent")
    print("=" * 60)
    center_gd, radius_gd = fit_sphere_gradient_descent(
        points, learning_rate=0.01, verbose=True
    )
    print(f"Fitted center: {center_gd}")
    print(f"Fitted radius: {radius_gd.item():.6f}")
    error = torch.norm(center_gd - true_center).item()
    print(f"Center error: {error:.6f}\n")

    print("=" * 60)
    print("METHOD 4: RANSAC")
    print("=" * 60)
    center_ransac, radius_ransac = fit_sphere_ransac(points, verbose=True)
    print(f"Fitted center: {center_ransac}")
    print(f"Fitted radius: {radius_ransac.item():.6f}")
    error = torch.norm(center_ransac - true_center).item()
    print(f"Center error: {error:.6f}\n")