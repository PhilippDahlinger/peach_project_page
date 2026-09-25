import torch
import time

from tqdm import tqdm

from point_to_mesh_loss import (
    point_to_triangle_distance,
    point_to_triangle_distance_vectorized,
    compute_point_to_mesh_loss, point_to_triangle_distance_compiled
)


def compute_point_to_mesh_loss_non_vectorized(pointcloud, predicted_trajectory, faces_trajectory,
                                              predicted_collider, faces_collider):
    """
    Non-vectorized version for comparison.
    Computes loss for a single timestep (pointcloud and mesh are single timestep).
    """
    num_points = pointcloud.shape[0]
    num_faces_traj = faces_trajectory.shape[0]
    num_faces_coll = faces_collider.shape[0]

    min_dist_sq = torch.full((num_points,), float('inf'), dtype=pointcloud.dtype)

    # Check trajectory mesh
    for face_idx in tqdm(range(num_faces_traj)):
        v_indices = faces_trajectory[face_idx]
        v0 = predicted_trajectory[v_indices[0]]
        v1 = predicted_trajectory[v_indices[1]]
        v2 = predicted_trajectory[v_indices[2]]

        for point_idx in range(num_points):
            point = pointcloud[point_idx]
            dist_sq = point_to_triangle_distance(point, v0, v1, v2)
            min_dist_sq[point_idx] = min(min_dist_sq[point_idx], dist_sq.item())

    # Check collider mesh
    for face_idx in tqdm(range(num_faces_coll)):
        v_indices = faces_collider[face_idx]
        v0 = predicted_collider[v_indices[0]]
        v1 = predicted_collider[v_indices[1]]
        v2 = predicted_collider[v_indices[2]]

        for point_idx in range(num_points):
            point = pointcloud[point_idx]
            dist_sq = point_to_triangle_distance(point, v0, v1, v2)
            min_dist_sq[point_idx] = min(min_dist_sq[point_idx], dist_sq.item())

    loss = torch.mean(min_dist_sq).item()
    return loss


def compute_point_to_mesh_loss_vectorized_simple(pointcloud, predicted_trajectory, faces_trajectory,
                                                 predicted_collider, faces_collider):
    """
    Vectorized version for a single timestep.
    """
    num_points = pointcloud.shape[0]
    num_faces_traj = faces_trajectory.shape[0]
    num_faces_coll = faces_collider.shape[0]

    min_dist_sq = torch.full((num_points,), float('inf'), dtype=pointcloud.dtype)

    # Check trajectory mesh
    for face_idx in range(num_faces_traj):
        v_indices = faces_trajectory[face_idx]
        v0 = predicted_trajectory[v_indices[0]]
        v1 = predicted_trajectory[v_indices[1]]
        v2 = predicted_trajectory[v_indices[2]]

        dist_sq = point_to_triangle_distance_compiled(pointcloud, v0, v1, v2)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    # Check collider mesh
    for face_idx in range(num_faces_coll):
        v_indices = faces_collider[face_idx]
        v0 = predicted_collider[v_indices[0]]
        v1 = predicted_collider[v_indices[1]]
        v2 = predicted_collider[v_indices[2]]

        dist_sq = point_to_triangle_distance_compiled(pointcloud, v0, v1, v2)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

    loss = torch.mean(min_dist_sq).item()
    return loss


if __name__ == "__main__":
    # Test parameters
    num_points = 1000
    num_vertices_traj = 1201
    num_vertices_coll = 455
    num_faces_traj = 200
    num_faces_coll = 100

    print("=" * 60)
    print("Test: Point-to-Mesh Loss Computation")
    print("=" * 60)
    print(f"Number of points: {num_points}")
    print(f"Trajectory mesh: {num_vertices_traj} vertices, {num_faces_traj} faces")
    print(f"Collider mesh: {num_vertices_coll} vertices, {num_faces_coll} faces")
    print()

    # Generate random data
    pointcloud = torch.randn(num_points, 3) * 2.0  # Random points
    predicted_trajectory = torch.randn(num_vertices_traj, 3) * 2.0
    predicted_collider = torch.randn(num_vertices_coll, 3) * 2.0

    # Generate random triangle faces (valid indices)
    faces_trajectory = torch.randint(0, num_vertices_traj, (num_faces_traj, 3))
    faces_collider = torch.randint(0, num_vertices_coll, (num_faces_coll, 3))

    print("Computing non-vectorized loss...")
    start = time.time()
    loss_non_vec = compute_point_to_mesh_loss_non_vectorized(
        pointcloud, predicted_trajectory, faces_trajectory,
        predicted_collider, faces_collider
    )
    time_non_vec = time.time() - start
    print(f"Non-vectorized loss: {loss_non_vec:.6f}")
    print(f"Time: {time_non_vec:.4f}s")
    print()

    print("Computing vectorized loss...")
    start = time.time()
    loss_vec = compute_point_to_mesh_loss_vectorized_simple(
        pointcloud, predicted_trajectory, faces_trajectory,
        predicted_collider, faces_collider
    )
    time_vec = time.time() - start
    print(f"Vectorized loss: {loss_vec:.6f}")
    print(f"Time: {time_vec:.4f}s")
    print()

    print("=" * 60)
    print("Comparison:")
    print(f"Loss difference: {abs(loss_non_vec - loss_vec):.10f}")
    print(f"Speedup: {time_non_vec / time_vec:.2f}x")
    print("=" * 60)

    # Check if results match (within numerical tolerance)
    if abs(loss_non_vec - loss_vec) < 1e-5:
        print("✓ Results match! Vectorized implementation is correct.")
    else:
        print("✗ Results differ! Check the implementation.")