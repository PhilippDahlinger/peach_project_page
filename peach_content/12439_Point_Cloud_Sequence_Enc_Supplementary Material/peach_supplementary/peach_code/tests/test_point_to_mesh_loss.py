import torch
import pytest

from pc_mango.util.trampoline_real_world_eval.point_to_mesh_loss import compute_point_to_mesh_loss, \
    compute_point_to_mesh_loss2


def test_point_to_mesh_loss_equivalence():
    torch.manual_seed(42)

    T, N, V1, V2, F1, F2 = 3, 50, 20, 15, 10, 8

    pointcloud = torch.randn(T, N, 3)
    predicted_trajectory = torch.randn(T, V1, 3)
    predicted_collider = torch.randn(T, V2, 3)

    # Generate valid face indices
    faces_trajectory = torch.stack([
        torch.randint(0, V1, (F1,)),
        torch.randint(0, V1, (F1,)),
        torch.randint(0, V1, (F1,)),
    ], dim=1)  # (F1, 3)

    faces_collider = torch.stack([
        torch.randint(0, V2, (F2,)),
        torch.randint(0, V2, (F2,)),
        torch.randint(0, V2, (F2,)),
    ], dim=1)  # (F2, 3)

    loss1 = compute_point_to_mesh_loss(
        pointcloud, predicted_trajectory, faces_trajectory,
        predicted_collider, faces_collider
    )
    loss2 = compute_point_to_mesh_loss2(
        pointcloud, predicted_trajectory, faces_trajectory,
        predicted_collider, faces_collider
    )

    assert loss1.shape == loss2.shape == (T,), f"Shape mismatch: {loss1.shape} vs {loss2.shape}"
    assert torch.allclose(loss1, loss2, atol=1e-6), (
        f"Results differ:\n  v1: {loss1}\n  v2: {loss2}\n  max diff: {(loss1 - loss2).abs().max()}"
    )


def test_point_to_mesh_loss_empty_collider():
    """Edge case: collider with zero faces."""
    torch.manual_seed(0)

    T, N, V1, V2 = 2, 10, 12, 5
    F1 = 6

    pointcloud = torch.randn(T, N, 3)
    predicted_trajectory = torch.randn(T, V1, 3)
    predicted_collider = torch.randn(T, V2, 3)
    faces_trajectory = torch.randint(0, V1, (F1, 3))
    faces_collider = torch.zeros((0, 3), dtype=torch.long)

    loss1 = compute_point_to_mesh_loss(
        pointcloud, predicted_trajectory, faces_trajectory,
        predicted_collider, faces_collider
    )
    loss2 = compute_point_to_mesh_loss2(
        pointcloud, predicted_trajectory, faces_trajectory,
        predicted_collider, faces_collider
    )

    assert torch.allclose(loss1, loss2, atol=1e-6), (
        f"Empty collider case differs: max diff {(loss1 - loss2).abs().max()}"
    )