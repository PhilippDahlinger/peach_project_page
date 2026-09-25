import torch

import torch
from tqdm import tqdm


def point_to_triangle_distance(point, v0, v1, v2):
    """
    Compute shortest distance from a point to a triangle in 3D.
    Uses exact orthogonal projections to edges and vertices.
    Based on Christer Ericson's algorithm from "Real-Time Collision Detection".

    Args:
        point: (3,) tensor, the point
        v0, v1, v2: (3,) tensors, triangle vertices

    Returns:
        scalar tensor, squared distance from point to closest point on triangle
    """
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0

    d1 = torch.dot(ab, ap)
    d2 = torch.dot(ac, ap)

    # Region #1: closest to vertex v0
    if d1 <= 0.0 and d2 <= 0.0:
        closest = v0
    else:
        bp = point - v1
        d3 = torch.dot(ab, bp)
        d4 = torch.dot(ac, bp)

        # Region #2: closest to vertex v1
        if d3 >= 0.0 and d4 <= d3:
            closest = v1
        else:
            cp = point - v2
            d5 = torch.dot(ab, cp)
            d6 = torch.dot(ac, cp)

            # Region #3: closest to vertex v2
            if d6 >= 0.0 and d5 <= d6:
                closest = v2
            else:
                vc = d1 * d4 - d3 * d2

                # Region #4: closest to edge v0-v1
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    v = d1 / (d1 - d3)
                    closest = v0 + v * ab
                else:
                    vb = d5 * d2 - d1 * d6

                    # Region #5: closest to edge v0-v2
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        v = d2 / (d2 - d6)
                        closest = v0 + v * ac
                    else:
                        va = d3 * d6 - d5 * d4

                        # Region #6: closest to edge v1-v2
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = v1 + v * (v2 - v1)
                        else:
                            # Region #0: inside the triangle
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            closest = v0 + v * ab + w * ac

    # Squared distance
    dist_sq = torch.sum((point - closest) ** 2)
    return dist_sq

def point_to_triangle_distance_vectorized(points, v0, v1, v2):
    ab = v1 - v0
    ac = v2 - v0
    ap = points - v0
    bp = points - v1
    cp = points - v2

    d1 = torch.matmul(ap, ab)
    d2 = torch.matmul(ap, ac)
    d3 = torch.matmul(bp, ab)
    d4 = torch.matmul(bp, ac)
    d5 = torch.matmul(cp, ab)
    d6 = torch.matmul(cp, ac)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # Compute all edge/interior candidates simultaneously without masking
    # Edge v0-v1
    t4 = torch.clamp(d1 / (d1 - d3 + 1e-10), 0.0, 1.0)
    c4 = v0 + t4.unsqueeze(-1) * ab

    # Edge v0-v2
    t5 = torch.clamp(d2 / (d2 - d6 + 1e-10), 0.0, 1.0)
    c5 = v0 + t5.unsqueeze(-1) * ac

    # Edge v1-v2
    t6 = torch.clamp((d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-10), 0.0, 1.0)
    c6 = v1 + t6.unsqueeze(-1) * (v2 - v1)

    # Interior
    denom = va + vb + vc
    v_coord = vb / (denom + 1e-10)
    w_coord = vc / (denom + 1e-10)
    c0 = v0 + v_coord.unsqueeze(-1) * ab + w_coord.unsqueeze(-1) * ac

    # Candidate distances for all regions
    d_v0 = torch.sum((points - v0) ** 2, dim=1)
    d_v1 = torch.sum((points - v1) ** 2, dim=1)
    d_v2 = torch.sum((points - v2) ** 2, dim=1)
    d_e01 = torch.sum((points - c4) ** 2, dim=1)
    d_e02 = torch.sum((points - c5) ** 2, dim=1)
    d_e12 = torch.sum((points - c6) ** 2, dim=1)
    d_int = torch.sum((points - c0) ** 2, dim=1)

    # Region masks
    mask1 = (d1 <= 0.0) & (d2 <= 0.0)
    mask2 = (d3 >= 0.0) & (d4 <= d3)
    mask3 = (d6 >= 0.0) & (d5 <= d6)
    mask4 = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    mask5 = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    mask6 = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    # Select correct distance per point using torch.where (no dynamic indexing)
    dist = d_int  # default: interior
    dist = torch.where(mask6, d_e12, dist)
    dist = torch.where(mask5, d_e02, dist)
    dist = torch.where(mask4, d_e01, dist)
    dist = torch.where(mask3, d_v2, dist)
    dist = torch.where(mask2, d_v1, dist)
    dist = torch.where(mask1, d_v0, dist)

    return dist

# point_to_triangle_distance_compiled = torch.compile(point_to_triangle_distance_vectorized)
point_to_triangle_distance_compiled = point_to_triangle_distance_vectorized

def point_to_triangle_distance_vectorized2(points, v0, v1, v2):
    """
    points: (N,1,3)
    v0,v1,v2: (1,F,3)

    returns:
        (N,F)
    """

    ab = v1 - v0
    ac = v2 - v0

    ap = points - v0
    bp = points - v1
    cp = points - v2

    def dot(a, b):
        return torch.sum(a * b, dim=-1)

    d1 = dot(ap, ab)
    d2 = dot(ap, ac)
    d3 = dot(bp, ab)
    d4 = dot(bp, ac)
    d5 = dot(cp, ab)
    d6 = dot(cp, ac)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    eps = 1e-10

    # Edge v0-v1
    t4 = torch.clamp(d1 / (d1 - d3 + eps), 0.0, 1.0)
    c4 = v0 + t4.unsqueeze(-1) * ab

    # Edge v0-v2
    t5 = torch.clamp(d2 / (d2 - d6 + eps), 0.0, 1.0)
    c5 = v0 + t5.unsqueeze(-1) * ac

    # Edge v1-v2
    t6 = torch.clamp(
        (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps),
        0.0,
        1.0,
    )
    c6 = v1 + t6.unsqueeze(-1) * (v2 - v1)

    # Interior
    denom = va + vb + vc + eps
    v_coord = vb / denom
    w_coord = vc / denom

    c0 = (
        v0
        + v_coord.unsqueeze(-1) * ab
        + w_coord.unsqueeze(-1) * ac
    )

    # Distances
    d_v0 = torch.sum((points - v0) ** 2, dim=-1)
    d_v1 = torch.sum((points - v1) ** 2, dim=-1)
    d_v2 = torch.sum((points - v2) ** 2, dim=-1)

    d_e01 = torch.sum((points - c4) ** 2, dim=-1)
    d_e02 = torch.sum((points - c5) ** 2, dim=-1)
    d_e12 = torch.sum((points - c6) ** 2, dim=-1)

    d_int = torch.sum((points - c0) ** 2, dim=-1)

    # Region masks
    mask1 = (d1 <= 0.0) & (d2 <= 0.0)
    mask2 = (d3 >= 0.0) & (d4 <= d3)
    mask3 = (d6 >= 0.0) & (d5 <= d6)

    mask4 = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    mask5 = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)

    mask6 = (
        (va <= 0.0)
        & ((d4 - d3) >= 0.0)
        & ((d5 - d6) >= 0.0)
    )

    dist = d_int
    dist = torch.where(mask6, d_e12, dist)
    dist = torch.where(mask5, d_e02, dist)
    dist = torch.where(mask4, d_e01, dist)
    dist = torch.where(mask3, d_v2, dist)
    dist = torch.where(mask2, d_v1, dist)
    dist = torch.where(mask1, d_v0, dist)

    return dist

point_to_triangle_distance_compiled2 = torch.compile(point_to_triangle_distance_vectorized2)

def compute_point_to_mesh_loss(pointcloud, predicted_trajectory, faces_trajectory,
                               predicted_collider, faces_collider, verbose=False):
    """
    Compute point-to-mesh surface distance loss.

    For each timestep, for each point in the pointcloud, find the closest triangle
    across both mesh objects and compute the squared distance to that triangle surface.
    Average over all points per timestep.

    Args:
        pointcloud: (T, N, 3) tensor of point clouds over time
        predicted_trajectory: (T, V1, 3) tensor of trajectory mesh vertices over time
        faces_trajectory: (F1, 3) tensor of triangle indices for trajectory mesh
        predicted_collider: (T, V2, 3) tensor of collider mesh vertices over time
        faces_collider: (F2, 3) tensor of triangle indices for collider mesh
        verbose: print debug info

    Returns:
        (T,) tensor of losses, one per timestep
    """
    from tqdm import tqdm

    T = pointcloud.shape[0]
    num_points = pointcloud.shape[1]
    num_faces_traj = faces_trajectory.shape[0]
    num_faces_coll = faces_collider.shape[0]

    losses = []

    for t in tqdm(range(T), desc="Computing point-to-mesh loss over time"):
        pc_t = pointcloud[t]  # (N, 3)
        mesh_traj_t = predicted_trajectory[t]  # (V1, 3)
        mesh_coll_t = predicted_collider[t]  # (V2, 3)

        # Initialize min distances for all points
        min_dist_sq = torch.full((num_points,), float('inf'), dtype=pc_t.dtype, device=pc_t.device)

        # Check trajectory mesh
        for face_idx in range(num_faces_traj):
            v_indices = faces_trajectory[face_idx]  # (3,) indices
            v0 = mesh_traj_t[v_indices[0]]
            v1 = mesh_traj_t[v_indices[1]]
            v2 = mesh_traj_t[v_indices[2]]

            # Vectorized: compute distance from all points to this triangle
            dist_sq = point_to_triangle_distance_compiled(pc_t, v0, v1, v2)  # (N,)

            # Update minimum distances
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

        # Check collider mesh
        for face_idx in range(num_faces_coll):
            v_indices = faces_collider[face_idx]  # (3,) indices
            v0 = mesh_coll_t[v_indices[0]]
            v1 = mesh_coll_t[v_indices[1]]
            v2 = mesh_coll_t[v_indices[2]]

            # Vectorized: compute distance from all points to this triangle
            dist_sq = point_to_triangle_distance_compiled(pc_t, v0, v1, v2)  # (N,)

            # Update minimum distances
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

        # Average over points
        loss_t = torch.mean(min_dist_sq)
        losses.append(loss_t)

        if verbose:
            print(f"Timestep {t}: loss = {loss_t:.6f}")

    return torch.stack(losses)



def compute_point_to_mesh_loss2(pointcloud, predicted_trajectory, faces_trajectory,
                               predicted_collider, faces_collider, compiled=True, verbose=False):
    """
    Compute point-to-mesh surface distance loss.

    For each timestep, for each point in the pointcloud, find the closest triangle
    across both mesh objects and compute the squared distance to that triangle surface.
    Average over all points per timestep.

    Args:
        pointcloud: (T, N, 3) tensor of point clouds over time
        predicted_trajectory: (T, V1, 3) tensor of trajectory mesh vertices over time
        faces_trajectory: (F1, 3) tensor of triangle indices for trajectory mesh
        predicted_collider: (T, V2, 3) tensor of collider mesh vertices over time
        faces_collider: (F2, 3) tensor of triangle indices for collider mesh
        verbose: print debug info

    Returns:
        (T,) tensor of losses, one per timestep
    """
    from tqdm import tqdm

    T = pointcloud.shape[0]
    num_points = pointcloud.shape[1]
    num_faces_traj = faces_trajectory.shape[0]
    num_faces_coll = faces_collider.shape[0]

    losses = []

    for t in range(T):
        pc_t = pointcloud[t]  # (N, 3)
        mesh_traj_t = predicted_trajectory[t]  # (V1, 3)
        mesh_coll_t = predicted_collider[t]  # (V2, 3)

        # Initialize min distances for all points
        min_dist_sq = torch.full((num_points,), float('inf'), dtype=pc_t.dtype, device=pc_t.device)

        # Check trajectory mesh
        triangles = mesh_traj_t[faces_trajectory]  # (F, 3, 3)

        v0 = triangles[:, 0]  # (F, 3)
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]
        if compiled:
            dist_sq = point_to_triangle_distance_compiled2(
                pc_t.unsqueeze(1),  # (N,1,3)
                v0.unsqueeze(0),  # (1,F,3)
                v1.unsqueeze(0),
                v2.unsqueeze(0),
            )  # (N,F)
        else:
            dist_sq = point_to_triangle_distance_vectorized2(
                pc_t.unsqueeze(1),  # (N,1,3)
                v0.unsqueeze(0),  # (1,F,3)
                v1.unsqueeze(0),
                v2.unsqueeze(0),
            )  # (N,F)

        min_dist_sq = torch.minimum(
            min_dist_sq,
            dist_sq.min(dim=1).values
        )

        # Check collider mesh
        if num_faces_coll > 0:
            coll_triangles = mesh_coll_t[faces_collider]  # (F,3,3)

            v0 = coll_triangles[:, 0]  # (F,3)
            v1 = coll_triangles[:, 1]
            v2 = coll_triangles[:, 2]

            if compiled:
                dist_sq = point_to_triangle_distance_compiled2(
                    pc_t.unsqueeze(1),  # (N,1,3)
                    v0.unsqueeze(0),  # (1,F,3)
                    v1.unsqueeze(0),
                    v2.unsqueeze(0),
                )  # (N,F)
            else:
                dist_sq = point_to_triangle_distance_vectorized2(
                    pc_t.unsqueeze(1),  # (N,1,3)
                    v0.unsqueeze(0),  # (1,F,3)
                    v1.unsqueeze(0),
                    v2.unsqueeze(0),
                )  # (N,F)

            min_dist_sq = torch.minimum(
                min_dist_sq,
                dist_sq.min(dim=1).values
            )

        # Average over points
        loss_t = torch.mean(min_dist_sq)
        losses.append(loss_t)

        if verbose:
            print(f"Timestep {t}: loss = {loss_t:.6f}")

    return torch.stack(losses)


if __name__ == "__main__":
    # Example usage
    point = torch.tensor([-1 , -1, 2.0])
    v0 = torch.tensor([0.0, 0.0, 0.0])
    v1 = torch.tensor([2.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 2.0, 0.0])

    dist_sq = point_to_triangle_distance(point, v0, v1, v2)
    print(f"Squared distance from point to triangle: {dist_sq.item()}")