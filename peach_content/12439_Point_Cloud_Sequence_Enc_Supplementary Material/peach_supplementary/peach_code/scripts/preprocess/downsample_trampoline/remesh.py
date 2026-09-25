"""
remesh.py

Remesh a flat mesh stored in an XDMF/H5 file using a regular checkerboard
pattern, interpolating node positions over time via barycentric coordinates.

Usage:
    python remesh.py <h5_path> <n_even>

Returns (when used as a module):
    positions: np.ndarray of shape [timesteps, num_resampled_nodes, 3]
"""

import h5py
import numpy as np
import sys


# ---------------------------------------------------------------------------
# 1. Generate resampled node positions (same pattern as mesh_faces.py)
# ---------------------------------------------------------------------------

def generate_resampled_nodes(n_even: int, x_min, x_max, y_min, y_max) -> np.ndarray:
    """
    Generate checkerboard node positions scaled to [x_min,x_max] x [y_min,y_max].
    Even rows: n_even nodes.  Odd rows: n_even-1 nodes, offset by half a cell.
    Returns array of shape (N, 2) with XY coordinates.
    """
    # Integer grid coordinates (same as before)
    max_i = 2 * (n_even - 1)
    nodes_2d = []
    for row in range(max_i + 1):
        if row % 2 == 0:
            xs = [2 * k for k in range(n_even)]
        else:
            xs = [2 * k + 1 for k in range(n_even - 1)]
        for x in xs:
            nodes_2d.append([x, row])
    nodes_2d = np.array(nodes_2d, dtype=float)

    # Scale to mesh bounding box
    nodes_2d[:, 0] = x_min + nodes_2d[:, 0] / max_i * (x_max - x_min)
    nodes_2d[:, 1] = y_min + nodes_2d[:, 1] / max_i * (y_max - y_min)
    return nodes_2d


# ---------------------------------------------------------------------------
# 2. Barycentric coordinate computation
# ---------------------------------------------------------------------------

def barycentric_coords(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    Compute barycentric coordinates (u, v, w) of point p in triangle (a, b, c).
    Works in 2D (XY only).
    Returns (u, v, w) with u+v+w=1, or None if the triangle is degenerate.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def find_triangle_and_bary(query_pts: np.ndarray,
                           verts_2d: np.ndarray,
                           faces: np.ndarray):
    """
    For each query point find which triangle it belongs to and compute
    barycentric coordinates.

    Parameters
    ----------
    query_pts : (M, 2)  XY coordinates of resampled nodes
    verts_2d  : (N, 2)  XY coordinates of original mesh nodes (z ignored)
    faces     : (F, 3)  triangle indices into verts_2d

    Returns
    -------
    tri_indices : (M,)  index of containing triangle (-1 if not found)
    bary_coords : (M, 3) barycentric weights (u, v, w)
    """
    M = len(query_pts)
    tri_indices = np.full(M, -1, dtype=int)
    bary_coords = np.zeros((M, 3), dtype=float)

    # Pre-extract triangle vertex arrays for speed
    A = verts_2d[faces[:, 0]]   # (F, 2)
    B = verts_2d[faces[:, 1]]
    C = verts_2d[faces[:, 2]]

    for i, p in enumerate(query_pts):
        # Vectorised containment test: check sign of cross products
        # A point is inside a CCW triangle if all cross products >= 0
        # (we also handle CW triangles by checking all <= 0)
        def cross2d(ox, oy, dx, dy, px, py):
            return (dx - ox) * (py - oy) - (dy - oy) * (px - ox)

        px, py = p
        c0 = (B[:, 0] - A[:, 0]) * (py - A[:, 1]) - (B[:, 1] - A[:, 1]) * (px - A[:, 0])
        c1 = (C[:, 0] - B[:, 0]) * (py - B[:, 1]) - (C[:, 1] - B[:, 1]) * (px - B[:, 0])
        c2 = (A[:, 0] - C[:, 0]) * (py - C[:, 1]) - (A[:, 1] - C[:, 1]) * (px - C[:, 0])

        inside = ((c0 >= -1e-10) & (c1 >= -1e-10) & (c2 >= -1e-10)) | \
                 ((c0 <= 1e-10) & (c1 <= 1e-10) & (c2 <= 1e-10))

        candidates = np.where(inside)[0]
        if len(candidates) == 0:
            print(f"Warning: node {i} at {p} not found in any triangle.")
            continue

        # Pick first valid candidate with proper barycentric coords
        for fi in candidates:
            a, b, c = A[fi], B[fi], C[fi]
            bary = barycentric_coords(p, a, b, c)
            if bary is None:
                continue
            tri_indices[i] = fi
            bary_coords[i] = bary
            break

    return tri_indices, bary_coords


# ---------------------------------------------------------------------------
# 3. Main remeshing function
# ---------------------------------------------------------------------------

def remesh(h5_path: str, n_even: int) -> np.ndarray:
    """
    Remesh the flat mesh in h5_path using a checkerboard grid with n_even
    nodes on even rows, then interpolate all timesteps.

    Parameters
    ----------
    h5_path : path to the .h5 file
    n_even  : number of nodes on even rows of the resampled grid

    Returns
    -------
    positions : np.ndarray of shape [timesteps, num_resampled_nodes, 3]
    """
    print(f"Loading mesh from {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        # Initial node positions (x, y, z) — z is always 0 (flat mesh)
        init_pos = f["data0"][:]          # (N, 3)
        faces    = f["data1"][:]          # (F, 3)  triangle connectivity

        # Find all displacement datasets: data2, data3, ...
        disp_keys = sorted(
            [k for k in f.keys() if k.startswith("data") and int(k[4:]) >= 2],
            key=lambda k: int(k[4:])
        )
        print(f"  {len(init_pos)} nodes, {len(faces)} triangles, {len(disp_keys)} timesteps")

        # Load all displacements into memory: shape (T, N, 3)
        displacements = np.stack([f[k][:] for k in disp_keys], axis=0)

    # Absolute positions at each timestep: shape (T, N, 3)
    abs_positions = init_pos[np.newaxis, :, :] + displacements  # broadcast over T

    # Bounding box from initial XY positions (z=0)
    verts_2d = init_pos[:, :2]
    x_min, x_max = verts_2d[:, 0].min(), verts_2d[:, 0].max()
    y_min, y_max = verts_2d[:, 1].min(), verts_2d[:, 1].max()
    print(f"  Mesh XY bbox: x=[{x_min:.4f}, {x_max:.4f}], y=[{y_min:.4f}, {y_max:.4f}]")

    # Generate resampled node positions in 2D
    print(f"Generating resampled grid (n_even={n_even}) ...")
    query_pts = generate_resampled_nodes(n_even, x_min, x_max, y_min, y_max)
    M = len(query_pts)
    print(f"  {M} resampled nodes")

    # Find triangle and barycentric coords for each resampled node
    print("Computing barycentric coordinates ...")
    tri_indices, bary_coords = find_triangle_and_bary(query_pts, verts_2d, faces)

    missing = np.sum(tri_indices == -1)
    if missing > 0:
        print(f"  Warning: {missing} nodes could not be mapped to a triangle.")

    # Interpolate absolute positions over all timesteps
    # For each timestep t and resampled node i in triangle (a,b,c) with bary (u,v,w):
    #   pos[t,i] = u * abs_pos[t,a] + v * abs_pos[t,b] + w * abs_pos[t,c]
    print("Interpolating positions over all timesteps ...")
    T = len(disp_keys)
    result = np.zeros((T, M, 3), dtype=np.float64)

    # Vectorised over timesteps and nodes
    fi   = tri_indices                    # (M,)   — triangle index per node
    u    = bary_coords[:, 0]             # (M,)
    v    = bary_coords[:, 1]
    w    = bary_coords[:, 2]

    idx_a = faces[fi, 0]                 # (M,)   — vertex indices
    idx_b = faces[fi, 1]
    idx_c = faces[fi, 2]

    # abs_positions: (T, N, 3)  →  gather per-node vertex positions
    pos_a = abs_positions[:, idx_a, :]   # (T, M, 3)
    pos_b = abs_positions[:, idx_b, :]
    pos_c = abs_positions[:, idx_c, :]

    result = (u[np.newaxis, :, np.newaxis] * pos_a +
              v[np.newaxis, :, np.newaxis] * pos_b +
              w[np.newaxis, :, np.newaxis] * pos_c)

    print(f"Done. Output shape: {result.shape}  (timesteps, resampled_nodes, xyz)")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remesh.py <h5_path> <n_even>")
        sys.exit(1)

    h5_path = sys.argv[1]
    n_even  = int(sys.argv[2])

    positions = remesh(h5_path, n_even)

    print(f"\nResult shape : {positions.shape}")
    print(f"t=0, node 0  : {positions[0, 0]}")
    print(f"t=-1, node 0 : {positions[-1, 0]}")

    # -------------------------------------------------------------------
    # Interactive 3D visualisation: close window to advance to next frame
    # -------------------------------------------------------------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    T = positions.shape[0]

    # Global axis limits so the view doesn't jump between frames
    lims = {
        "x": (positions[:, :, 0].min(), positions[:, :, 0].max()),
        "y": (positions[:, :, 1].min(), positions[:, :, 1].max()),
        "z": (positions[:, :, 2].min(), positions[:, :, 2].max()),
    }

    for t in range(T):
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")

        pts = positions[t]           # (M, 3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=pts[:, 2],      # colour by Z height
                   cmap="viridis", s=10, depthshade=True)

        ax.set_xlim(*lims["x"])
        ax.set_ylim(*lims["y"])
        ax.set_zlim(*lims["z"])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Timestep {t + 1} / {T}  (close to advance)")

        plt.tight_layout()
        plt.show()   # blocks until window is closed
        print(f"  closed t={t + 1}/{T}")