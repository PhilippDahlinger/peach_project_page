import numpy as np


def generate_mesh(n_even):
    """
    Generate mesh for the checkerboard grid pattern.

    n_even: number of points on the bottom (even) edge
            e.g. n_even=3 gives the original 13-point grid

    Points are indexed left-to-right, bottom-to-top, row by row.
    Even rows: x = 0, 2, 4, ..., 2*(n_even-1)
    Odd rows:  x = 1, 3, 5, ..., 2*(n_even-1)-1  → (n_even-1) points
    Total rows: 2*(n_even-1) + 1
    """
    max_x = 2 * (n_even - 1)
    max_y = 2 * (n_even - 1)

    # Build point list and lookup dict, indexed left-to-right bottom-to-top
    point_dict = {}  # (x, y) -> index
    points = []

    for row in range(max_y + 1):
        if row % 2 == 0:
            xs = range(0, max_x + 1, 2)
        else:
            xs = range(1, max_x, 2)
        for x in xs:
            point_dict[(x, row)] = len(points)
            points.append((x, row))

    # Generate triangles
    # Each "cell" is centered on an odd point at (x, y) where y is odd.
    # The odd point + its 4 even corners form 4 triangles:
    #   (odd, bottom-left, bottom-right)  — bottom two corners
    #   (odd, bottom-right, top-right)    — right two corners
    #   (odd, top-right, top-left)        — top two corners
    #   (odd, top-left, bottom-left)      — left two corners
    # This gives a consistent CCW winding.

    faces = []

    for (x, y), idx in point_dict.items():
        if y % 2 == 1:  # odd point — center of a quad cell
            bl = (x - 1, y - 1)
            br = (x + 1, y - 1)
            tr = (x + 1, y + 1)
            tl = (x - 1, y + 1)
            if all(c in point_dict for c in [bl, br, tr, tl]):
                o = idx
                i_bl = point_dict[bl]
                i_br = point_dict[br]
                i_tr = point_dict[tr]
                i_tl = point_dict[tl]
                # 4 triangles, CCW winding
                faces.append((o, i_bl, i_br))
                faces.append((o, i_br, i_tr))
                faces.append((o, i_tr, i_tl))
                faces.append((o, i_tl, i_bl))

    return points, faces


def visualize_mesh(points, faces, n_even, show_indices=True):
    """
    Visualize the mesh with nodes, edges, and optional node indices.

    show_indices: label each node with its index (recommended only for small grids)
    """
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc

    pts = np.array(points)

    # Collect unique edges from faces
    edges = set()
    for a, b, c in faces:
        for i, j in [(a, b), (b, c), (c, a)]:
            edges.add((min(i, j), max(i, j)))

    lines = [[pts[i], pts[j]] for i, j in edges]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges
    lc = mc.LineCollection(lines, colors='steelblue', linewidths=0.8, alpha=0.8)
    ax.add_collection(lc)

    # Draw nodes
    ax.scatter(pts[:, 0], pts[:, 1], color='crimson', s=20, zorder=5)

    # Label nodes (only practical for small grids)
    if show_indices:
        for i, (x, y) in enumerate(points):
            ax.annotate(str(i), (x, y),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=7, color='#222')

    ax.set_xlim(-1, pts[:, 0].max() + 1)
    ax.set_ylim(-1, pts[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.set_title(f'Mesh — n_even={n_even}, {len(points)} nodes, '
                 f'{len(faces)} faces, {len(edges)} edges',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    n_even = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    points, faces = generate_mesh(n_even)

    print(f"Grid with {n_even} points on even edge")
    print(f"Total nodes : {len(points)}")
    print(f"Total faces : {len(faces)}")

    print("\nNodes (index: x, y):")
    for i, (x, y) in enumerate(points):
        print(f"  {i:4d}: ({x}, {y})")

    print("\nFaces (n0, n1, n2):")
    for i, (a, b, c) in enumerate(faces):
        print(f"  {i:4d}: ({a}, {b}, {c})")

    # Visualize — disable index labels for large grids
    visualize_mesh(points, faces, n_even, show_indices=(n_even <= 5))