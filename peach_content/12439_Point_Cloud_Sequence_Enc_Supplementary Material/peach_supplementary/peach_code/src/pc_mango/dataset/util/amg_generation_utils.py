import numpy as np
import torch
from scipy.sparse import csr_matrix
from pyamg import rootnode_solver
from pyamg.multilevel import MultilevelSolver


def init_adj_matrix(node_pos: np.ndarray | torch.Tensor, edge_index: np.ndarray | torch.Tensor) -> csr_matrix:
    """Initialize a adjacency matrix in CSR format from edges."""
    if isinstance(node_pos, torch.Tensor):
        node_pos = node_pos.detach().cpu().numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    n = int(node_pos.shape[0])
    rows = edge_index[0].astype(np.int64, copy=False)
    cols = edge_index[1].astype(np.int64, copy=False)

    data = np.ones(rows.shape[0], dtype=np.int64)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    A.sum_duplicates()
    A.data[:] = 1
    return A


def init_trimesh_levels_rn(A, num_trials: int = 1000, **cfg):
    """Initialize the multigrid levels for a trimesh."""
    # Some numerical issues can occur inside rootnode in rare situations, so we may have to try multiple times
    for i in range(num_trials):
        try:
            ml = rootnode_solver(A, keep=True, **cfg)
            break
        except Exception:
            pass
    levels = ml.levels
    return levels


def get_mesh_hierarchical_mapping_indices(levels: list[MultilevelSolver.Level]) -> torch.Tensor:
    """Get the mapping indices for the mapping from mesh nodes to hierarchical nodes.

    This function creates a tensor that maps the hierarchical mesh nodes back to the fine mesh nodes.
    That means, that each node in the hierarchical mesh is assigned an index that corresponds to its position in the fine mesh.
    The first level (finest) is assigned indices from 0 to the number of fine mesh nodes minus one.
    The second level indices are derived from the coarse points of the first level,
    and so on for subsequent levels.

    Args:
        levels (list[MultilevelSolver.Level]): List of multilevel solver levels.

    Returns:
        torch.Tensor: A tensor containing the mapping indices from hierarchical mesh nodes to fine mesh nodes.
    """

    indices_lvl = torch.arange(levels[0].A.shape[0])
    indices = []
    for lvl in range(len(levels)):
        indices.append(indices_lvl)
        if lvl < len(levels) - 1:
            coarse_points = levels[lvl].Cpts if hasattr(levels[lvl], "Cpts") else levels[lvl].splitting
            indices_lvl = indices_lvl[coarse_points]

    return torch.cat(indices, dim=0)


def get_node_attr_level(levels: list[MultilevelSolver.Level]) -> torch.Tensor:
    """Get a tensor containing the level of each node.

    This function creates a tensor where each node is assigned a level index based on the multilevel solver levels.
    The first level (finest) is assigned index 0, the second level index 1, and so on.
    For example, a output tensor for three levels might look like:
    tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])

    Args:
        levels (list[MultilevelSolver.Level]): List of multilevel solver levels.

    Returns:
        torch.Tensor: A tensor where each node is assigned a level index.
    """
    node_attr_level = [torch.zeros(levels[0].P.shape[0], dtype=torch.long)]
    for i, lvl in enumerate(levels[:-1]):
        node_attr_level.append(torch.ones(lvl.P.shape[1], dtype=torch.long) * (i + 1))
    node_attr_level = torch.cat(node_attr_level)
    return node_attr_level


def hierarchical_edge_index_list(levels: list[MultilevelSolver.Level]):
    """Get the edge index for the hierarchical mesh."""
    edge_index_hir = []
    start_idx_hir = 0
    for i, lvl in enumerate(levels):
        edge_index_hir.append(torch.tensor(np.array(lvl.A.nonzero())) + start_idx_hir)
        start_idx_hir += lvl.A.shape[0]
    return edge_index_hir


def hierarchical_edge_index(levels: list[MultilevelSolver.Level]):
    """Get the edge index for the hierarchical mesh."""
    return torch.cat(hierarchical_edge_index_list(levels), dim=1).long()


def hierarchical_i_to_hierarchical_ip1_edge_index_list(P_list):
    """Get the edge index for the mapping between the mesh nodes of two levels."""
    edge_index_lvl = []
    start_idx_lvl = 0
    for i, P in enumerate(P_list):
        edge_idx = torch.tensor(np.array(P.nonzero()))
        edge_idx[0] = edge_idx[0] + start_idx_lvl
        start_idx_lvl += P.shape[0]
        edge_idx[1] = edge_idx[1] + start_idx_lvl
        edge_index_lvl.append(edge_idx)
    return edge_index_lvl


def hierarchical_i_to_hierarchical_ip1_edge_index(P_list):
    """Get the edge index for the mapping between the mesh nodes of two levels."""
    return torch.cat(hierarchical_i_to_hierarchical_ip1_edge_index_list(P_list), dim=1).long()
