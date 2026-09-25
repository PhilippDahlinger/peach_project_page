import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import BaseTransform
import torch.nn as nn

from pc_mango.dataset.util.amg_generation_utils import init_adj_matrix, init_trimesh_levels_rn, \
    get_mesh_hierarchical_mapping_indices, get_node_attr_level, hierarchical_edge_index, \
    hierarchical_i_to_hierarchical_ip1_edge_index


class ModuleBaseTransform(nn.Module, BaseTransform):

    def __init__(self):
        super().__init__()


class AMGBuildHierarchicalGraph(ModuleBaseTransform):

    def __init__(
        self,
        max_coarse: int = 10,  # maximum number of coarse nodes
        max_mesh_level: int = 5,
    ):
        super().__init__()
        self.max_mesh_level = max_mesh_level
        self.max_coarse = max_coarse

    def forward(self, graph: Batch) -> Batch:

        adj = init_adj_matrix(graph["mesh"].const["pos"], graph["mesh", "mesh", "mesh"].edge_index)

        # print("nnz per row:", np.diff(adj.indptr))  # any zeros?
        # print("is connected:", np.sum(np.diff(adj.indptr) == 0), "isolated nodes")

        levels = init_trimesh_levels_rn(
            adj,
            max_coarse=self.max_coarse,
        )

        mesh_hir_mapping = get_mesh_hierarchical_mapping_indices(levels)
        pos_hir = graph["mesh"].const["pos"][mesh_hir_mapping]
        node_attr_level = get_node_attr_level(levels)
        edge_index_h = hierarchical_edge_index(levels)

        edge_index_hi_to_hip1 = hierarchical_i_to_hierarchical_ip1_edge_index([lvl.P for lvl in levels[:-1]])

        graph["hierarchical", "res", "hierarchical"].edge_index = edge_index_hi_to_hip1.long()
        graph["hierarchical", "prol", "hierarchical"].edge_index = edge_index_hi_to_hip1.flip(0).long()

        # node_type_mask_hir = {}
        # for key, value in graph["mesh"].bnd_nt_mask.items():
        #     node_type_mask_hir[key] = {k: v[mesh_hir_mapping] for k, v in value.items()}

        # graph["hierarchical"].bnd_nt_mask = node_type_mask_hir

        graph["hierarchical"].num_nodes = pos_hir.shape[0]

        graph["hierarchical"].const = {
            "pos": pos_hir,
            # "phys_nt": graph["mesh"].const["phys_nt"][mesh_hir_mapping],
            # "bnd_nt": {k: v[mesh_hir_mapping] for k, v in graph["mesh"].const["bnd_nt"].items()},
            # "mesh_nt": graph["mesh"].const["mesh_nt"][mesh_hir_mapping],
            "node_attr_level": node_attr_level,
        }

        graph["mesh", "map", "hierarchical"].edge_index = torch.stack(
            [mesh_hir_mapping, torch.arange(mesh_hir_mapping.shape[0])], dim=0
        ).long()
        graph["hierarchical", "mesh", "hierarchical"].edge_index = edge_index_h.long()

        # state_hir = {}
        # for key, value in graph["mesh"].state.items():
        #     state_hir[key] = (
        #         torch.tensor(value, dtype=torch.float)[mesh_hir_mapping]
        #         if isinstance(value, np.ndarray)
        #         else value.float()[mesh_hir_mapping]
        #     )

        # graph["hierarchical"].state = state_hir

        # state_bc_dir_hir = {}
        # for key, value in graph["mesh"].state_bc.items():
        #     state_bc_dir_hir[key] = (
        #         torch.tensor(value, dtype=torch.float)[mesh_hir_mapping]
        #         if isinstance(value, np.ndarray)
        #         else value.float()[mesh_hir_mapping]
        #     )
        # graph["hierarchical"].state_bc = state_bc_dir_hir

        graph["global_node"].num_levels = (node_attr_level.max() + 1).unsqueeze(0)
        graph["global_node"].max_mesh_level = torch.tensor(self.max_mesh_level).unsqueeze(0)

        return graph
