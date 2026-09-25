from copy import deepcopy

from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset

try:
    import torch_cluster
    HAS_TORCH_CLUSTER = True
except (ImportError, OSError):
    HAS_TORCH_CLUSTER = False

import torch
import torch_geometric.transforms as T

from pc_mango.ggns.util.Types import *
from pc_mango.ggns.src.utils.get_collider_triangles import get_tube_collider_triangles


def fallback_radius_graph(pos: torch.Tensor, r: float, max_num_neighbors: int = 100) -> torch.Tensor:
    """
    Fallback radius graph implementation using pure PyTorch (when torch_cluster is not available).
    Creates edges between nodes within radius r of each other.

    Args:
        pos: Node positions of shape (N, D)
        r: Radius for edge creation
        max_num_neighbors: Maximum number of neighbors to connect

    Returns:
        edge_index: Edge indices of shape (2, E)
    """
    # Compute pairwise distances using broadcasting
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, D)
    dist = torch.norm(diff, dim=2)  # (N, N)

    # Find edges where distance <= r, excluding self-loops
    edge_mask = (dist <= r) & (dist > 0)  # (N, N)

    # Get all valid edges
    sources, targets = torch.where(edge_mask)

    if sources.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)

    # Limit neighbors per source node efficiently
    if max_num_neighbors > 0 and sources.shape[0] > 0:
        # Get distances for valid edges only
        edge_dists = dist[sources, targets]  # (E,)

        # Sort indices by (source, distance) to group by source and keep closest neighbors
        source_order = sources.argsort(stable=True)
        sources_sorted = sources[source_order]
        targets_sorted = targets[source_order]
        edge_dists_sorted = edge_dists[source_order]

        # Find where source changes and limit neighbors per source
        source_changes = torch.cat([
            torch.tensor([1], dtype=torch.bool, device=pos.device),
            sources_sorted[1:] != sources_sorted[:-1]
        ])

        # Count edges per source and create mask for keeping top max_num_neighbors
        cumsum = torch.cumsum(source_changes.float(), dim=0)  # Group number for each edge
        cumsum_within_group = torch.arange(sources_sorted.shape[0], device=pos.device, dtype=torch.long) - torch.cat([
            torch.tensor([0], dtype=torch.long, device=pos.device),
            torch.cumsum(source_changes[:-1].long(), dim=0)
        ])

        keep_mask = cumsum_within_group < max_num_neighbors

        sources = sources_sorted[keep_mask]
        targets = targets_sorted[keep_mask]

    edge_index = torch.stack([sources, targets], dim=0)
    return edge_index.long()


def fallback_radius(pos_receiver: torch.Tensor, pos_sender: torch.Tensor, r: float,
                   max_num_neighbors: int = 100) -> torch.Tensor:
    """
    Fallback radius neighbor search between two sets of nodes (when torch_cluster is not available).
    Creates edges from sender to receiver nodes within radius r.

    Args:
        pos_receiver: Receiver node positions of shape (M, D)
        pos_sender: Sender node positions of shape (N, D)
        r: Radius for edge creation
        max_num_neighbors: Maximum number of neighbors to connect

    Returns:
        edge_index: Edge indices of shape (2, E) connecting senders to receivers
    """
    # Compute distances from each receiver to all senders
    diff = pos_receiver.unsqueeze(1) - pos_sender.unsqueeze(0)  # (M, N, D)
    dist = torch.norm(diff, dim=2)  # (M, N)

    # Find edges where distance <= r
    edge_mask = dist <= r

    # Get all valid edges (sender, receiver)
    receivers, senders = torch.where(edge_mask)

    if receivers.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=pos_receiver.device)

    # Limit neighbors per receiver efficiently
    if max_num_neighbors > 0 and receivers.shape[0] > 0:
        # Get distances for valid edges only
        edge_dists = dist[receivers, senders]  # (E,)

        # Sort indices by (receiver, distance) to group by receiver and keep closest neighbors
        receiver_order = receivers.argsort(stable=True)
        receivers_sorted = receivers[receiver_order]
        senders_sorted = senders[receiver_order]
        edge_dists_sorted = edge_dists[receiver_order]

        # Find where receiver changes and limit neighbors per receiver
        receiver_changes = torch.cat([
            torch.tensor([1], dtype=torch.bool, device=pos_receiver.device),
            receivers_sorted[1:] != receivers_sorted[:-1]
        ])

        # Count edges per receiver and create mask for keeping top max_num_neighbors
        cumsum_within_group = torch.arange(receivers_sorted.shape[0], device=pos_receiver.device, dtype=torch.long) - torch.cat([
            torch.tensor([0], dtype=torch.long, device=pos_receiver.device),
            torch.cumsum(receiver_changes[:-1].long(), dim=0)
        ])

        keep_mask = cumsum_within_group < max_num_neighbors

        receivers = receivers_sorted[keep_mask]
        senders = senders_sorted[keep_mask]

    edge_index = torch.stack([senders, receivers], dim=0)
    return edge_index.long()


def build_one_hot_features(num_per_type: list) -> Tensor:
    """
    Builds one-hot feature tensor indicating the edge/node type from numbers per type
    Args:
        num_per_type: List of numbers of nodes per type

    Returns:
        features: One-hot features Tensor
    """
    total_num = sum(num_per_type)
    features = torch.zeros(total_num, len(num_per_type))
    for typ in range(len(num_per_type)):
        features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ+1]), typ] = 1
    return features


def build_type(num_per_type: list) -> Tensor:
    """
    Build node or edge type tensor from list of numbers per type
    Args:
        num_per_type: list of numbers per type

    Returns:
        features: Tensor containing the type as number
    """
    total_num = sum(num_per_type)
    features = torch.zeros(total_num)
    for typ in range(len(num_per_type)):
        features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ+1])] = typ
    return features


def add_static_info(x: Tensor, mesh_positions: Tensor, num_per_node_type: list) -> Tensor:
    """
    Add one-hot encoding for (fixed) static nodes to the 2D data (bottom row)
    Args:
        mesh_positions: Tensor containing the mesh positions
        num_per_type: list of numbers per node type

    Returns:
        x: Updated node features with one-hot encoding
    """
    indices = torch.where(mesh_positions[:,1] == -1.0)
    static = torch.zeros_like(x[:,0]).view(-1,1)
    static[indices[0] + sum(num_per_node_type[0:2])] = 1
    x = torch.cat((x, static), dim=1)
    return x


def add_static_tissue_info(x: Tensor, num_per_node_type: list) -> Tensor:
    """
    Adds one-hot encoding for (fixed) static nodes to the (bottom face of the T part) tissue mesh
    Non-moving nodes ar assigned a zero, one else.
    Args:
        x: Node features
        num_per_type: list of numbers per node type

    Returns:
        x: Updated node features with one-hot encoding
    """
    if num_per_node_type[2] > 400:
        # only for tube task (750 nodes)  todo/quick and dirty solution
        indices = torch.linspace(0, 29, 30, dtype=int)
    else:
        indices = torch.tensor([59,  60,  61,  62,  84,  87, 105, 106, 169, 170, 171, 172, 194, 196,
            215, 216, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
            246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
            260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 273, 274, 275,
            276, 277, 280, 281, 310, 311, 312, 313, 314, 317, 318, 321, 322, 323,
            324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
            338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
            352, 353, 354, 355, 356, 357, 358, 359, 360])
    static = torch.ones_like(x[:,0]).view(-1,1)
    static[indices + sum(num_per_node_type[0:2])] = 0
    x = torch.cat((x, static), dim=1)
    return x


def get_relative_mesh_positions(mesh_edge_index: Tensor, mesh_positions: Tensor) -> Tensor:
    """
    Transform the positions of the mesh into a relative position encoding along with the Euclidean distance in the edges
    Args:
        mesh_edge_index: Tensor containing the mesh edge indices
        mesh_positions: Tensor containing mesh positions

    Returns:
        edge_attr: Tensor containing the batched edge features
    """
    data = Data(pos=mesh_positions,
                edge_index=mesh_edge_index)
    transforms = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
    data = transforms(data)
    return data.edge_attr


def add_relative_mesh_positions(edge_attr: Tensor, edge_type: Tensor, input_mesh_edge_index: Tensor, initial_mesh_positions: Tensor) -> Tensor:
    """
    Adds the relative mesh positions to the mesh edges (in contrast to the world edges) and zero anywhere else.
    Refer to MGN by Pfaff et al. 2020 for more details.
    Args:
        edge_attr: Current edge features
        edge_type: Tensor containing the edges types
        input_mesh_edge_index: Mesh edge index tensor
        initial_mesh_positions: Initial positions of the mesh nodes "mesh coordinates"

    Returns:
        edge_attr: updated edge features
    """
    indices = torch.where(edge_type == 2)[0]  # type 2: mesh edges
    mesh_edge_index = input_mesh_edge_index
    mesh_attr = get_relative_mesh_positions(mesh_edge_index, initial_mesh_positions)
    mesh_positions = torch.zeros(edge_attr.shape[0], mesh_attr.shape[1])
    mesh_positions[indices,:] = mesh_attr
    edge_attr = torch.cat((edge_attr, mesh_positions), dim=1)
    return edge_attr


def remove_duplicates_with_mesh_edges(mesh_edges: Tensor, world_edges: Tensor) -> Tensor:
    """
    Removes the duplicates with the mesh edges have of the world edges that are created using a nearset neighbor search. (only MGN)
    To speed this up the adjacency matrices are used
    Args:
        mesh_edges: edge list of the mesh edges
        world_edges: edge list of the world edges

    Returns:
        new_world_edges: updated world edges without duplicates
    """
    import torch_geometric.utils as utils
    adj_mesh = utils.to_dense_adj(mesh_edges)
    if world_edges.shape[1] > 0:
        adj_world = utils.to_dense_adj(world_edges)
    else:
        adj_world = torch.zeros_like(adj_mesh)
    if adj_world.shape[1] < adj_mesh.shape[1]:
        padding_size = adj_mesh.shape[1] - adj_world.shape[1]
        padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
        adj_world = padding_mask(adj_world)
    elif adj_world.shape[1] > adj_mesh.shape[1]:
        padding_size = adj_world.shape[1] - adj_mesh.shape[1]
        padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
        adj_mesh = padding_mask(adj_mesh)
    new_adj = adj_world-adj_mesh
    new_adj[new_adj < 0] = 0
    new_world_edges = utils.dense_to_sparse(new_adj)[0]
    return new_world_edges


def create_graph_from_raw(input_data,
                          edge_radius_dict: Dict,
                          output_device,
                          use_mesh_coordinates: bool = False,
                          hetero: bool = False,
                          tissue_task: bool = False,
                          predict_collider=False):
    """
    Choose correct graph creation function for homogeneous or heterogeneous data
    """
    if hetero:
        raise NotImplementedError("do not use this")
        return create_hetero_graph_from_raw(input_data,
                                            edge_radius_dict,
                                            output_device,
                                            use_mesh_coordinates,
                                            tissue_task)
    else:
        return create_homo_graph_from_raw(input_data,
                                          edge_radius_dict,
                                          output_device,
                                          use_mesh_coordinates,
                                          tissue_task,
                                          predict_collider=predict_collider)


def create_homo_graph_from_raw(input_data,
                               edge_radius_dict: Dict,
                               output_device,
                               use_mesh_coordinates: bool = False,
                               tissue_task: bool = False,
                               predict_collider=False) -> Data: # TODO: set accordingly from the dataset
    """
    Creates a homogeneous graph from the raw data (point cloud, collider, mesh) given the connectivity of the edge radius dict
    Args:
        input_data: Tuple containing the data for the time step
        edge_radius_dict: Edge radius dict describing the connectivity setting
        output_device: Working device for output data, either cpu or cuda
        use_mesh_coordinates: enables message passing also in mesh coordinate space
        tissue_task: True if 3D data is used

    Returns:
        data: Data element containing the built graph
    """
    if tissue_task:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio, forces, combined_vel = input_data
    else:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio, forces, combined_vel = input_data

    # dictionary for positions
    pos_dict = {'grid': grid_positions,
                'collider': collider_positions,
                'mesh': mesh_positions}

    # build nodes features (one hot)
    num_nodes = []
    for values in pos_dict.values():
        num_nodes.append(values.shape[0])
    if predict_collider:
        original_num_nodes = deepcopy(num_nodes)
        # set the num_nodes of collider to 0
        num_nodes[2] += num_nodes[1]
        num_nodes[1] = 0
        additional_one_hot = build_one_hot_features(original_num_nodes)
        # remove grid column from additional one hot
        additional_one_hot = additional_one_hot[:, 1:]

    x = build_one_hot_features(num_nodes)
    x_description = ["grid", "collider", "mesh"]
    if predict_collider:
        # add a distinction between collider and mesh
        x = torch.cat((x, additional_one_hot), dim=1)
        x_description += ["collider_identifier", "mesh_identifier"]

    # add colors and collider velocity to point cloud (only 2D)
    if grid_colors is not None:
        x_colors = torch.zeros_like(x)
        x_colors[0:num_nodes[0], :] = grid_colors
        x_colors[num_nodes[0]:num_nodes[0] + num_nodes[1], 2] = torch.ones(num_nodes[1])*(-200.0/100*0.01)
        x = torch.cat((x, x_colors), dim=1)
        x_description += ["x_colors"] * x_colors.shape[1]

    # check for forces
    if forces is not None:
        force_features = []
        for force_dict in forces.values():
            force_nodes = SheetDeformationDataset.get_node_indices_with_force_influence(initial_mesh_positions.cpu() * 140, force_dict["position"].cpu())
            force_feature = torch.zeros((initial_mesh_positions.shape[0], 1))
            force_feature[force_nodes] = force_dict["direction"][2] / 200  # force normalization factor hardcoded
            force_features.append(force_feature)
            x_description.append("force")
        force_features = torch.cat(force_features, dim=-1)
        # force feature for point cloud is always zero
        force_features_pcd = torch.zeros((num_nodes[0], force_features.shape[1]))
        force_features = torch.cat((force_features_pcd, force_features), dim=0)
        x = torch.cat((x, force_features), dim=1)
    node_type = build_type(num_nodes)


    # # used if poisson ratio needed as input feature, but atm incompatible with Imputation training
    if poisson_ratio is not None:
        poisson_ratio = poisson_ratio.float()
        x_poisson = torch.ones_like(x[:, 0])
        x_poisson = x_poisson * poisson_ratio
        x = torch.cat((x, x_poisson.unsqueeze(1)), dim=1)
    else:
        poisson_ratio = torch.tensor([1.0])

    # add velocity
    x = torch.cat((x, combined_vel), dim=1)
    x_description += ["vel"]*combined_vel.shape[1]

    # index shift dict for edge index matrix
    index_shift_dict = {'grid': 0,
                        'collider': num_nodes[0],
                        'mesh': num_nodes[0] + num_nodes[1]}

    if predict_collider:
        # rebuild pos dict so that there is no collider node
        pos_dict = {'grid': grid_positions,
                    "collider": torch.empty(0, collider_positions.shape[1]),
                    'mesh': torch.cat((collider_positions, mesh_positions), dim=0)}


    # create edge_index dict with the same keys as edge_radius
    edge_index_dict = {}
    for key in edge_radius_dict.keys():
        if key[0] == key[2]:

            # use mesh connectivity instead of nearest neighbor
            if key[0] == 'mesh':
                # mesh indices are already connected in both ways, so no need to add reversed edges
                edge_index_dict[key] = torch.clone(input_mesh_edge_index)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            # elif key[0] == 'collider':
            #     if tube_task:
            #         edge_index_dict[key] = torch.cat((collider_edge_index, collider_edge_index[[1, 0]]), dim=1)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #     else:
            #         edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            # use radius graph for edges between nodes of the same type
            else:
                if HAS_TORCH_CLUSTER:
                    edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
                else:
                    edge_index_dict[key] = fallback_radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # use radius for edges between different sender and receiver nodes
        else:
            if HAS_TORCH_CLUSTER:
                edge_index_dict[key] = torch_cluster.radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            else:
                edge_index_dict[key] = fallback_radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            edge_index_dict[key][1, :] += index_shift_dict[key[2]]

    # add world edges if edge radius for mesh is not none
    mesh_key = ('mesh', '2', 'mesh')
    world_key = ('mesh', '9', 'mesh')
    if edge_radius_dict[mesh_key] is not None:
        if HAS_TORCH_CLUSTER:
            edge_index_dict[world_key] = torch_cluster.radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key], max_num_neighbors=100)
        else:
            edge_index_dict[world_key] = fallback_radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key], max_num_neighbors=100)
        edge_index_dict[world_key][0, :] += index_shift_dict['mesh']
        edge_index_dict[world_key][1, :] += index_shift_dict['mesh']
        edge_index_dict[world_key] = remove_duplicates_with_mesh_edges(edge_index_dict[mesh_key], edge_index_dict[world_key])

    # build edge_attr (one-hot)
    num_edges = []
    for value in edge_index_dict.values():
        num_edges.append(value.shape[1])
    edge_attr = build_one_hot_features(num_edges)
    edge_type = build_type(num_edges)

    # add mesh_coordinates to mesh edges if used
    if use_mesh_coordinates:
        edge_attr = add_relative_mesh_positions(edge_attr, edge_type, input_mesh_edge_index, initial_mesh_positions)

    # create node positions tensor and edge_index from dicts
    pos = torch.cat(tuple(pos_dict.values()), dim=0)
    edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)

    # create data object for torch
    data = Data(x=x.float(),
                x_description=x_description,
                u=poisson_ratio,
                pos=pos.float(),
                edge_index=edge_index.long(),
                edge_attr=edge_attr.float(),
                y=label.float(),
                y_old=pos[node_type == 2].float(),
                node_type=node_type,
                edge_type=edge_type,
                poisson_ratio=poisson_ratio).to(output_device)
    if forces is not None:
        # need them for full rollout
        data.forces = forces
    return data


def create_hetero_graph_from_raw(input_data,
                                 edge_radius_dict: Dict,
                                 output_device,
                                 use_mesh_coordinates: bool = False,
                                 tissue_task: bool = False) -> Data:
    """
    Creates a homogeneous graph from the raw data (point cloud, collider, mesh) given the connectivity of the edge radius dict which is then later converted to a heterogeneous graph by .to_heterogeneous
    Args:
        input_data: Tuple containing the data for the time step
        edge_radius_dict: Edge radius dict describing the connectivity setting
        output_device: Working device for output data, either cpu or cud
        use_mesh_coordinates: enables message passing also in mesh coordinate space (currently not supported for hetero)
        tissue_task: True if 3D data is used

    Returns:
        data: Data element containing the built graph
    """
    if tissue_task:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio = input_data
    else:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio = input_data

    # create node position dict
    pos_dict = {'grid': grid_positions,
                'collider': collider_positions,
                'mesh': mesh_positions}

    # build nodes features (one hot)
    num_nodes = []
    for values in pos_dict.values():
        num_nodes.append(values.shape[0])
    node_type = build_type(num_nodes)

    # index shift dict for edge index matrix
    index_shift_dict = {'grid': 0,
                        'collider': num_nodes[0],
                        'mesh': num_nodes[0] + num_nodes[1]}

    # create edge_index dict with the same keys as edge_radius
    edge_index_dict = {}
    for key in edge_radius_dict.keys():
        if key[0] == key[2]:

            # use mesh connectivity instead of nearest neighbor
            if key[0] == 'mesh':
                edge_index_dict[key] = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #
            # elif key[0] == 'collider':
            #     if tube_task:
            #         edge_index_dict[key] = torch.cat((collider_edge_index, collider_edge_index[[1, 0]]), dim=1)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #     else:
            #         edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]

            # use radius graph for edges between nodes of the same type
            else:
                if HAS_TORCH_CLUSTER:
                    edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
                else:
                    edge_index_dict[key] = fallback_radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # use radius for edges between different sender and receiver nodes
        else:
            if HAS_TORCH_CLUSTER:
                edge_index_dict[key] = torch_cluster.radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            else:
                edge_index_dict[key] = fallback_radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # add world edges if edge radius for mesh is not none
    mesh_key = ('mesh', '2', 'mesh')
    world_key = ('mesh', '9', 'mesh')
    if edge_radius_dict[mesh_key] is not None:
        if HAS_TORCH_CLUSTER:
            edge_index_dict[world_key] = torch_cluster.radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key], max_num_neighbors=100)
        else:
            edge_index_dict[world_key] = fallback_radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key], max_num_neighbors=100)
        edge_index_dict[world_key][0, :] += index_shift_dict['mesh']
        edge_index_dict[world_key][1, :] += index_shift_dict['mesh']
        edge_index_dict[world_key] = remove_duplicates_with_mesh_edges(edge_index_dict[mesh_key], edge_index_dict[world_key])

    # build edge_attr (one-hot)
    num_edges = []
    for value in edge_index_dict.values():
        num_edges.append(value.shape[1])
    edge_type = build_type(num_edges)

    # create node pos tensor and edge_index from dicts
    pos = torch.cat(tuple(pos_dict.values()), dim=0)
    edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)

    if poisson_ratio is not None:
        x_poisson = torch.ones(pos.shape[0], 1)
        x_poisson = x_poisson * poisson_ratio
        x = x_poisson
    else:
        x = torch.ones(pos.shape[0], 1)
        poisson_ratio = torch.tensor([-1.0])

    # create data object for torch
    data = Data(x=x.float(),
                pos=pos.float(),
                edge_index=edge_index.long(),
                y=label.float(),
                y_old=mesh_positions.float(),
                node_type=node_type.long(),
                edge_type=edge_type.long(),
                poisson_ratio=poisson_ratio.view(1, -1),
                mesh_edges=torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1).long().view(1, 2, -1))

    # add features to data which are later used after converting to a heterogeneous graph
    if grid_colors is not None:
        data.grid_colors = grid_colors.float()
    if tissue_task:
        collider_velocities = (next_collider_positions - collider_positions).squeeze()
        data.collider_velocities = torch.nn.functional.normalize(collider_velocities, dim=0).float()
        data.initial_mesh_positions = initial_mesh_positions.float().view(1, -1, 3)
    else:
        data.initial_mesh_positions = initial_mesh_positions.float().view(1, -1, 2)
    data.to(output_device)
    return data
