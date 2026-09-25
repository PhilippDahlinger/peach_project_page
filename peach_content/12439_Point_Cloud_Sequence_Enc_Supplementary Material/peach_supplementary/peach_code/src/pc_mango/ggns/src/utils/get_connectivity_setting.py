from pc_mango.ggns.util.Types import *


def get_connectivity_setting(dataset: str) -> Tuple:
    """
    Outputs the corresponding properties: edge radii, input time step and euclidean_distance for the used dataset.
    Args:
        dataset: Name that specifies the connectivity setting
    Returns: Tuple.
        edge_radius_dict: Resulting edge radius dict for dataset
        input_timestep: Use point cloud of time step 't' or 't+1'
        euclidian_distance: Is a Euclidean distance as edges feature used
        tissue_task: Is this a 3D dataset (deprecated — use config.is_3d instead)
    """
    normalization_corrector =  2.5

    # 2D Deformable Block connectivity setting (homogeneous mode)
    if dataset == "deformable_block":
        edge_radius = [0.1 / normalization_corrector, 0.08 / normalization_corrector, None,
                       0.08 / normalization_corrector, 0.08 / normalization_corrector, 0.3 / normalization_corrector,
                       0.08 / normalization_corrector, 0.08 / normalization_corrector, 0.3 / normalization_corrector]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Sheet Deformation connectivity setting (no collider)
    elif dataset == "sheet_deformation":
        edge_radius = [
            0.18 / normalization_corrector,   # 0: grid-grid
            0.0,                              # 1: collider-collider (no collider)
            None,                             # 2: mesh-mesh (use mesh connectivity)
            0.0,                              # 3: collider-grid (no collider)
            0.1 / normalization_corrector,   # 4: mesh-grid
            0.0,                              # 5: mesh-collider (no collider)
            0.0,                              # 6: grid-collider (no collider)
            0.1 / normalization_corrector,   # 7: grid-mesh
            0.0,                              # 8: collider-mesh (no collider)
        ]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Trampoline connectivity setting (sheet mesh + sphere collider)
    elif dataset == "trampoline":
        edge_radius = [
            0.12 / normalization_corrector,    # 0: grid-grid
            0.065 / normalization_corrector,   # 1: collider-collider (sphere self-edges)
            None,                              # 2: mesh-mesh (use sheet connectivity)
            0.05 / normalization_corrector,   # 3: collider-grid
            0.08 / normalization_corrector,   # 4: mesh-grid
            0.15 / normalization_corrector,    # 5: mesh-collider
            0.05 / normalization_corrector,   # 6: grid-collider
            0.08 / normalization_corrector,   # 7: grid-mesh
            0.15 / normalization_corrector,    # 8: collider-mesh
        ]
        input_timestep = "t"
        euclidian_distance = True

    else:
        raise ValueError(f"Dataset {dataset} does currently not exist. Consider adding it to get_connectivity_setting")

    edge_radius_dict = get_radius_dict(edge_radius)
    tissue_task = False

    return edge_radius_dict, input_timestep, euclidian_distance, tissue_task


def get_radius_dict(edge_radius: list) -> Dict:
    """
    Build an edge radius dict from a list of edge radii
    Args:
        edge_radius: List of the used edge radii
    Returns:
        edge_radius_dict: Dict containing the edge radii with their names
    """
    edge_radius_keys = [('grid', '0', 'grid'),
                        ('collider', '1', 'collider'),
                        ('mesh', '2', 'mesh'),
                        ('collider', '3', 'grid'),
                        ('mesh', '4', 'grid'),
                        ('mesh', '5', 'collider'),
                        ('grid', '6', 'collider'),
                        ('grid', '7', 'mesh'),
                        ('collider', '8', 'mesh')]

    edge_radius_dict = dict(zip(edge_radius_keys, edge_radius))
    return edge_radius_dict
