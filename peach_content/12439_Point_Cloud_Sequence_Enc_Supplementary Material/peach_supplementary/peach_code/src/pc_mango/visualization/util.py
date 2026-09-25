import os

from pc_mango.visualization.save_trajectory import export_deforming_mesh_to_xdmf, export_two_meshes_to_xdmf_merged


def save_visualizations(vis_dict, out_path):
    os.makedirs(out_path, exist_ok=True)
    if "collider" in vis_dict and vis_dict["collider"] is not None:
        # dual mesh case. but collider is not predicted
        export_two_meshes_to_xdmf_merged(
            positions_A=vis_dict["predicted_trajectory"],
            connectivity_A=vis_dict["mesh_connectivity"],
            positions_B=vis_dict["collider"],
            connectivity_B=vis_dict["collider_connectivity"],
            cell_type="triangle",
            out_path=os.path.join(out_path, "predicted_trajectory.xdmf"),
        )
        export_two_meshes_to_xdmf_merged(
            positions_A=vis_dict["ground_truth_trajectory"],
            connectivity_A=vis_dict["mesh_connectivity"],
            positions_B=vis_dict["collider"],
            connectivity_B=vis_dict["collider_connectivity"],
            cell_type="triangle",
            out_path=os.path.join(out_path, "ground_truth_trajectory.xdmf"),
        )
    elif "predicted_collider" in vis_dict and vis_dict["predicted_collider"] is not None:
        # dual mesh case, but collider is predicted
        export_two_meshes_to_xdmf_merged(
            positions_A=vis_dict["predicted_trajectory"],
            connectivity_A=vis_dict["mesh_connectivity"],
            positions_B=vis_dict["predicted_collider"],
            connectivity_B=vis_dict["collider_connectivity"],
            cell_type="triangle",
            out_path=os.path.join(out_path, "predicted_trajectory.xdmf"),
        )
        export_two_meshes_to_xdmf_merged(
            positions_A=vis_dict["ground_truth_trajectory"],
            connectivity_A=vis_dict["mesh_connectivity"],
            positions_B=vis_dict["ground_truth_collider"],
            connectivity_B=vis_dict["collider_connectivity"],
            cell_type="triangle",
            out_path=os.path.join(out_path, "ground_truth_trajectory.xdmf"),
        )
    else:
        export_deforming_mesh_to_xdmf(
            positions=vis_dict["predicted_trajectory"],
            connectivity=vis_dict["mesh_connectivity"],
            cell_type="triangle",
            out_path=os.path.join(out_path, "predicted_trajectory.xdmf"),
            features=None,
        )

        export_deforming_mesh_to_xdmf(
            positions=vis_dict["ground_truth_trajectory"],
            connectivity=vis_dict["mesh_connectivity"],
            cell_type="triangle",
            out_path=os.path.join(out_path, "ground_truth_trajectory.xdmf"),
            features=None,
        )
    return True
    # TODO: also log pointclouds if needed