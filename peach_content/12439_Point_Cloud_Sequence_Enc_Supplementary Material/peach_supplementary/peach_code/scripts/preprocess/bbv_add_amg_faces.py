import h5py
import torch
from torch_geometric.data import HeteroData, Batch
from tqdm import tqdm

from pc_mango.dataset.edges.edge_indices import get_edge_indices
from pc_mango.dataset.util.amg_generation import AMGBuildHierarchicalGraph


def bbv_add_amg():
    amg_builder = AMGBuildHierarchicalGraph()

    current_dataset_file = "../datasets/pc_mango/bbv_v3.hdf5"
    amg_output_file = "../datasets/pc_mango/bbv_v3_amg.hdf5"

    output = {}

    with h5py.File(current_dataset_file, "r") as f:
        # Process tasks
        task_names = list(f.keys())
        for task_idx, task_name in tqdm(enumerate(task_names), desc="Processing Tasks", total=len(task_names)):
            print("starting task_idx", task_idx, "task_name", task_name)
            if not task_name.startswith("task_"):
                continue

            task = f[task_name]
            output[task_name] = {}
            faces = task["trajs"]["traj_000"]["faces"][:]
            edge_indices = get_edge_indices(faces)
            pos = torch.tensor(task["trajs"]["traj_000"]["mesh_pos"][:][0])  # shape (num_ts, num_nodes, 3) -> take first ts for edge index

            # if task_idx == 6:
            #     import matplotlib.pyplot as plt
            #     import matplotlib.tri as mtri
            #
            #     # if tensors, convert
            #     pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else pos
            #     pos_np = pos_np[:, :2]  # take only x and y for plotting
            #     faces_np = faces.numpy() if isinstance(faces, torch.Tensor) else faces
            #
            #     tri = mtri.Triangulation(pos_np[:, 0], pos_np[:, 1], faces_np)
            #
            #     plt.figure(figsize=(8, 8))
            #     plt.triplot(tri, 'b-', linewidth=0.5)
            #     plt.scatter(pos_np[:, 0], pos_np[:, 1], c='red', s=10, zorder=5)
            #     plt.axis('equal')
            #     plt.title(f"Mesh: {pos_np.shape[0]} nodes, {faces_np.shape[0]} faces")
            #     plt.show()

            data = HeteroData()
            data["mesh"].const = {"pos": pos}
            data["mesh"].num_nodes = pos.shape[0]
            data["mesh", "mesh", "mesh"].edge_index = edge_indices

            batch = Batch.from_data_list([data])
            amg_batch = amg_builder(batch)
            mesh_hir_mapping = amg_batch["mesh", "map", "hierarchical"].edge_index[0]  # original -> hir
            edge_index_h = amg_batch["hierarchical", "mesh", "hierarchical"].edge_index
            node_attr_level = amg_batch["hierarchical"].const["node_attr_level"]

            # invert the mapping: hir_node_idx -> original_node_idx
            hir_to_orig = torch.full((mesh_hir_mapping.shape[0],), -1, dtype=torch.long)
            hir_to_orig[amg_batch["mesh", "map", "hierarchical"].edge_index[1]] = mesh_hir_mapping

            edges_per_level = {}
            num_levels = node_attr_level.max().item() + 1

            #ignore original level (0), start from level 1
            for lvl in range(1, num_levels):
                # nodes belonging to this level (hierarchical indices)
                lvl_nodes = (node_attr_level == lvl).nonzero(as_tuple=True)[0]
                lvl_node_set = set(lvl_nodes.tolist())

                # edges where both endpoints are at this level
                src, dst = edge_index_h
                mask = torch.tensor([s.item() in lvl_node_set and d.item() in lvl_node_set
                                     for s, d in zip(src, dst)], dtype=torch.bool)

                edge_index_lvl_hir = edge_index_h[:, mask]

                # remap to original node indices
                edge_index_lvl_orig = hir_to_orig[edge_index_lvl_hir]

                edges_per_level[lvl] = edge_index_lvl_orig

            # save in output
            output[task_name]["trajs"] = {}
            for traj_name in task["trajs"].keys():
                output[task_name]["trajs"][traj_name] = {}
                # add edge index as additional feature dimension
                for lvl in range(1, num_levels):
                    edge_index_lvl_orig = edges_per_level[lvl]
                    # mark nodes that are part of an edge at this level
                    output[task_name]["trajs"][traj_name][f"amg_lvl_{lvl}"] = edge_index_lvl_orig.numpy()



        # save output to new hdf5 file
        with h5py.File(amg_output_file, "w") as out:
            for task_name, trajs in output.items():
                task_group = out.create_group(task_name)
                task_group = task_group.create_group("trajs")
                for traj_name, amgs in trajs["trajs"].items():
                    traj_group = task_group.create_group(traj_name)
                    for key, value in amgs.items():
                        # save pc and pc_node_type in separate datasets
                        traj_group.create_dataset(key, data=value)
        print("Saved amg data to", amg_output_file)


if __name__ == "__main__":
    bbv_add_amg()