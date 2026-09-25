import torch

from pc_mango.dataset.util.util import add_noise

def step_length(config, tasks, task_size, start_time_steps):
    true_length = len(tasks) * task_size * start_time_steps
    if config.get("dataset_size", None) is None:
        return true_length
    else:
        return min(config.dataset_size, true_length)

def step_get(idx, config, super_get, start_time_steps, task_size, ):
    """

    :param idx: step index
    :param config: dataset config
    :param super_get: super method to get the batched ml data
    :param start_time_steps: number of indices to start from
    :param task_size: number of trajs in task
    :return:
    """
    all_traj_idx, step_idx = idx // start_time_steps, idx % start_time_steps
    task_idx, traj_idx = all_traj_idx // task_size, all_traj_idx % task_size
    ml_result = super_get(task_idx)

    x = ml_result["x"][traj_idx, step_idx, :, :]
    if len(ml_result["h"].shape) == 4:
        h = ml_result["h"][traj_idx, step_idx, :, :]
    else:
        h = ml_result["h"][traj_idx, :, :]
    y = ml_result["x"][traj_idx, step_idx + 1, :, :]
    h_description = ml_result["h_description"]
    edge_indices = ml_result["edge_indices"]
    edge_features = ml_result["edge_features"][traj_idx, :, :]
    edge_feature_description = ml_result["edge_feature_description"]
    regression_features = ml_result["regression_features"]
    meta_data = ml_result.get("meta_data", None)

    # add noise
    deformable_mask = h[:, 0] == 1
    x, _ = add_noise(config.get("noise_scale", 0.0), x, history_pos=torch.zeros((0, x.shape[0], x.shape[1])),
                     deformable_mask=deformable_mask)

    # recompute velocity since we added noise to the position.
    if step_idx == 0:
        if config.initial_vel == "zero":
            v = torch.zeros_like(x)
        elif config.initial_vel == "constant":
            v = y - x
        else:
            raise ValueError(f"Unknown initial velocity type: {config.initial_vel}")
    else:
        prev_pos = ml_result["x"][traj_idx, step_idx - 1, :, :]
        v = x - prev_pos

    result = {
        "x": x,  # shape (num_nodes, world_dim)
        "v": v,  # shape (num_nodes, world_dim)
        "h": h,  # shape (num_nodes, node_feature_dim)
        "h_description": h_description,
        "edge_indices": edge_indices,  # shape (2, num_edges)
        "edge_features": edge_features,  # shape (num_edges, num_edge_features)
        "edge_feature_description": edge_feature_description,
        "regression_features": regression_features,  # shape (num_regression_features,)
        "y": y,  # shape (num_nodes, world_dim)
        "context_trajs": [0],  # dummy context trajs needed for parsing
        "target_trajs": [0],  # dummy target trajs needed for parsing
    }
    if meta_data is not None:
        result["meta_data"] = meta_data
    return result