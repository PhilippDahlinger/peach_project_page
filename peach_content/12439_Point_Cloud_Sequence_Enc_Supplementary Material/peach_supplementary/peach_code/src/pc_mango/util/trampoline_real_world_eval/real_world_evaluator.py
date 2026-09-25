import copy
import os
import re

import fpsample
import h5py
import torch
from matplotlib import pyplot as plt
from omegaconf import open_dict
from tqdm import tqdm

from pc_mango.algorithm import get_algorithm
from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
from pc_mango.dataset.util.util import (
    hdf5_group_to_dict,
    get_collider_connectivity,
    get_mesh_connectivity,
)
from pc_mango.util.initialization import get_data
from pc_mango.util.trampoline_real_world_eval.combined_visualization import (
    visualize_prediction_interactive,
)
from pc_mango.util.trampoline_real_world_eval.pc_processing import (
    classify_pink_hsv,
    estimate_pink_reference,
    iterative_pink_segmentation_knn,
    crop_robot_out_and_subsample,
)
from pc_mango.util.trampoline_real_world_eval.point_to_mesh_loss import (
    compute_point_to_mesh_loss,
    compute_point_to_mesh_loss2
)
from pc_mango.visualization.debug_pc_plot import visualize_pointcloud_sequence
from pc_mango.visualization.save_pointcloud_trampoline_v4 import write_pointcloud_timeseries_pvd
from pc_mango.visualization.save_trajectory import export_two_meshes_to_xdmf_merged


class RealWorldEvaluator:
    diameter_dict = {
        1: 15,
        2: 30,
        3: 40,
        4: 50,
        5: 30,
        6: 40,
        7: 50,
        8: 60,
        9: 15,
        10: 30,
        11: 40,
        12: 15,
        13: 30,
        14: 40,
    }

    def __init__(self, config):
        self.config = config
        # load meshes
        train_ds, eval_ds, train_dl, eval_dl = get_data(self.config.dataset)
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.eval_ds = eval_ds
        self.sim_dataset: TrampolineDataset = eval_ds
        self.faces = self.sim_dataset.get_faces()
        self.initial_sheet_pos = self.sim_dataset.tasks[0]["trajs"]["traj_001"][
            "sheet_pos"
        ][0]
        # acquire all sphere positions for all radii
        required_diameters = [30.0, 40.0, 50.0, 60.0]
        self.sphere_pos = {}

        with h5py.File(self.config.dataset.root, "r") as hdf:
            for task_key in sorted(hdf.keys()):
                if task_key.startswith("task_"):
                    task_group = hdf5_group_to_dict(hdf[task_key])
                    ball_diameter = task_group["params"]["ball_diameter"].item()
                    if ball_diameter in required_diameters:
                        sphere_pos = task_group["trajs"]["traj_001"][
                            "sphere_pos"
                        ]  # shape (T, 3)
                        self.sphere_pos[str(ball_diameter)] = sphere_pos[0]
                        required_diameters.remove(ball_diameter)
                        if len(required_diameters) == 0:
                            break
        # print("stop")

    def load_model_checkpoint(self, model_checkpoint):
        algorithm = get_algorithm(
            config=self.config.algorithm,
            train_dl=self.train_dl,
            train_ds=self.train_ds,
            eval_ds=self.eval_ds,
            loading=True,
            checkpoint_path=model_checkpoint,
        )

        if "test_opt_oracle" in self.config.algorithm:
            with open_dict(algorithm.config):
                algorithm.config.test_opt_oracle = self.config.algorithm.test_opt_oracle

        self.encoder = algorithm.encoder
        self.simulator = algorithm.simulator
        self.meta_aggregation = algorithm.meta_aggregation
        self.mat_prop_head = algorithm.mat_prop_head

    def evaluate(self):
        if not hasattr(self, "_active_model_checkpoint"):
            if len(self.config.model_checkpoints) == 0:
                raise ValueError("model_checkpoints must contain at least one checkpoint")
            for model_checkpoint in self.config.model_checkpoints:
                print(f"Evaluating checkpoint {model_checkpoint}")
                self._active_model_checkpoint = model_checkpoint
                self.load_model_checkpoint(model_checkpoint)
                self.evaluate()
            del self._active_model_checkpoint
            return

        pc_root = self.config.pointcloud_root
        pc_root_dir = pc_root.split("/")[-1]
        cache_dir = os.path.join(self.config.cache_path, f"{pc_root_dir}_cache")
        os.makedirs(cache_dir, exist_ok=True)
        seed = self.get_checkpoint_seed(self._active_model_checkpoint)
        results_root = os.path.join(self.config.results_output_path, f"seed_{seed}")
        vis_root = os.path.join(self.config.vis_output_path, f"seed_{seed}")
        os.makedirs(results_root, exist_ok=True)
        xdmf_per_condition = int(self.config.xdmf_per_condition)
        xdmf_counts = {}
        if self.config.latex_ids is None or self.config.latex_ids == []:
            latex_ids = None
        else:
            latex_ids = set(self.config.latex_ids)
        if self.config.sphere_ids is None or self.config.sphere_ids == []:
            sphere_ids = None
        else:
            sphere_ids = set(self.config.sphere_ids)
        if self.config.traj_filter is None or self.config.traj_filter == []:
            trajectory_filter = None
        else:
            trajectory_filter = set(self.config.traj_filter)

        for subdir in sorted(os.listdir(pc_root)):
            subdir_path = os.path.join(pc_root, subdir)
            if not os.path.isdir(subdir_path):
                continue

            match = re.match(r"latex-(\d+)_sphere-(\d+)", subdir)
            if match:
                latex = int(match.group(1))
                sphere_id = int(match.group(2))
                if latex_ids is not None and latex not in latex_ids:
                    continue
                if sphere_ids is not None and sphere_id not in sphere_ids:
                    continue

                # check cache
                cache_file = os.path.join(
                    cache_dir, f"latex-{latex}_sphere-{sphere_id}.pt"
                )
                if os.path.exists(cache_file):
                    # load from cache
                    print(
                        f"Loading cached results for latex {latex} and sphere {sphere_id}"
                    )
                    cache_dict = torch.load(cache_file)
                    stacked_pc_data = cache_dict["stacked_pc_data"]
                    stacked_pc_type = cache_dict["stacked_pc_type"]
                    stacked_sheet_pos = cache_dict["stacked_sheet_pos"]
                    stacked_sphere_pos = cache_dict["stacked_sphere_pos"]
                    stacked_time_stamps = cache_dict["stacked_time_stamps"]
                    diameter = float(cache_dict["diameter"])
                else:
                    # compute and save to cache
                    stacked_pc_data = []
                    stacked_pc_type = []
                    stacked_sheet_pos = []
                    stacked_sphere_pos = []
                    stacked_time_stamps = []
                    diameter = None
                    for traj in tqdm(os.listdir(subdir_path), "Loading Trajs"):
                        if (
                            trajectory_filter is not None
                            and traj not in trajectory_filter
                        ):
                            continue
                        traj_path = os.path.join(subdir_path, traj)
                        pc_data, pc_type, center_pos, time_stamps = (
                            self.load_pointcloud(traj_path, sphere_id, sheet_id=latex)
                        )
                        diameter = self.diameter_dict[sphere_id]
                        sheet_pos, sheet_faces, sphere_pos, sphere_faces = (
                            self.build_initial_mesh(center_pos[0], diameter)
                        )
                        stacked_pc_data.append(pc_data)
                        stacked_pc_type.append(pc_type)
                        stacked_sheet_pos.append(sheet_pos[None, :, :])
                        stacked_sphere_pos.append(sphere_pos[None, :, :])
                        stacked_time_stamps.append(
                            torch.tensor(time_stamps, dtype=torch.float32)
                        )
                    stacked_pc_data = torch.stack(stacked_pc_data, dim=0)
                    stacked_pc_type = torch.stack(stacked_pc_type, dim=0)
                    stacked_sheet_pos = torch.stack(stacked_sheet_pos)
                    stacked_sphere_pos = torch.stack(stacked_sphere_pos)
                    stacked_time_stamps = torch.stack(stacked_time_stamps)
                    # save to cache
                    cache_dict = {
                        "stacked_pc_data": stacked_pc_data,
                        "stacked_pc_type": stacked_pc_type,
                        "stacked_sheet_pos": stacked_sheet_pos,
                        "stacked_sphere_pos": stacked_sphere_pos,
                        "stacked_time_stamps": stacked_time_stamps,
                        "diameter": diameter,
                    }
                    # save it
                    with open(cache_file, "wb") as f:
                        torch.save(cache_dict, f)

                predict = True
                if not predict:
                    continue

                for target_traj_idx in range(0, 10):
                    context_trajs = list(range(0, 10))
                    context_trajs.remove(target_traj_idx)
                    context_trajs = torch.tensor(context_trajs)
                    # select 8 random context trajs if there are more than 8
                    if len(context_trajs) > 8:
                        context_trajs = context_trajs[
                            torch.randperm(len(context_trajs))[:8]
                        ]
                    # remove target traj from context traj
                    # context_trajs = context_trajs[context_trajs != target_traj_idx]
                    target_trajs = torch.tensor([target_traj_idx])
                    print(
                        "Evaluating latex {}, sphere {}, target traj {}".format(
                            latex, sphere_id, target_traj_idx
                        )
                    )
                    print(context_trajs)
                    oracle_flag = self.config.algorithm.get("oracle", False)
                    batch = self.sim_dataset.get_batch_from_real_data(
                        stacked_pc_data,
                        stacked_pc_type,
                        stacked_sheet_pos,
                        stacked_sphere_pos,
                        diameter,
                        # sheet_faces,
                        # sphere_faces,
                        context_trajs=context_trajs,
                        target_trajs=target_trajs,
                        oracle=oracle_flag,
                        oracle_data={
                            "latex": latex,
                            "sphere_id": sphere_id,
                        }
                    )
                    prediction, mat_props = self.predict(batch)

                    if oracle_flag:
                        # use the one "gth" ones
                        mat_props = batch["h"][0, 0, 0, 4:]

                    print("relevant mat prop:", mat_props[2])

                    meta_data = {"diameter": str(float(diameter))}

                    for key, val in batch.items():
                        if isinstance(val, torch.Tensor):
                            batch[key] = val.to("cpu")

                    results = {}
                    target_h = batch["h"][0][batch["target_trajs"][0]]

                    # predicted collider
                    collider_mask = target_h[0, :, 3] == 1
                    results["predicted_collider"] = (
                        prediction[-1, :, collider_mask, :].detach().cpu()
                    )
                    # also update the deformable trajectory: keep only the ones which are not collider nodes
                    results["predicted_trajectory"] = (
                        prediction[-1, :, ~collider_mask, :].detach().cpu()
                    )
                    # save connectivity
                    results["collider_connectivity"] = torch.tensor(
                        get_collider_connectivity(self.sim_dataset, meta_data)
                    )
                    results["mesh_connectivity"] = get_mesh_connectivity(
                        self.sim_dataset, meta_data
                    )
                    results["pointcloud"] = batch["pc"][0][
                        batch["target_trajs"].to("cpu")[0]
                    ][-1]
                    results["pc_time_stamps"] = (
                        stacked_time_stamps[batch["target_trajs"].to("cpu")[0]][-1]
                        .detach()
                        .cpu()
                        * self.config.pc_time_stamp_scaler + self.config.pc_time_stamp_offset
                    )
                    results = self.interpolate_results(results, pc_to_mesh=False, verbose=False)

                    loss_over_time = self.get_pc_loss_over_time(results)
                    material_id = f"latex-{latex}"
                    condition_id = f"{material_id}_sphere-{sphere_id}"
                    traj_id = int(target_traj_idx)
                    result_output_dir = os.path.join(
                        results_root, material_id, f"sphere-{sphere_id}"
                    )
                    os.makedirs(result_output_dir, exist_ok=True)
                    result_output_path = os.path.join(
                        result_output_dir, f"traj-{traj_id}.pt"
                    )
                    result_dict = {
                        "seed": seed,
                        "config_seed": int(self.config.seed),
                        "model_checkpoint": self._active_model_checkpoint,
                        "latex": latex,
                        "sphere_id": sphere_id,
                        "material_id": material_id,
                        "condition_id": condition_id,
                        "diameter": diameter,
                        "target_traj_idx": traj_id,
                        "context_trajs": context_trajs.detach().cpu(),
                        "target_trajs": target_trajs.detach().cpu(),
                        "mat_props": mat_props.detach().cpu(),
                        "loss_over_time": loss_over_time.detach().cpu(),
                        "loss_mean": loss_over_time.mean().detach().cpu(),
                        "loss_final": loss_over_time[-1].detach().cpu(),
                        "results": results,
                    }
                    torch.save(result_dict, result_output_path)
                    print(f"Wrote results to {result_output_path}")
                    # plot loss over time
                    # plt.figure()
                    # plt.plot(loss_over_time.numpy())
                    # plt.xlabel("Timestep")
                    # plt.ylabel("Point-to-Mesh Loss")
                    # plt.title(
                    #     f"Point-to-Mesh Loss over Time for Latex {latex}, Sphere {sphere_id}, Target Traj {target_traj_idx}"
                    # )
                    # plt.grid()
                    # plt.show()

                    if xdmf_counts.get(condition_id, 0) < xdmf_per_condition:
                        vis_output_path = os.path.join(
                            vis_root,
                            material_id,
                            f"sphere-{sphere_id}_traj-{traj_id}",
                        )
                        os.makedirs(vis_output_path, exist_ok=True)

                        export_two_meshes_to_xdmf_merged(
                            positions_A=results["predicted_trajectory"],
                            connectivity_A=results["mesh_connectivity"],
                            positions_B=results["predicted_collider"],
                            connectivity_B=results["collider_connectivity"],
                            cell_type="triangle",
                            out_path=os.path.join(
                                vis_output_path, "predicted_trajectory.xdmf"
                            ),
                        )
                        write_pointcloud_timeseries_pvd(
                            results["pointcloud"],
                            out_dir=vis_output_path,
                            name="gth_pc",
                        )
                        xdmf_counts[condition_id] = (
                            xdmf_counts.get(condition_id, 0) + 1
                        )

    def predict(self, batch):
        self.encoder.eval()
        self.simulator.eval()
        self.mat_prop_head.eval()
        self.meta_aggregation.eval()
        with torch.no_grad():
            # move to device
            device = next(self.simulator.parameters()).device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            pc = batch["pc"][0][
                batch["context_trajs"][0]
            ]  # shape (context_size, traj_length, num_points, point_dim)
            color = batch["pc_color"][0][
                batch["context_trajs"][0]
            ]  # shape (context_size, traj_length, num_points, color_dim)
            encoder_results = self.encoder(pc, color)  # (context_size, latent_dim)
            if isinstance(encoder_results, tuple):
                latent_pred, tokens, center_scaled_spacetime = encoder_results
            else:
                latent_pred = encoder_results
                tokens, center_scaled_spacetime = None, None
            latent_pred = self.meta_aggregation(latent_pred)

            if self.config.algorithm.get("test_opt_oracle", False):
                batch = self.test_opt_oracle(batch, latent_pred)

            # simulate with predicted material properties
            prediction = self.simulator(batch, latent_pred)
            mat_props = self.mat_prop_head(latent_pred)[0]

            return prediction, mat_props

    def get_pc_loss_over_time(self, results):
        return compute_point_to_mesh_loss2(
            results["pointcloud"],
            results["predicted_trajectory"],
            results["mesh_connectivity"],
            results["predicted_collider"],
            results["collider_connectivity"],
            compiled=False,
            verbose=False,
        )


    def test_opt_oracle(self, batch, latent_pred):
        self.device = batch["h"].device
        if isinstance(self.sim_dataset, TrampolineDataset):
            dataset = "trampoline_v4"
        else:
            raise NotImplementedError("Dataset type not supported for test_opt_oracle")
        mat_prop_dim = 5
        if dataset == "trampoline_v4":
            mat_params = torch.nn.Parameter(
                torch.tensor([ 0.7935, -0.2937, -1.7150, -2.1124, -1.1252], device=self.device)
            )
            PC_SUBSAMPLE = None  # number of point cloud points to keep  (None = all)
            # TIME_SUBSAMPLE = 10  # use every Nth timestep during opt      (1 = all)
        else:
            raise NotImplementedError

        if dataset == "db_v4":
            lr = 0.03
            num_steps = 50
        elif dataset == "sd_v1":
            lr = 0.015
            num_steps = 200
        elif dataset == "trampoline_v4":
            lr = 0.06
            num_steps = 300
        elif dataset == "bbv_v3":
            lr = 0.015
            num_steps = 200
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam([mat_params], lr=lr)

        if dataset == "db_v4":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        elif dataset == "sd_v1":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
        elif dataset == "trampoline_v4":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
        elif dataset == "bbv_v3":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25)
        else:
            raise NotImplementedError

        best_loss = float('inf')
        best_params = mat_params.detach().clone()

        original_batch = copy.deepcopy(batch)

        try:
            # set target_trajs as context_trajs
            batch["target_trajs"] = batch["context_trajs"]
            if self.config.algorithm.test_opt_oracle:
                batch["target_trajs"] = batch["target_trajs"][0, :1] # only 1 cause OOM
            gth_pointcloud = batch["pc"][0]

            # get faces for the trajectory and the collider
            meta_data = batch["meta_data"] if "meta_data" in batch else {}
            if "diameter" in meta_data:
                col_meta_data = {"diameter":  str(meta_data["diameter"][0].item())}
            else:
                col_meta_data = meta_data
            faces_trajectory = get_mesh_connectivity(self.sim_dataset, meta_data)
            if dataset == "bbv_v3":
                faces_trajectory = faces_trajectory[0]
            faces_collider = get_collider_connectivity(self.sim_dataset, col_meta_data)
            if faces_collider is None:
                faces_collider = torch.empty((0, 3), dtype=torch.long)
            else:
                faces_collider = torch.tensor(faces_collider)

            # remove h last columns (or set to zero rather)
            if len(batch["h"].shape) == 5:
                batch["h"][:, :, :, :, -mat_prop_dim:] = 0
            else:
                batch["h"][:, :, :, -mat_prop_dim:] = 0
            if dataset == "sd_v1":
                batch["regression_features"] = batch["regression_features"].view(-1, mat_prop_dim)
            print("------------------------------------------------")
            for step in range(num_steps):
                optimizer.zero_grad()

                # with torch.enable_grad():  # re-enable gradients inside no_grad context
                with torch.no_grad():
                    if dataset == "db_v4":
                        # set the current mat_params in the batch for simulator
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            torch.tensor(0.5).to("cuda").expand_as(batch["h"][..., 0:1]),
                            mat_params.expand_as(batch["h"][..., 0:1])
                        ], dim=-1)
                    elif dataset == "sd_v1":
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            mat_params.expand_as(batch["h"][..., 0:1])
                        ], dim=-1)
                    elif dataset == "trampoline_v4":
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            mat_params.expand_as(batch["h"][..., 0:mat_prop_dim])
                        ], dim=-1)
                    elif dataset == "bbv_v3":
                        batch["h"] = torch.cat([
                            batch["h"][..., :-mat_prop_dim],
                            mat_params.expand_as(batch["h"][..., 0:mat_prop_dim])
                        ], dim=-1)
                    else:
                        raise NotImplementedError
                    prediction = self.simulator(batch, latent_pred)
                    print("prediction shape: ", prediction.shape, "gth shape: ", gth_pointcloud.shape)
                    if dataset in ["db_v4", "sd_v1", "bbv_v3"]:
                        # for now pointclouds always have stride 2. if that is not the case: change this and make it parametric
                        prediction = prediction[:, ::2, :, :]
                    # add z dim for loss
                    if prediction.shape[-1] == 2:
                        prediction = torch.cat([prediction, torch.zeros_like(prediction[:, :, :, :1])], dim=-1)
                    assert prediction.shape[1] == gth_pointcloud.shape[1], "Prediction and ground truth pointcloud have different number of time steps, adjust stride or check data"
                    # check for each context traj separately and average the loss over them
                    if dataset != "bbv_v3":
                        collider = batch["x"][0][batch["target_trajs"][0]][:, ::2, batch["h"][0, 0, :, 0] == 0, :]
                    else:
                        # there is no collider
                        collider = torch.empty((prediction.shape[0], prediction.shape[1], 0, 3), device=prediction.device)
                    # add z dim
                    if collider.shape[-1] == 2:
                        collider = torch.cat([collider, torch.zeros_like(collider[:, :, :, :1])], dim=-1)
                    total_loss = torch.tensor(0.0).to(prediction.device)
                    for pc_traj, prediction_traj, collider_traj in zip(gth_pointcloud, prediction, collider):
                        # debug plot
                        # visualize_pointcloud_sequence(pc_traj, types=torch.zeros_like(pc_traj)[..., 0:1])
                        # visualize_pointcloud_sequence(prediction_traj.detach(), types=torch.zeros_like(prediction_traj.detach())[..., 0:1])
                        # loss = compute_point_to_mesh_loss(pc_traj.to("cuda"), prediction_traj.to("cuda"), faces_trajectory.to("cuda"),
                        #                            collider_traj.to("cuda"), faces_collider.to("cuda"), verbose=False).sum()
                        if dataset == "trampoline_v4":
                            if PC_SUBSAMPLE is not None and pc_traj.shape[1] > PC_SUBSAMPLE:
                                perm = torch.randperm(pc_traj.shape[1], device=pc_traj.device)[:PC_SUBSAMPLE]
                                pc_traj = pc_traj[:, perm]  # [T, PC_SUBSAMPLE, 3]
                            mesh_traj = prediction_traj[:, batch["h"][0, 0, :, 2] == 1, :]
                            collider_traj = prediction_traj[:, batch["h"][0, 0, :, 3] == 1, :]
                            loss = compute_point_to_mesh_loss2(pc_traj[:, ...].to("cuda"),
                                                               mesh_traj[:, ...].to("cuda"),
                                                               faces_trajectory.to("cuda"),
                                                               collider_traj[:, ...].to("cuda"), faces_collider.to("cuda"),
                                                               verbose=False).sum()
                        else:
                            loss = compute_point_to_mesh_loss2(pc_traj[:, ...].to("cuda"), prediction_traj[:,...].to("cuda"), faces_trajectory.to("cuda"),
                                                       collider_traj[:,...].to("cuda"), faces_collider.to("cuda"), compiled=False, verbose=False).sum()
                        total_loss += loss
                    # correct magnitude
                    total_loss *= 10000
                    total_loss.backward()
                    print("total_loss: ", total_loss.item())
                optimizer.step()
                scheduler.step()

                print("current mat params: ", mat_params.detach().cpu().numpy(), "gth: ", batch["regression_features"][0, :mat_prop_dim].cpu().numpy())
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_params = mat_params.detach().clone()
        except torch.OutOfMemoryError:
            print("OOM during test-time optimization, returning best found parameters so far.")
        # insert best params into batch for final evaluation
        if dataset == "db_v4":
            batch["h"] = torch.cat([
                original_batch["h"][..., :-mat_prop_dim],
                torch.tensor(0.5).to("cuda").expand_as(original_batch["h"][..., 0:1]),
                best_params.expand_as(original_batch["h"][..., 0:1])
            ], dim=-1)
        elif dataset == "sd_v1":
            batch["h"] = torch.cat([
                original_batch["h"][..., :-mat_prop_dim],
                best_params.expand_as(original_batch["h"][..., 0:1])
            ], dim=-1)
        elif dataset in ["trampoline_v4", "bbv_v3"]:
            batch["h"] = torch.cat([
                original_batch["h"][..., :-mat_prop_dim],
                best_params.expand_as(original_batch["h"][..., 0:mat_prop_dim])
            ], dim=-1)
        else:
            raise NotImplementedError
        return batch

    def load_pointcloud(
        self, hdf5_path: str, sphere_id: int, sheet_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(hdf5_path, "r") as pcd_file:
            pc_data = hdf5_group_to_dict(pcd_file["obs"]["pcd"])
            time_stamps = pcd_file["obs"]["cpu_time"][:]
            time_stamps = time_stamps - time_stamps[0]
        raw_pos_list = []
        raw_color_list = []
        # get the min number of points across all frames to ensure consistent tensor shapes
        min_points = 1000000
        for key in sorted(pc_data.keys(), key=lambda x: int(x[1:])):
            min_points = min(min_points, pc_data[key]["pos"].shape[0])
        for key in sorted(pc_data.keys(), key=lambda x: int(x[1:])):
            pos = pc_data[key]["pos"]
            color = pc_data[key]["x"]
            # more points now for preproc, later we subsample to the final size
            indices = fpsample.fps_sampling(pos, n_samples=min_points)
            raw_pos_list.append(pos[indices])
            raw_color_list.append(color[indices])
        pos = torch.stack(raw_pos_list)
        color = torch.stack(raw_color_list)

        # move to 0 0 0
        initial_pc = pos[0]
        sheet_pc = initial_pc[initial_pc[:, 2] < 4.0]
        sheet_avg = sheet_pc.mean(dim=0)
        pos = pos - sheet_avg[None, None, :]

        # Create initial pink mask (for reference estimation)
        if sheet_id == 1:
            pink_mask_init = classify_pink_hsv(
                color, hue_range=(300, 60), saturation_min=0.2, value_min=0.2
            )
        else:
            pink_mask_init = classify_pink_hsv(
                color, hue_range=(320, 20), saturation_min=0.2, value_min=0.2
            )
        # Estimate pink reference
        pink_ref_rgb, pink_ref_hsv = estimate_pink_reference(color, pink_mask_init)
        pc_type = iterative_pink_segmentation_knn(
            color,
            pos,
            pink_ref_rgb=None,
            pink_ref_hsv=pink_ref_hsv,
            n_neighbors=10,
            max_iterations=50,
            color_weight=0.8,
            neighbor_weight=0.2,
            init_threshold=0.3,  # higher means more points classified initially
            iter_threshold=0.7,  # higher means less points classified
            verbose=False,
        )
        pos, pc_type, center_pos = crop_robot_out_and_subsample(
            pos,
            pc_type,
            sphere_id=sphere_id,
            margin=3.0,
            z_threshold=50.0,
            final_num_points=self.config.fps_num_points,
        )

        # set the pink type to all points above a z threshold now
        pc_type[pos[:, :, 2] > 4] = 1
        # check if there are more than x points higher than z threshold, if yes set all points below to not pink
        pos_above = pos[:, :, 2] > 30
        ts_above = pos_above.sum(dim=1) > 10
        nodes_to_reset = ts_above[:, None] & (pos[:, :, 2] < 10)
        pc_type[nodes_to_reset] = 0

        rel_diffs = center_pos[1:] - center_pos[:1]
        rel_diffs = torch.norm(rel_diffs, dim=-1)
        is_movement = rel_diffs > 4.0
        # find first index of movement
        movement_start_idx = torch.where(is_movement)[0][0]

        start_time = time_stamps[movement_start_idx]
        # physical computation
        # desired_end_time = start_time + 0.5
        # end_idx = torch.searchsorted(torch.from_numpy(time_stamps), desired_end_time)
        # use x frames
        end_idx = movement_start_idx + self.config.pc_length

        # crop pointcloud to movement start
        pos = pos[movement_start_idx:end_idx]
        pc_type = pc_type[movement_start_idx:end_idx]
        time_stamps = time_stamps[movement_start_idx:end_idx]
        center_pos = center_pos[movement_start_idx:end_idx]
        # normalize to 0
        time_stamps = time_stamps - time_stamps[0]
        # visualize_pointcloud_sequence(pos, pc_type[:, :, None])

        return pos, pc_type, center_pos, time_stamps

    def build_initial_mesh(self, initial_center, diameter):
        # get sheet mesh
        sheet_pos = self.initial_sheet_pos
        sheet_faces = self.faces["sheet_faces"]
        # get sphere mesh
        wrongly_placed_sphere_pos = self.sphere_pos[f"{diameter:.1f}"]
        sphere_faces = self.faces["ball_faces"][f"{diameter:.1f}"]
        # align sphere mesh to initial position
        wrong_center = torch.mean(wrongly_placed_sphere_pos, dim=0)
        translation = initial_center - wrong_center
        aligned_sphere_pos = wrongly_placed_sphere_pos + translation
        return sheet_pos, sheet_faces, aligned_sphere_pos, sphere_faces


    @staticmethod
    def interpolate_results(vis_results, pc_to_mesh=True, verbose=False):
        sim_time_stamps = (
            torch.arange(vis_results["predicted_trajectory"].shape[0]) * 0.02
        )
        if pc_to_mesh:
            pc_time_stamps = vis_results["pc_time_stamps"]
            output_pc = []
            for ts in sim_time_stamps:
                # find closest pc time stamp
                diff = pc_time_stamps - ts
                closest_index = torch.argmin(torch.abs(diff))
                if verbose:
                    print(ts, closest_index)
                output_pc.append(vis_results["pointcloud"][closest_index])
            vis_results["pc_time_stamps"] = sim_time_stamps
            vis_results["pointcloud"] = torch.stack(output_pc, dim=0)
        else:
            # Interpolate mesh to pointcloud timesteps
            pc_time_stamps = vis_results["pc_time_stamps"]

            mesh_trajectory = vis_results["predicted_trajectory"]  # shape (25, 1201, 3)
            mesh_collider = vis_results["predicted_collider"]  # shape (25, 455, 3)

            interpolated_trajectory = []
            interpolated_collider = []

            skipped_pc_indices = []

            for pc_idx, pc_ts in enumerate(pc_time_stamps):
                # Find the 2 closest mesh timesteps
                diff = sim_time_stamps - pc_ts
                sorted_indices = torch.argsort(torch.abs(diff))

                idx_closest = sorted_indices[0].item()
                idx_second = sorted_indices[1].item()

                # check that shortest distance is smaller than 0.02, otherwise pc is out of mesh time and we discard it
                if diff[sorted_indices[0]].abs() > 0.02:
                    if verbose:
                        print(f"PC timestamp {pc_ts.item():.4f} is out of mesh time range, skipping interpolation")
                    skipped_pc_indices.append(pc_idx)
                    continue

                # Ensure idx_below < pc_ts <= idx_above (or both for extrapolation)
                if sim_time_stamps[idx_closest] <= pc_ts:
                    idx_below = idx_closest
                    idx_above = idx_second
                else:
                    idx_below = idx_second
                    idx_above = idx_closest

                t_below = sim_time_stamps[idx_below].item()
                t_above = sim_time_stamps[idx_above].item()

                # Linear interpolation parameter
                # t_interp in [0, 1] means between the two points
                # t_interp < 0 or > 1 means extrapolation
                if t_above != t_below:
                    t_interp = (pc_ts.item() - t_below) / (t_above - t_below)
                else:
                    t_interp = 0.0
                # extrapolation does not work well, clamp to 0, 1 for now
                t_interp = max(0.0, min(1.0, t_interp))

                if verbose:
                    print(
                        f"PC ts: {pc_ts.item():.4f}, mesh ts below: {t_below:.4f}, above: {t_above:.4f}, t_interp: {t_interp:.4f}"
                    )

                # Linear interpolation: x(t) = x_below + t_interp * (x_above - x_below)
                interp_traj = mesh_trajectory[idx_below] + t_interp * (
                    mesh_trajectory[idx_above] - mesh_trajectory[idx_below]
                )
                interp_coll = mesh_collider[idx_below] + t_interp * (
                    mesh_collider[idx_above] - mesh_collider[idx_below]
                )

                interpolated_trajectory.append(interp_traj)
                interpolated_collider.append(interp_coll)

            # Stack and update results
            vis_results["predicted_trajectory"] = torch.stack(
                interpolated_trajectory, dim=0
            )  # shape (num_non_skipped_ts, 1201, 3)
            vis_results["predicted_collider"] = torch.stack(
                interpolated_collider, dim=0
            )  # shape (num_non_skipped_ts, 455, 3)
            # also update pc time stamps and pointcloud to only include non-skipped indices
            vis_results["pc_time_stamps"] = vis_results["pc_time_stamps"][
                torch.tensor([i for i in range(len(pc_time_stamps)) if i not in skipped_pc_indices])
            ]
            vis_results["pointcloud"] = vis_results["pointcloud"][
                torch.tensor([i for i in range(len(pc_time_stamps)) if i not in skipped_pc_indices])
            ]

            if verbose:
                print(
                    f"Interpolated mesh to PC timesteps. New trajectory shape: {vis_results['predicted_trajectory'].shape}"
                )
                print(f"New collider shape: {vis_results['predicted_collider'].shape}")

        return vis_results

    @staticmethod
    def get_checkpoint_seed(model_checkpoint) -> str:
        match = re.search(r"seed[_-]?(\d+)", model_checkpoint)
        if match is not None:
            return match.group(1)
        return "unknown"


def visualize_meshes(sheet_pos, sheet_faces, sphere_pos, sphere_faces, pc=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --- Sheet mesh ---
    ax.plot_trisurf(
        sheet_pos[:, 0],
        sheet_pos[:, 1],
        sheet_pos[:, 2],
        triangles=sheet_faces,
        color="lightblue",
        alpha=0.6,
        edgecolor="none",
    )

    # --- Sphere mesh ---
    ax.plot_trisurf(
        sphere_pos[:, 0],
        sphere_pos[:, 1],
        sphere_pos[:, 2],
        triangles=sphere_faces,
        color="tomato",
        alpha=0.4,
        edgecolor="none",
    )

    if pc is not None:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c="green", s=5, alpha=0.8)

    # --- Equal aspect ratio ---
    all_pts = torch.cat([sheet_pos, sphere_pos], dim=0)
    mins = all_pts.min(dim=0).values
    maxs = all_pts.max(dim=0).values
    mid = (mins + maxs) / 2
    half = (maxs - mins).max() / 2

    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
