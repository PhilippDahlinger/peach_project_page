import os

import hydra
import torch
from lightning import LightningModule
from sympy.abc import alpha

from pc_mango.algorithm.util.get_optimizer import _get_optimizer, _get_scheduler
from pc_mango.network.occupancy_network import get_occupancy_network
from pc_mango.network.pc_encoder.tokenizer import get_tokenizer
from pc_mango.util.own_types import ConfigDict
import matplotlib.pyplot as plt

class PcToOccupancy(LightningModule):
    """
    Takes a stream of pointclouds, predicts a material property for this stream. No simulation involved.
    """

    def __init__(self, config: ConfigDict, train_dl, train_ds: torch.utils.data.Dataset,
                 eval_ds: torch.utils.data.Dataset):
        super().__init__()
        self.config: ConfigDict = config
        self._train_ds = train_ds
        self._eval_ds = eval_ds
        self._vis_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "visualizations")

        # load networks
        example_batch = None
        for example_batch in train_dl:
            break
        self.tokenizer = get_tokenizer(config.tokenizer, example_batch)
        self.occupancy_network = get_occupancy_network(config.occupancy_network, example_batch)

        # temp dicts to save eval results
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.best_val_mse = 1000000.0

        self.criterion = torch.nn.BCEWithLogitsLoss()

        # save config as hyperparameter
        self.save_hyperparameters("config")

    def training_step(self, batch, batch_idx):
        result = self._evaluate_batch(batch, batch_idx)
        loss = result["metrics"]["loss"]
        self.training_step_outputs.append(loss.detach())
        return loss

    def on_train_epoch_start(self):
        print("Training Epoch: ", self.current_epoch)

    def on_train_epoch_end(self):
        # `outputs` is a list of losses from the `training_step` for each batch
        # Calculate the mean loss for the epoch
        epoch_average = torch.stack(self.training_step_outputs).mean()
        # Log the mean epoch loss to WandB
        self.log('train_loss', epoch_average, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        result = self._evaluate_batch(batch, batch_idx, visualization=True)
        self.validation_step_outputs.append(result)
        return result

    def _evaluate_batch(self, batch, batch_idx, visualization=False):
        total_loss = 0.0
        total_correct = 0
        total_queries = 0

        pc = batch["pc"][0]  # shape (context_size, traj_length, num_points, point_dim)
        pc_color = batch["pc_color"][0]
        queries = batch["queries"][0]
        occupancy_labels = batch["occupancy"][0]
        num_trajectories = pc.shape[0]

        vis_results = {}

        for traj_idx in range(num_trajectories):
            pc_trajectory = pc[traj_idx]
            num_timesteps, num_points, _ = pc_trajectory.shape
            device = pc_trajectory.device

            pc_batch_idx = torch.arange(num_timesteps, device=device).view(-1, 1).repeat(1, num_points).view(-1)
            pc_trajectory_flat = pc_trajectory.view(-1, 3)
            pc_color_flat = pc_color[traj_idx].view(-1, pc_color.shape[-1])

            tokens, center_points, graph_batch_idx = self.tokenizer(pc_color_flat, pc_trajectory_flat, pc_batch_idx)

            queries_trajectory = queries[traj_idx]
            queries_batch = torch.arange(num_timesteps, device=device).view(-1, 1).repeat(1, queries_trajectory.shape[
                1]).view(-1)
            queries_flat = queries_trajectory.view(-1, 3)

            occupancy_logits = self.occupancy_network(queries_flat, queries_batch, tokens, center_points,
                                                      graph_batch_idx)
            occupancy_labels_flat = occupancy_labels[traj_idx].view(-1, 2)

            # ----- LOSS -----
            loss = self.criterion(occupancy_logits, occupancy_labels_flat)
            total_loss += loss

            # ----- ACCURACY -----
            preds = (torch.sigmoid(occupancy_logits) > 0.5)
            targets = occupancy_labels_flat.bool()
            correct_queries = (preds == targets).all(dim=1).sum().item()
            total_queries += targets.shape[0]
            total_correct += correct_queries

            # ----- VISUALIZATION -----
            if visualization and traj_idx == 0 and batch_idx in self.config.evaluator.task_ids_to_visualize:
                # Select only points and queries from the last timestep

                # Select only points and queries from the last timestep
                last_timestep_idx = pc_batch_idx.max()

                # Filter point cloud to last timestep
                last_pc_mask = pc_batch_idx == last_timestep_idx
                pc_xy = pc_trajectory_flat[last_pc_mask][:, :2].cpu().numpy()  # z-projection

                # Filter queries to last timestep
                last_queries_mask = queries_batch == last_timestep_idx
                queries_xy = queries_flat[last_queries_mask][:, :2].cpu().numpy()
                preds_np = preds[last_queries_mask].cpu().numpy()

                # Map predictions to colors
                color_map = {
                    (0, 0): "green",
                    (1, 0): "red",
                    (0, 1): "gray",
                    (1, 1): "purple"
                }
                query_colors = [color_map[tuple(p)] for p in preds_np]

                # Create figure
                fig, ax = plt.subplots(figsize=(6, 6))

                # Plot point cloud (larger x markers)
                ax.scatter(
                    pc_xy[:, 0],
                    pc_xy[:, 1],
                    c="blue",
                    s=20,
                    marker="x",
                    linewidths=1.0,
                    label="PointCloud"
                )

                # Plot occupancy queries
                ax.scatter(
                    queries_xy[:, 0],
                    queries_xy[:, 1],
                    c=query_colors,
                    s=10,
                    label="Occupancy Queries",
                    alpha=0.25
                )

                # Formatting
                ax.set_title(f"Trajectory {traj_idx} Visualization (last timestep)")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect("equal", adjustable="box")
                ax.legend()

                plt.tight_layout()

                # Store figure instead of showing it
                vis_results["trajectory_0"] = fig
        avg_loss = total_loss / num_trajectories
        accuracy = total_correct / total_queries
        metric_results = {"loss": avg_loss, "accuracy": torch.tensor(accuracy)}

        result = {"metrics": metric_results, "visualizations": vis_results}
        return result

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        # metrics
        metrics = [output["metrics"] for output in outputs]
        current_val_mse = None
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            val_loss = metric_values.mean()
            if metric_name == "loss":
                current_val_mse = val_loss
            # if self.logger is not None:
            self.log(f"val_{metric_name}", val_loss, on_epoch=True, prog_bar=True)
        if current_val_mse < self.best_val_mse:
            self.best_val_mse = current_val_mse
            # create a checkpoint from the encoder and save it
            ckpt_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                                     "best_tokenizer.ckpt")
            torch.save(self.tokenizer.state_dict(), ckpt_path)


        # visualizations
        vis_outputs = [output["visualizations"] for output in outputs]
        for batch_idx, vis_output in enumerate(vis_outputs):
            for traj_key, fig in vis_output.items():
                vis_dir = os.path.join(self._vis_path, f"epoch_{self.current_epoch}")
                os.makedirs(vis_dir, exist_ok=True)

                fig_path = os.path.join(
                    vis_dir,
                    f"val_batch_{batch_idx}_{traj_key}.pdf"
                )

                fig.savefig(fig_path, format="pdf", bbox_inches="tight")
                plt.close(fig)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        # TODO: is this correct? other algos have more code to generate the context sizes
        result = self._evaluate_batch(batch, batch_idx)
        self.test_step_outputs.append(result)
        return result

    def on_test_end(self) -> None:
        outputs = self.test_step_outputs
        metrics = [output["metrics"] for output in outputs]
        # average over all test batches
        converted_metrics = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            metric_values = torch.stack(metric_values)  # shape (num_batches, len(traj))
            test_metric = metric_values.mean()
            converted_metrics[f"test_{metric_name}"] = test_metric

        output = {
            "metrics": metrics,
            "config": self.config,
        }
        self.logger.log_metrics(output, self.current_epoch)
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = _get_optimizer(
            config=self.config.optimizer,
            models={
                "tokenizer": self.tokenizer,
                "occupancy_network": self.occupancy_network,
            }
        )
        scheduler = _get_scheduler(config=self.config.scheduler, optimizer=optimizer)
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": self.config.check_val_every_n_epoch,
            "monitor": "val_loss",
            "strict": False,
        }
        return [optimizer], [scheduler]
