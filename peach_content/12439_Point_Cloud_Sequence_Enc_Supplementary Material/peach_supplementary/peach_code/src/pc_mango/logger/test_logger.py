import os
from argparse import Namespace
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import OmegaConf

from pc_mango.visualization.util import save_visualizations


class TestLogger(Logger):
    """
    A custom logger for testing purposes that saves metrics to disk. Currently, no visualization logging is implemented.
    """
    def __init__(self, output_path, exp_name, job_type, seed, meta_data: dict):
        super().__init__()
        # Initialize your logger (e.g., file writer, custom logging server, etc.)
        self.output_path = os.path.join(output_path, exp_name, job_type, seed)
        # create output path
        os.makedirs(self.output_path, exist_ok=True)
        # save metadata
        with open(os.path.join(self.output_path, "meta_data.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(meta_data))

    @property
    def name(self) -> Optional[str]:
        return "Test Logger"

    @property
    def version(self) -> Optional[Union[int, str]]:
        return "1.1"

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    def log_metrics(self, outputs, step):
        metrics = outputs["metrics"]
        config = outputs.get("config", None)
        visualizations = outputs.get("visualizations", None)
        mat_props = outputs.get("material_properties", None)
        mat_prop_description = outputs.get("mat_prop_description", None)

        # Implement how metrics are logged (e.g., to a file or external system)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                torch.save(value, os.path.join(self.output_path, f"{key}.pt"))
                arr = value.detach().cpu().numpy()
                # check for 0d array
                if arr.ndim == 0:
                    arr = arr[None]
                np.savetxt(os.path.join(self.output_path, f"{key}.csv"), arr, delimiter=",", fmt="%0.8g")
            else:
                raise ValueError(f"Cannot save metric of type {type(value)} for key {key}.")

        if config is not None:
            # save config
            with open(os.path.join(self.output_path, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(config, resolve=True))

        if visualizations is not None:
            # save visualizations
            for task_idx, vis_dict in enumerate(visualizations):
                if not vis_dict:
                    # empty dict, no vis logged for this task
                    continue
                task_path = os.path.join(self.output_path, "vis", f"task_{task_idx}")
                os.makedirs(task_path, exist_ok=True)
                result = save_visualizations(vis_dict, task_path)
                if result:
                    print(f"Saved visualizations for task {task_idx} in {task_path}.")

        if mat_props is not None and mat_prop_description is not None:
            mat_props = torch.cat(mat_props, dim=0).detach().cpu().numpy()
            # check if there are 0d arrays and add dimension if needed
            if mat_props.ndim == 1:
                mat_props = mat_props[:, None]
            mat_props_path = os.path.join(self.output_path, "material_properties.csv")
            task_ids = np.arange(len(mat_props)).reshape(-1, 1)
            mat_props = np.hstack([task_ids, mat_props])
            header = ",".join(["Task ID"] + list(mat_prop_description))
            np.savetxt(mat_props_path, mat_props,
                       delimiter=",", fmt="%0.8g", header=header, comments="")


        print(f"Saved test metrics in {self.output_path}.")

    def save(self):
        # Optionally implement saving or finalizing logging state if needed
        pass

    def finalize(self, status):
        # Finalize logging when done (e.g., close connections)
        pass