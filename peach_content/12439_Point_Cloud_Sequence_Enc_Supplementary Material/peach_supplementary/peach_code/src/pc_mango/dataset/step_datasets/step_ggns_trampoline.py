from pc_mango.dataset.step_datasets.abstract_step_ggns_dataset import AbstractStepGGNSDataset


class StepGGNSTrampolineDataset(AbstractStepGGNSDataset):
    """GGNS dataset for the 3D trampoline (sheet + sphere) environment."""

    def get_normalization_params(self):
        return {
            "min_x": -145.0,
            "max_x": 145.0,
            "normalize_mesh": True,
            "normalize_collider": True,
            "normalize_pcd": True,
        }
