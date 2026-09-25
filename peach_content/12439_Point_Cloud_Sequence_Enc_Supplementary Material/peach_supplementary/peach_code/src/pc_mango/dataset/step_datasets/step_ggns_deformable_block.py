from pc_mango.dataset.step_datasets.abstract_step_ggns_dataset import AbstractStepGGNSDataset


class StepGGNSDeformableBlockDataset(AbstractStepGGNSDataset):
    """GGNS dataset for the deformable block/plate environment."""

    def get_normalization_params(self):
        return {
            "min_x": -167.8057,
            "max_x": 215.2644,
            "normalize_mesh": True,
            "normalize_collider": True,
            "normalize_pcd": False,  # PCs already normalized in this dataset
        }
