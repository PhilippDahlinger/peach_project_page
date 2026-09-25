from pc_mango.dataset.step_datasets.abstract_step_ggns_dataset import AbstractStepGGNSDataset


class StepGGNSSheetDeformationDataset(AbstractStepGGNSDataset):
    """GGNS dataset for the 3D sheet deformation environment."""

    def get_normalization_params(self):
        return {
            "min_x": 0.0,
            "max_x": 140.0,
            "normalize_mesh": True,
            "normalize_collider": False,  # No collider in this dataset
            "normalize_pcd": False,       # PCDs already pre-normalized (mesh/100)
        }

    def postprocess_pc(self, pc):
        # renormalize pointcloud, right now it is normalized by /100 from the dataset
        # however, the mesh is normalized by /140, and we need to match
        # so multiply by 100/140
        pc = pc * (100.0 / 140.0)
        return pc

