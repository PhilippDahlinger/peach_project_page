from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset
from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
from pc_mango.dataset.util.step_methods import step_get, step_length
from pc_mango.util.own_types import ConfigDict


class StepTrampolineDataset(TrampolineDataset):
    def __init__(self, config: ConfigDict, transform=None, pre_transform=None, pre_filter=None, split="train"):
        super().__init__(config, transform, pre_transform, pre_filter, split)
        self.start_time_steps = self.traj_length - 1

    def len(self):
        return step_length(self.config, self.tasks, self.task_size, self.start_time_steps)

    def get(self, idx):
        return step_get(
            idx=idx,
            config=self.config,
            super_get=super().get,
            start_time_steps=self.start_time_steps,
            task_size=self.task_size,
        )

