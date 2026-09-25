def get_dataset(config, loading=False):
    eval_split = "val" if not loading else "test"
    match config.name:
        case "db":
            from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset
            train_ds = DeformableBlockDataset(config.train_dataset, split="train")
            eval_ds = DeformableBlockDataset(config.eval_dataset, split=eval_split)
        case "step_db":
            from pc_mango.dataset.step_datasets.step_deformable_block import StepDeformableBlockDataset
            train_ds = StepDeformableBlockDataset(config.train_dataset, split="train")
            # Use ML dataset for evaluation since the eval framework is trajectory-based and the step dataset is not compatible with that.
            from pc_mango.dataset.ml_datasets.deformable_block import DeformableBlockDataset
            eval_ds = DeformableBlockDataset(config.eval_dataset, split=eval_split)
        case "sd":
            from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset
            train_ds = SheetDeformationDataset(config.train_dataset, split="train")
            eval_ds = SheetDeformationDataset(config.eval_dataset, split=eval_split)
        case "step_sd":
            from pc_mango.dataset.step_datasets.step_sheet_deformation import StepSheetDeformationDataset
            train_ds = StepSheetDeformationDataset(config.train_dataset, split="train")
            # Use ML dataset for evaluation since the eval framework is trajectory-based and the step dataset is not compatible with that.
            from pc_mango.dataset.ml_datasets.sheet_deformation import SheetDeformationDataset
            eval_ds = SheetDeformationDataset(config.eval_dataset, split=eval_split)
        case "trampoline":
            from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
            train_ds = TrampolineDataset(config.train_dataset, split="train")
            eval_ds = TrampolineDataset(config.eval_dataset, split=eval_split)
        case "step_trampoline":
            from pc_mango.dataset.step_datasets.step_trampoline import StepTrampolineDataset
            train_ds = StepTrampolineDataset(config.train_dataset, split="train")
            # Use ML dataset for evaluation since the eval framework is trajectory-based and the step dataset is not compatible with that.
            from pc_mango.dataset.ml_datasets.trampoline import TrampolineDataset
            eval_ds = TrampolineDataset(config.eval_dataset, split=eval_split)
        case "bbv":
            from pc_mango.dataset.ml_datasets.bbv import BBVDataset
            train_ds = BBVDataset(config.train_dataset, split="train")
            eval_ds = BBVDataset(config.eval_dataset, split=eval_split)
        case "step_bbv":
            from pc_mango.dataset.step_datasets.step_bbv import StepBBVDataset
            train_ds = StepBBVDataset(config.train_dataset, split="train")
            from pc_mango.dataset.ml_datasets.bbv import BBVDataset
            eval_ds = BBVDataset(config.eval_dataset, split=eval_split)

        case "step_ggns_db":
            from pc_mango.dataset.step_datasets.step_ggns_deformable_block import StepGGNSDeformableBlockDataset
            train_ds = StepGGNSDeformableBlockDataset(config.train_dataset, split="train")
            eval_ds = StepGGNSDeformableBlockDataset(config.eval_dataset, split=eval_split)
        case "step_ggns_sd":
            from pc_mango.dataset.step_datasets.step_ggns_sheet_deformation import StepGGNSSheetDeformationDataset
            train_ds = StepGGNSSheetDeformationDataset(config.train_dataset, split="train")
            eval_ds = StepGGNSSheetDeformationDataset(config.eval_dataset, split=eval_split)
        case "step_ggns_trampoline":
            from pc_mango.dataset.step_datasets.step_ggns_trampoline import StepGGNSTrampolineDataset
            train_ds = StepGGNSTrampolineDataset(config.train_dataset, split="train")
            eval_ds = StepGGNSTrampolineDataset(config.eval_dataset, split=eval_split)
        case _:
            raise ValueError(f"Dataset {config.name} unknown.")
    return train_ds, eval_ds
