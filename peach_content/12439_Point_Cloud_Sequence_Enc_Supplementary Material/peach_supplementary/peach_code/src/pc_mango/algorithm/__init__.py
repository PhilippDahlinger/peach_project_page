from lightning import LightningModule

from pc_mango.util.own_types import ConfigDict


def get_algorithm(config: ConfigDict, train_dl, train_ds, eval_ds, loading=False,
                  checkpoint_path=None) -> LightningModule:
    algorithm_type = config.type
    match algorithm_type:
        case "pc_to_mat_prop":
            from pc_mango.algorithm.pc_to_mat_prop import PcToMatProp
            algorithm_class = PcToMatProp
        case "pc_to_sim":
            from pc_mango.algorithm.pc_to_sim import PcToSim
            algorithm_class = PcToSim
        case "mesh_to_mat_prop":
            from pc_mango.algorithm.mesh_to_mat_prop import MeshToMatProp
            algorithm_class = MeshToMatProp
        case "mesh_to_sim":
            from pc_mango.algorithm.mesh_to_sim import MeshToSim
            algorithm_class = MeshToSim
        case "pc_to_occupancy":
            from pc_mango.algorithm.pc_to_occupancy import PcToOccupancy
            algorithm_class = PcToOccupancy
        case "ggns":
            from pc_mango.algorithm.ggns_algorithm import GGNSAlgorithm
            algorithm_class = GGNSAlgorithm
        case _:
            raise ValueError(f"Unknown algorithm {algorithm_type}")

    if loading:
        algorithm = algorithm_class.load_from_checkpoint(checkpoint_path, train_dl=train_dl, train_ds=train_ds,
                                                         eval_ds=eval_ds)
        # update the evaluator config since they might have changed
        algorithm.config.evaluator = config.get("evaluator", algorithm.config.evaluator)
    else:
        algorithm = algorithm_class(config, train_dl, train_ds, eval_ds)

    return algorithm
