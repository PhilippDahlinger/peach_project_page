from pathlib import Path

from hgns.utils.xdmf_utils.simulation_data import RawSimulationData


def load_xdmf(sim_data_path: Path) -> RawSimulationData:
    """Loads time-series mesh and field data from an XDMF file.

    This function reads the static mesh (points and cells) and then iterates
    through each time step to read the corresponding point and cell data.

    Args:
        sim_data_path: The file path to the .xdmf file.

    Returns:
        An XDMFSimulationData object containing the complete mesh and
        simulation data.
    """
    return RawSimulationData.load(sim_data_path)
