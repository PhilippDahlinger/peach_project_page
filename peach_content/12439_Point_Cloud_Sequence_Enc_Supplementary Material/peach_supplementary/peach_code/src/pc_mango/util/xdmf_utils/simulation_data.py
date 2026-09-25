from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import meshio
import numpy as np


@dataclass
class RawSimulationData:
    """A container for raw time-series mesh data loaded from an XDMF file.

    Attributes:
        points (np.ndarray):
            Vertex coordinates. Shape: (num_vertices, dim).
        cells (List[meshio.CellBlock]):
            Connectivity information for the mesh elements (e.g., triangles, quads).
            Note: For a mesh with a single element type, this is often a list
            containing just one CellBlock.
        timesteps (List[float]):
            List of time values for each step of the simulation. May either be floating-point values (indicating
            the actual time) or integers (indicating the index of the timestep).
        point_data (List[Dict[str, np.ndarray]]):
            Time-series data associated with the mesh points. Each item in the
            list corresponds to a timestep. The dictionary maps data field names
            (e.g., "velocity", "pressure") to their NumPy array values.
        cell_data (List[Dict[str, np.ndarray]]):
            Time-series data associated with the mesh cells. Structured identically
            to point_data.
    """

    points: np.ndarray
    cells: List[meshio.CellBlock]
    timesteps: List[float]
    point_data: List[Dict[str, np.ndarray]]
    cell_data: List[Dict[str, np.ndarray]]
    pre_processed: bool = False  # Indicates if the data has been processed by a preprocessor.

    @property
    def time_idxs(self) -> List[int]:
        """Returns the time indices corresponding to the timesteps.

        Returns: A simple range from 0 to the number of timesteps minus one.
        """
        return list(range(len(self.timesteps)))

    def __iter__(self):
        """Allows unpacking the dataclass like a tuple. Mostly for compatibility with existing code."""
        yield self.points
        yield self.cells
        yield self.time_idxs  # Using time indices instead of actual time values for iteration
        yield self.point_data
        yield self.cell_data

    def save(self, save_path: Path, data_format: str = "HDF"):
        """Saves the simulation data to an XDMF file.

        Args:
            save_path: The file path for the output .xdmf file.
            data_format: The data storage format ("HDF", "XML", or "Binary").
        """
        from pc_mango.util.xdmf_utils.fixed_time_series_writer import FixedTimeSeriesWriter

        with FixedTimeSeriesWriter(save_path, data_format=data_format) as writer:
            writer.write_points_cells(self.points, self.cells)
            for t, pd, cd in zip(self.timesteps, self.point_data, self.cell_data):
                writer.write_data(t, point_data=pd, cell_data=cd)

    @staticmethod
    def load(sim_data_path: str | Path) -> "RawSimulationData":
        """Loads time-series mesh and field data from an XDMF file.

        Args:
            sim_data_path: The file path to the .xdmf file.

        Returns:
            A RawSimulationData object containing the complete mesh and simulation data.
        """
        timesteps = []
        point_data_per_step = []
        cell_data_per_step = []

        from pc_mango.util.xdmf_utils.fixed_time_series_reader import FixedTimeSeriesReader

        with FixedTimeSeriesReader(sim_data_path) as reader:
            points, cells = reader.read_points_cells()

            for k in range(reader.num_steps):
                t, point_data, cell_data = reader.read_data(k)
                timesteps.append(t)
                point_data_per_step.append(point_data)
                cell_data_per_step.append(cell_data)

        return RawSimulationData(
            points=points,
            cells=cells,
            timesteps=timesteps,
            point_data=point_data_per_step,
            cell_data=cell_data_per_step,
        )

    def apply(self, preprocessor: "SimulationPreprocessor") -> "RawSimulationData":
        """Applies a processing function to this data object.

        Args:
            preprocessor: A function that transforms the data.

        Returns:
            A new, processed RawSimulationData object.
        """
        # The preprocessor is expected to return a new instance
        processed_simulation_data = preprocessor(self)
        processed_simulation_data.pre_processed = True
        return processed_simulation_data


class SimulationPreprocessor(Protocol):
    def __call__(self, data: RawSimulationData, dim: Optional[int] = None) -> RawSimulationData:
        """Type alias for a function that preprocesses raw simulation data.

        Args:
            data: The raw simulation data to preprocess.
            dim: Optional dimension argument for preprocessing.

        Returns:
            A new instance of RawSimulationData with the preprocessed data.
        """
        ...
