from pathlib import Path

from hgns.utils.xdmf_utils.fixed_time_series_reader import FixedTimeSeriesReader
from hgns.utils.xdmf_utils.fixed_time_series_writer import FixedTimeSeriesWriter


def convert_binary_xdmf_to_xml(data_path: str, target_path: str = None):
    """Converts a binary XDMF file to an XML XDMF file."""

    t_list, point_data_list, cell_data_list = [], [], []
    with FixedTimeSeriesReader(data_path) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k)
            t_list.append(t)
            point_data_list.append(point_data)
            cell_data_list.append(cell_data)

    if target_path is None:
        target_path = Path(data_path).parent / (Path(data_path).stem + "_xml.xdmf")
    with FixedTimeSeriesWriter(
        target_path, data_format="XML"
    ) as writer:  # Binary format is more efficient but cannot be readed by Paraview. HDF cannot be reimportet in meshio. XML can be readed by Paraview, but is inefficient.
        writer.write_points_cells(points, cells)
        for t, point_data, cell_data in zip(t_list, point_data_list, cell_data_list):
            writer.write_data(t, point_data=point_data, cell_data=cell_data)


def convert_binary_xdmf_to_hdf(data_path: str, target_path: str = None):
    """Converts a binary XDMF file to an XML XDMF file."""

    t_list, point_data_list, cell_data_list = [], [], []
    with FixedTimeSeriesReader(data_path) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k)
            t_list.append(t)
            point_data_list.append(point_data)
            cell_data_list.append(cell_data)

    if target_path is None:
        target_path = Path(data_path).parent / (Path(data_path).stem + "_xml.xdmf")
    with FixedTimeSeriesWriter(target_path, data_format="HDF") as writer:
        writer.write_points_cells(points, cells)
        for t, point_data, cell_data in zip(t_list, point_data_list, cell_data_list):
            writer.write_data(t, point_data=point_data, cell_data=cell_data)


if __name__ == "__main__":
    convert_binary_xdmf_to_xml(
        "/mnt/nvme_ssd/work/hgns/data/simulation/geometric_nonlinear_deformation_large_changing_bc_1316_meshes2/train/sim_00000000/meshes/mesh.xdmf"
    )

    #
