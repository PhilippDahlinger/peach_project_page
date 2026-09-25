import os
from io import BytesIO

import numpy as np
from meshio import WriteError
from meshio.xdmf import TimeSeriesWriter
from meshio.xdmf.common import dtype_to_format_string


class FixedTimeSeriesWriter(TimeSeriesWriter):

    def numpy_to_xml_string(self, data):
        if self.data_format == "XML":
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype.name]
            np.savetxt(s, data.flatten(), fmt)
            return s.getvalue().decode()
        elif self.data_format == "Binary":
            bin_save_path = f"{self.filename.parent / self.filename.stem}{self.data_counter}.bin"  # Fix this path
            bin_filename = f"{self.filename.stem}{self.data_counter}.bin"  # Fix this path
            self.data_counter += 1
            # write binary data to file
            with open(bin_save_path, "wb") as f:
                data.tofile(f)
            return bin_filename

        if self.data_format != "HDF":
            raise WriteError()
        name = f"data{self.data_counter}"
        self.data_counter += 1
        self.h5_file.create_dataset(name, data=data)
        return os.path.basename(self.h5_filename) + ":/" + name

    def __enter__(self):
        """Open the XDMF file for writing."""

        if self.data_format == "HDF":
            import h5py

            self.h5_filename = self.filename.stem + ".h5"
            self.h5_file = h5py.File(self.filename.parent / self.h5_filename, "w")
        return self
