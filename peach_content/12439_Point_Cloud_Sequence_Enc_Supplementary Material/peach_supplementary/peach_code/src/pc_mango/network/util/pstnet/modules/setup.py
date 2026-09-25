from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

# Absolute path to _ext_src
this_dir = os.path.dirname(os.path.abspath(__file__))
ext_src_root = os.path.join(this_dir, "_ext_src")
include_dir = os.path.join(ext_src_root, "include")

sources = glob.glob(os.path.join(ext_src_root, "src", "*.cpp")) + \
          glob.glob(os.path.join(ext_src_root, "src", "*.cu"))

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=sources,
            include_dirs=[include_dir],  # <--- IMPORTANT
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
