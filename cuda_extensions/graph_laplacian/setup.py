"""
setup.py for Graph Laplacian CUDA extension

Supports both Windows and Linux compilation.

Installation:
    python setup.py install

Or for development:
    pip install -e .
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import platform

# Determine platform-specific flags
system = platform.system()

if system == 'Windows':
    extra_compile_args = {
        'cxx': ['/O2', '/std:c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
            '-gencode=arch=compute_80,code=sm_80',  # RTX 30xx
            '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx (Ti)
            '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
        ]
    }
elif system == 'Linux':
    extra_compile_args = {
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_89,code=sm_89',
        ]
    }
else:
    raise RuntimeError(f"Unsupported platform: {system}")

setup(
    name='graph_laplacian_cuda',
    ext_modules=[
        CUDAExtension(
            name='graph_laplacian_cuda',
            sources=[
                'graph_laplacian_cuda.cpp',
                'graph_laplacian_kernel.cu',
            ],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
