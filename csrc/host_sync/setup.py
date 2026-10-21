"""Build the optional Host synchronization extension without Torch/CANN."""

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="vllm-ascend-host-sync",
    version="0.2.0",
    py_modules=["vllm_ascend_host_sync"],
    ext_modules=[
        Pybind11Extension(
            "_vllm_ascend_host_sync",
            ["host_sync.cpp"],
            depends=["host_memory.h", "protocol.h"],
            cxx_std=17,
            extra_compile_args=["/O2"] if os.name == "nt" else ["-O2", "-pthread"],
            extra_link_args=[] if os.name == "nt" else ["-pthread"],
            libraries=["dl"] if sys.platform.startswith("linux") else [],
        )
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
)
