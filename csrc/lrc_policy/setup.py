"""Build only the optional CPU LRC extension, without Torch/CANN."""

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="vllm-ascend-lrc",
    version="0.3.0",
    py_modules=[],
    ext_modules=[
        Pybind11Extension(
            "_vllm_ascend_lrc",
            ["lrc.cpp"],
            depends=["load_submit.h", "native_update.h", "native_task.h", "native_profile.h"],
            libraries=["dl"] if sys.platform.startswith("linux") else [],
            cxx_std=17,
            extra_compile_args=["/O2", "/fp:strict"] if os.name == "nt" else ["-O3", "-ffp-contract=off", "-pthread"],
            extra_link_args=[] if os.name == "nt" else ["-pthread"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
)
