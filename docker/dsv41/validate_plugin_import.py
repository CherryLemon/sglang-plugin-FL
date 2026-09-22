"""CPU-only import/registration smoke test with explicit device/driver doubles.

FlagGems initializes autotuners at import. These doubles allow checking the
Torch/Triton Python API and plugin registration without exposing a GPU. This
does not validate device detection, allocation, kernel launches or collectives.
Run in a fresh process/container with GEMS_VENDOR=nvidia and SGLANG_PLUGINS=.
"""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from triton.backends.compiler import GPUTarget
from triton.runtime import driver


def forbidden_benchmark(*args, **kwargs):
    raise AssertionError("The import smoke test must not run kernels or benchmarks")


assert not torch.cuda.is_available(), "Run this import test in a CPU-only container"
properties = SimpleNamespace(
    name="NVIDIA H100 80GB HBM3",
    major=9,
    minor=0,
    multi_processor_count=132,
    total_memory=80 * 1024**3,
    max_shared_memory_per_multiprocessor=228 * 1024,
)
stub = SimpleNamespace(
    get_benchmarker=lambda: forbidden_benchmark,
    get_current_target=lambda: GPUTarget("cuda", 90, 32),
    get_current_device=lambda: 0,
)
with (
    patch("torch.cuda.get_device_properties", return_value=properties),
    patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    patch("torch.cuda.get_device_name", return_value=properties.name),
    patch("torch.cuda.current_device", return_value=0),
    patch.object(type(driver), "active", new_callable=PropertyMock, return_value=stub),
):
    import flag_gems
    import sglang_fl
    from sglang_fl.dsv41 import backend

    sglang_fl.load_plugin()
    assert sglang_fl.is_plugin_active() and backend._backend is not None
    assert backend._backend.capabilities.device == "cpu"
    print(
        "FlagGems import and plugin registration passed with CPU doubles:",
        flag_gems.__version__,
    )
