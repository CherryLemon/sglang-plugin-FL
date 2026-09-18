# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-type registry describing how to enqueue a host completion callback.

Each accelerator SDK exposes the same primitive under a different name:

    MUSA : ``musaLaunchHostFunc(musaStream_t, void(*)(void*), void*)``
    CUDA : ``cudaLaunchHostFunc(cudaStream_t, void(*)(void*), void*)``
    HIP  : ``hipLaunchHostFunc(hipStream_t, void(*)(void*), void*)``

The signatures are identical, so a single shim source can be parameterised at
compile time with the header, stream type and symbol belonging to the runtime
that is actually present.  Nothing here imports a vendor module; the failing
device simply has no spec and therefore no completion support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VendorSpec:
    """Describes one accelerator runtime for the stream-completion shim."""

    device_type: str
    stream_type: str
    launch_host_func: str
    runtime_header: str
    sdk_root_env: tuple[str, ...]
    default_sdk_root: str
    runtime_libs: tuple[str, ...]
    stream_attrs: tuple[str, ...]
    include_subdir: str = "include"
    lib_subdir: str = "lib"


VENDOR_SPECS: dict[str, VendorSpec] = {
    "musa": VendorSpec(
        device_type="musa",
        stream_type="musaStream_t",
        launch_host_func="musaLaunchHostFunc",
        runtime_header="musa_runtime_api.h",
        sdk_root_env=("MUSA_HOME",),
        default_sdk_root="/usr/local/musa",
        runtime_libs=("musart",),
        stream_attrs=("musa_stream",),
    ),
    "cuda": VendorSpec(
        device_type="cuda",
        stream_type="cudaStream_t",
        launch_host_func="cudaLaunchHostFunc",
        runtime_header="cuda_runtime_api.h",
        sdk_root_env=("CUDA_HOME", "CUDA_PATH"),
        default_sdk_root="/usr/local/cuda",
        runtime_libs=("cudart",),
        stream_attrs=("cuda_stream",),
        lib_subdir="lib64",
    ),
    "hip": VendorSpec(
        device_type="hip",
        stream_type="hipStream_t",
        launch_host_func="hipLaunchHostFunc",
        runtime_header="hip_runtime_api.h",
        sdk_root_env=("ROCM_PATH", "HIP_PATH"),
        default_sdk_root="/opt/rocm",
        runtime_libs=("amdhip64",),
        stream_attrs=("hip_stream",),
    ),
}


def get_vendor_spec(device_type: str) -> Optional[VendorSpec]:
    """Return the spec for a torch device namespace, or ``None`` if unsupported."""
    return VENDOR_SPECS.get(device_type)


def stream_pointer(stream, spec: VendorSpec) -> int:
    """Extract the raw stream handle from a torch stream for ``spec``."""
    for attr in spec.stream_attrs:
        value = getattr(stream, attr, None)
        if value:
            return int(value)
    raise AttributeError(
        f"{type(stream).__name__} exposes none of {spec.stream_attrs!r}"
    )


__all__ = ["VendorSpec", "VENDOR_SPECS", "get_vendor_spec", "stream_pointer"]
