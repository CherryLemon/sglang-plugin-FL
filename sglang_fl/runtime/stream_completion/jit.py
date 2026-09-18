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

"""Just-in-time build of the stream-completion host-callback shim.

The shim only needs libc (``eventfd_write``) plus the accelerator runtime
header; the ``*LaunchHostFunc`` symbol is referenced from that header and
resolved by the linker against the vendor runtime named in the spec.  The
result is cached on disk, keyed by vendor and source digest, and compiled under
an exclusive lock so concurrent ranks share a single artifact.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

from .vendors import VendorSpec

_SOURCE = Path(__file__).resolve().parent / "csrc" / "stream_eventfd_completion.cpp"


def _sdk_root(spec: VendorSpec) -> Path:
    for name in spec.sdk_root_env:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(spec.default_sdk_root)


def _cache_root() -> Path:
    explicit = os.environ.get(
        "SGLANG_FL_EVENTFD_COMPLETION_CACHE_DIR"
    ) or os.environ.get("SGLANG_MUSA_EVENTFD_CACHE_DIR")
    if explicit:
        return Path(explicit)
    jit_root = Path(
        os.environ.get("SGLANG_FL_JIT_CACHE_DIR")
        or os.environ.get("SGLANG_MUSA_JIT_CACHE_DIR")
        or (Path.home() / ".cache" / "sglang_fl_jit")
    )
    return jit_root / "eventfd_completion"


def build_completion_library(spec: VendorSpec) -> Path:
    """Compile (or reuse) the shim matching ``spec`` and return its path."""
    root = _sdk_root(spec)
    source_digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / (
        f"eventfd_completion_{spec.device_type}_{source_digest}.so"
    )
    lock_path = cache_root / "build.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            return output
        temporary = output.with_suffix(f".tmp.{os.getpid()}.so")
        command = [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            str(_SOURCE),
            f"-I{root / spec.include_subdir}",
            f"-DDEVICE_RUNTIME_HEADER=<{spec.runtime_header}>",
            f"-DSTREAM_TYPE={spec.stream_type}",
            f"-DLAUNCH_HOST_FUNC={spec.launch_host_func}",
            f"-L{root / spec.lib_subdir}",
            *[f"-l{lib}" for lib in spec.runtime_libs],
            "-o",
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return output


__all__ = ["build_completion_library"]
