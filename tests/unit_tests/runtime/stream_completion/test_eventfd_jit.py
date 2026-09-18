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

"""Tests for the vendor-parameterized JIT build command."""

import subprocess
from pathlib import Path

from sglang_fl.runtime.stream_completion import jit
from sglang_fl.runtime.stream_completion.vendors import VENDOR_SPECS


def _fake_compiler(captured):
    def _run(command, check, capture_output, text):
        captured.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(command, 0)

    return _run


def test_build_command_is_vendor_parameterized(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_FL_EVENTFD_COMPLETION_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    captured = []
    monkeypatch.setattr(jit.subprocess, "run", _fake_compiler(captured))

    output = jit.build_completion_library(VENDOR_SPECS["cuda"])

    assert output.is_file()
    command = captured[0]
    assert "-DDEVICE_RUNTIME_HEADER=<cuda_runtime_api.h>" in command
    assert "-DSTREAM_TYPE=cudaStream_t" in command
    assert "-DLAUNCH_HOST_FUNC=cudaLaunchHostFunc" in command
    assert "-lcudart" in command
    assert "-L" + str(tmp_path / "cuda" / "lib64") in command


def test_build_result_is_cached_per_vendor(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_FL_EVENTFD_COMPLETION_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MUSA_HOME", str(tmp_path / "musa"))
    captured = []
    monkeypatch.setattr(jit.subprocess, "run", _fake_compiler(captured))

    first = jit.build_completion_library(VENDOR_SPECS["musa"])
    second = jit.build_completion_library(VENDOR_SPECS["musa"])

    assert first == second
    assert len(captured) == 1
