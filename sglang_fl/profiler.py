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

"""MUSA profiling integration for SGLang.

SGLang's public ``/start_profile`` endpoint is platform neutral, but the
implementation in v0.5.11 assumes CUDA in two places:

* ``GPU`` is translated to ``torch.profiler.ProfilerActivity.CUDA``.
* ``CUDA_PROFILER`` calls ``cudaProfilerStart``/``cudaProfilerStop``.

TorchMUSA exposes a PrivateUse1/MUSA profiler activity and the MUSA runtime
provides the equivalent ``musaProfilerStart``/``musaProfilerStop`` markers.
This module patches those two implementation details without changing the
HTTP API or SGLang core.
"""

from __future__ import annotations

import ctypes
import logging
from functools import wraps
from typing import Iterable

import torch

logger = logging.getLogger(__name__)

# ``CUDA_PROFILER`` keeps existing SGLang profiling clients working on MUSA.
# The MUSA-specific names make new configurations self-documenting.
MUSA_API_ACTIVITIES = frozenset({"CUDA_PROFILER", "MUSA_PROFILER", "MSYS"})
MUSA_TORCH_ACTIVITIES = frozenset({"GPU", "MUSA"})


class MusaProfilerApi:
    """Call MUSA profiler range markers through the runtime shared library."""

    def __init__(self) -> None:
        self._library = None

    def _load_library(self):
        if self._library is None:
            try:
                library = ctypes.CDLL("libmusart.so")
            except OSError as exc:
                raise RuntimeError(
                    "Cannot load libmusart.so; install the MUSA runtime and expose "
                    "its library directory to the SGLang worker process."
                ) from exc

            for name in ("musaProfilerStart", "musaProfilerStop"):
                function = getattr(library, name)
                function.argtypes = []
                function.restype = ctypes.c_int
            library.musaGetLastError.argtypes = []
            library.musaGetLastError.restype = ctypes.c_int
            self._library = library
        return self._library

    def _call(self, name: str) -> int:
        library = self._load_library()
        result = int(getattr(library, name)())

        # MUSA 4.3 may return musaErrorNotSupported (801) even though msys has
        # observed and acted on the range marker. Clear the sticky runtime
        # error immediately; otherwise the next TorchMUSA kernel reports the
        # profiler error as if the kernel itself had failed.
        if result != 0:
            cleared = int(library.musaGetLastError())
            logger.warning(
                "%s returned MUSA error %d (cleared error %d); Moore Perf may "
                "still have accepted the capture-range marker",
                name,
                result,
                cleared,
            )
        return result

    def start(self) -> int:
        return self._call("musaProfilerStart")

    def stop(self) -> int:
        return self._call("musaProfilerStop")


_MUSA_PROFILER_API = MusaProfilerApi()
_patches_applied = False


def _has_activity(activities: Iterable[str], choices: frozenset[str]) -> bool:
    return not choices.isdisjoint(activities)


def _torch_musa_activity():
    activity = getattr(torch.profiler.ProfilerActivity, "MUSA", None)
    if activity is None:
        # TorchMUSA registers PrivateUse1 as MUSA on supported releases. Keep a
        # fallback for builds that expose only the generic enum name.
        activity = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
    if activity is None:
        raise RuntimeError(
            "This PyTorch/TorchMUSA build does not expose ProfilerActivity.MUSA"
        )
    return activity


def _make_torch_profiler(activities, with_stack, record_shapes):
    torch_activities = []
    if "CPU" in activities:
        torch_activities.append(torch.profiler.ProfilerActivity.CPU)
    if _has_activity(activities, MUSA_TORCH_ACTIVITIES):
        torch_activities.append(_torch_musa_activity())

    return torch.profiler.profile(
        activities=torch_activities,
        with_stack=with_stack if with_stack is not None else True,
        record_shapes=record_shapes if record_shapes is not None else False,
    )


def _is_first_rank_in_node(scheduler) -> bool:
    from sglang.srt.server_args import get_global_server_args

    return scheduler.gpu_id == get_global_server_args().base_gpu_id


def _patch_legacy_scheduler_profiler() -> None:
    """Patch the default (SGLANG_PROFILE_V2=0) profiler implementation."""

    from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin

    _wrap_legacy_scheduler_profiler(SchedulerProfilerMixin)


def _wrap_legacy_scheduler_profiler(profiler_mixin) -> None:
    """Wrap a SchedulerProfilerMixin class (split out for isolated tests)."""

    original_start = profiler_mixin.start_profile
    original_stop = profiler_mixin.stop_profile

    @wraps(original_start)
    def start_profile_with_musa(self, stage=None):
        activities = list(self.profiler_activities or [])
        use_musa_torch = _has_activity(activities, MUSA_TORCH_ACTIVITIES)
        use_musa_api = _has_activity(activities, MUSA_API_ACTIVITIES)

        if not use_musa_torch and not use_musa_api:
            return original_start(self, stage)

        # Prevent the CUDA-specific branches in SGLang core from running. If
        # GPU profiling is requested, CPU and MUSA must be created as one
        # torch.profiler session rather than as two independent sessions.
        filtered = list(activities)
        if use_musa_torch:
            filtered = [
                activity
                for activity in filtered
                if activity not in MUSA_TORCH_ACTIVITIES and activity != "CPU"
            ]
        if use_musa_api:
            filtered = [
                activity for activity in filtered if activity not in MUSA_API_ACTIVITIES
            ]

        self.profiler_activities = filtered
        try:
            result = original_start(self, stage)
        finally:
            self.profiler_activities = activities

        api_started = False
        try:
            if use_musa_api and _is_first_rank_in_node(self):
                _MUSA_PROFILER_API.start()
                api_started = True

            if use_musa_torch:
                self.torch_profiler = _make_torch_profiler(
                    activities,
                    self.torch_profiler_with_stack,
                    self.torch_profiler_record_shapes,
                )
                self.torch_profiler.start()

            self.profile_in_progress = True
            return result
        except Exception:
            if api_started:
                _MUSA_PROFILER_API.stop()
            raise

    @wraps(original_stop)
    def stop_profile_with_musa(self, stage=None):
        activities = list(self.profiler_activities or [])
        use_musa_api = _has_activity(activities, MUSA_API_ACTIVITIES)
        api_error = None

        # Emit the stop marker before SGLang spends time exporting and merging
        # torch-profiler traces.
        if use_musa_api and self.profile_in_progress and _is_first_rank_in_node(self):
            try:
                _MUSA_PROFILER_API.stop()
            except Exception as exc:  # still let SGLang stop its other profilers
                api_error = exc

        self.profiler_activities = [
            activity for activity in activities if activity not in MUSA_API_ACTIVITIES
        ]
        try:
            result = original_stop(self, stage)
        finally:
            self.profiler_activities = activities

        if api_error is not None:
            raise api_error
        return result

    profiler_mixin.start_profile = start_profile_with_musa
    profiler_mixin.stop_profile = stop_profile_with_musa


def _patch_profile_v2() -> None:
    """Patch stage-based profiling used when SGLANG_PROFILE_V2=1."""

    from sglang.srt.utils.profile_utils import (
        _ProfilerBase,
        _ProfilerCudart,
        _ProfilerTorch,
    )

    _wrap_profile_v2(_ProfilerBase, _ProfilerTorch, _ProfilerCudart)


def _wrap_profile_v2(profiler_base, profiler_torch, profiler_cudart) -> None:
    """Wrap profile-v2 classes (split out for isolated tests)."""

    original_create = profiler_base.create
    original_torch_start = profiler_torch.start

    def create(activities, with_stack, record_shapes, **kwargs):
        normalized = []
        for activity in activities:
            if activity == "MUSA":
                activity = "GPU"
            elif activity in MUSA_API_ACTIVITIES:
                activity = "CUDA_PROFILER"
            if activity not in normalized:
                normalized.append(activity)
        return original_create(normalized, with_stack, record_shapes, **kwargs)

    @wraps(original_torch_start)
    def torch_start_with_musa(self):
        if not _has_activity(self.activities, MUSA_TORCH_ACTIVITIES):
            return original_torch_start(self)
        self.torch_profiler = _make_torch_profiler(
            self.activities,
            self.with_stack,
            self.record_shapes,
        )
        self.torch_profiler.start()

    def musa_api_start(self):
        if self.first_rank_in_node:
            logger.info("Call musaProfilerStart")
            _MUSA_PROFILER_API.start()

    def musa_api_stop(self):
        if self.first_rank_in_node:
            logger.info("Call musaProfilerStop")
            _MUSA_PROFILER_API.stop()

    profiler_base.create = staticmethod(create)
    profiler_torch.start = torch_start_with_musa
    profiler_cudart.start = musa_api_start
    profiler_cudart.stop = musa_api_stop


def apply_musa_profiler_patches() -> None:
    """Install MUSA profiler support once in the current SGLang process."""

    global _patches_applied
    if _patches_applied:
        return

    _patch_legacy_scheduler_profiler()
    _patch_profile_v2()
    _patches_applied = True
    logger.info(
        "MUSA profiling patches applied: GPU->MUSA and "
        "MSYS/MUSA_PROFILER/CUDA_PROFILER capture markers enabled"
    )
