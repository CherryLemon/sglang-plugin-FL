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

"""Bounded accelerator stream completion using ``*LaunchHostFunc`` + eventfd.

Instead of blocking on the accelerator Event recorded by SGLang, a native host
callback is enqueued on the stream; when the stream reaches the callback it
writes to a Linux eventfd that the waiting thread reads.  The original Event is
retained behind a proxy and is the correctness fallback for build, enqueue,
pool-exhaustion and wait failures.

The capability is vendor-neutral: the device namespace selects a
:class:`~sglang_fl.runtime.stream_completion.vendors.VendorSpec` and the shim is
compiled against that runtime on first use.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import Any, Optional

import torch

from .jit import build_completion_library
from .vendors import VendorSpec, get_vendor_spec, stream_pointer

logger = logging.getLogger(__name__)

_DEFAULT_CAPACITY = 64
_POOL: Optional["_StreamCompletionPool"] | bool = None
_POOL_LOCK = threading.Lock()
_ENQUEUE_CACHE: dict[str, Any] = {}
_ENQUEUE_LOCK = threading.Lock()


def _load_enqueue(spec: VendorSpec):
    """Load ``enqueue_eventfd_completion`` for ``spec``, compiling once."""
    cached = _ENQUEUE_CACHE.get(spec.device_type)
    if cached is not None:
        return cached
    with _ENQUEUE_LOCK:
        cached = _ENQUEUE_CACHE.get(spec.device_type)
        if cached is None:
            library = ctypes.CDLL(str(build_completion_library(spec)))
            enqueue = library.enqueue_eventfd_completion
            enqueue.argtypes = [ctypes.c_uint64, ctypes.c_int]
            enqueue.restype = ctypes.c_int
            # Keep the library alive until every queued host callback has run.
            enqueue._eventfd_library = library
            _ENQUEUE_CACHE[spec.device_type] = enqueue
            cached = enqueue
    return cached


class _StreamCompletion:
    def __init__(self, pool: "_StreamCompletionPool", slot: int):
        self._pool = pool
        self._slot = slot
        self._lock = threading.Lock()
        self._done = False

    def wait(self) -> None:
        # Make repeated synchronize calls safe.  CPython releases the GIL
        # around eventfd_read, while the native stream callback does not need
        # the GIL to signal completion.
        with self._lock:
            if self._done:
                return
            try:
                payload = os.eventfd_read(self._pool.event_fd(self._slot))
                if payload != 1:
                    raise RuntimeError(
                        f"Invalid completion eventfd payload: {payload}"
                    )
            except Exception:
                # Do not recycle a descriptor if its callback state is
                # unknown.  The proxy will synchronize the original Event;
                # leaking one slot is safer than delivering a late signal to
                # a later result.
                self._pool.abandon()
                self._done = True
                raise
            self._done = True
            self._pool.release(self._slot)


class _StreamCompletionPool:
    def __init__(self, spec: VendorSpec, capacity: int):
        if capacity <= 0:
            raise ValueError("eventfd completion pool capacity must be positive")
        if not hasattr(os, "eventfd"):
            raise RuntimeError("Linux eventfd is unavailable")
        enqueue = _load_enqueue(spec)
        event_fds: list[int] = []
        try:
            for _ in range(capacity):
                event_fds.append(os.eventfd(0, os.EFD_CLOEXEC))
        except Exception:
            for event_fd in event_fds:
                os.close(event_fd)
            raise
        self._spec = spec
        self._enqueue = enqueue
        self._event_fds = event_fds
        self._available = list(range(capacity - 1, -1, -1))
        self._lock = threading.Lock()
        self._supported = True
        self._exhaustion_logged = False

    def event_fd(self, slot: int) -> int:
        return self._event_fds[slot]

    def enqueue(self, stream: Any) -> Optional[_StreamCompletion]:
        with self._lock:
            if not self._supported or not self._available:
                if self._supported and not self._exhaustion_logged:
                    logger.warning(
                        "eventfd completion pool exhausted; using Event fallback"
                    )
                    self._exhaustion_logged = True
                return None
            slot = self._available.pop()
        try:
            stream_ptr = stream_pointer(stream, self._spec)
            status = self._enqueue(stream_ptr, self._event_fds[slot])
            if status != 0:
                raise RuntimeError(f"{self._spec.launch_host_func} returned {status}")
        except Exception:
            with self._lock:
                self._supported = False
                self._available.append(slot)
            logger.exception(
                "Failed to enqueue eventfd completion; using Event fallback"
            )
            return None
        return _StreamCompletion(self, slot)

    def release(self, slot: int) -> None:
        with self._lock:
            self._available.append(slot)
            self._exhaustion_logged = False

    def abandon(self) -> None:
        with self._lock:
            self._supported = False


def _pool_capacity() -> int:
    raw = os.environ.get(
        "SGLANG_FL_EVENTFD_COMPLETION_POOL_SIZE"
    ) or os.environ.get("SGLANG_MUSA_EVENTFD_COMPLETION_POOL_SIZE")
    return int(raw) if raw else _DEFAULT_CAPACITY


def _get_pool(device: torch.device) -> Optional[_StreamCompletionPool]:
    global _POOL
    spec = get_vendor_spec(device.type) if device is not None else None
    if spec is None or _POOL is False:
        return None
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                try:
                    _POOL = _StreamCompletionPool(spec, _pool_capacity())
                except Exception:
                    _POOL = False
                    logger.exception(
                        "Failed to initialize eventfd completion; "
                        "using Event fallback"
                    )
    return _POOL if isinstance(_POOL, _StreamCompletionPool) else None


def try_enqueue_completion(device: torch.device):
    pool = _get_pool(device)
    if pool is None:
        return None
    stream = torch.get_device_module(device).current_stream(device)
    return pool.enqueue(stream)


class CompletionEventProxy:
    """Event-compatible synchronization proxy with an exact Event fallback."""

    def __init__(self, completion: _StreamCompletion, fallback_event: Any):
        self._completion = completion
        self._fallback_event = fallback_event
        self._lock = threading.Lock()
        self._done = False

    def synchronize(self) -> None:
        with self._lock:
            if self._done:
                return
            try:
                self._completion.wait()
            except Exception:
                logger.exception(
                    "eventfd completion wait failed; using Event fallback"
                )
                self._fallback_event.synchronize()
            self._done = True

    def __getattr__(self, name: str):
        return getattr(self._fallback_event, name)


def reset_for_test() -> None:
    """Reset lazy globals; only unit tests should call this helper."""
    global _POOL
    _POOL = None
    _ENQUEUE_CACHE.clear()


__all__ = ["CompletionEventProxy", "try_enqueue_completion"]
