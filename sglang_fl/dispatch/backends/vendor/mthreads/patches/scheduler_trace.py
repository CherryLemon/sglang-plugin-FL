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

"""Bounded, opt-in scheduler events for MUSA system-timeline diagnosis.

This instrumentation is intentionally separate from the performance patches.
It writes only while SGLang's profiling endpoint is active and is enabled by
``SGLANG_MUSA_SCHEDULER_TRACE_PATH``. Request identifiers are reduced to stable
16-hex digests; prompts, token IDs, and generated text are never written.

The JSONL timestamps use ``time.monotonic_ns()``. ``profile_start_enter`` and
``profile_start_exit`` bracket the MUSA profiler marker so an MSYS host-time
offset can be checked rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Optional

from .sglang_0_5_11_profiler_lifecycle import _require_sglang_0_5_11

logger = logging.getLogger(__name__)

_TRACE_PATH_ENV = "SGLANG_MUSA_SCHEDULER_TRACE_PATH"
_TRACE_FLUSH_EVERY_ENV = "SGLANG_MUSA_SCHEDULER_TRACE_FLUSH_EVERY"
_PREFILL_BURST_LIMIT_ENV = "SGLANG_MUSA_PREFILL_BURST_LIMIT"
_DECODE_STARVATION_LIMIT_MS_ENV = "SGLANG_MUSA_DECODE_STARVATION_LIMIT_MS"
_DEFAULT_FLUSH_EVERY = 64
_patches_applied = False


def _rid_digest(rid: Any) -> str:
    return hashlib.blake2s(str(rid).encode("utf-8"), digest_size=8).hexdigest()


def _mode_name(batch: Any) -> Optional[str]:
    if batch is None:
        return None
    mode = getattr(batch, "forward_mode", None)
    name = getattr(mode, "name", None)
    return str(name if name is not None else mode)


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class _SchedulerTraceState:
    def __init__(self, scheduler: Any) -> None:
        base = Path(os.environ[_TRACE_PATH_ENV])
        raw_tp_rank = getattr(scheduler, "tp_rank", None)
        raw_dp_rank = getattr(scheduler, "dp_rank", None)
        tp_rank = -1 if raw_tp_rank is None else int(raw_tp_rank)
        dp_rank = -1 if raw_dp_rank is None else int(raw_dp_rank)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.path = base.with_name(
            f"{base.name}.tp{tp_rank}.dp{dp_rank}.pid{os.getpid()}.jsonl"
        )
        self.file = self.path.open("a", encoding="utf-8")
        self.active = False
        self.event_count = 0
        self.flush_every = max(
            1,
            int(os.environ.get(_TRACE_FLUSH_EVERY_ENV, _DEFAULT_FLUSH_EVERY)),
        )
        self.last_schedule_ns: dict[str, int] = {}
        self.batch_sequence = 0
        self.current_batch_sequence: Optional[int] = None
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.prefill_burst_limit = max(
            0, int(os.environ.get(_PREFILL_BURST_LIMIT_ENV, "0"))
        )
        self.decode_starvation_limit_ms = max(
            0, int(os.environ.get(_DECODE_STARVATION_LIMIT_MS_ENV, "0"))
        )
        self.consecutive_prefill_batches = 0
        self.decode_starvation_start_ns: Optional[int] = None
        self.forced_decode_pending = False
        self.forced_decode_reason: Optional[str] = None
        self.forced_decode_count = 0

    def emit(self, event: str, *, force: bool = False, **fields: Any) -> None:
        if not (force or self.active):
            return
        row = {
            "schema": 1,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "tp_rank": self.tp_rank,
            "dp_rank": self.dp_rank,
            **fields,
        }
        self.file.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
        self.file.write("\n")
        self.event_count += 1
        if force or self.event_count % self.flush_every == 0:
            self.file.flush()


def _state(scheduler: Any) -> _SchedulerTraceState:
    state = getattr(scheduler, "_musa_scheduler_trace_state", None)
    if state is None:
        state = _SchedulerTraceState(scheduler)
        scheduler._musa_scheduler_trace_state = state
    return state


def _request_progress(reqs: Iterable[Any]) -> list[list[Any]]:
    return [
        [
            _rid_digest(getattr(req, "rid", "unknown")),
            len(getattr(req, "output_ids", ()) or ()),
        ]
        for req in reqs
    ]


def _decode_and_prefill_are_pending(scheduler: Any) -> bool:
    running_batch = getattr(scheduler, "running_batch", None)
    running_reqs = getattr(running_batch, "reqs", ()) or ()
    if not running_reqs or bool(getattr(running_batch, "is_prefill_only", False)):
        return False

    return bool(getattr(scheduler, "waiting_queue", ()) or ()) or getattr(
        scheduler, "chunked_req", None
    ) is not None


def _should_force_decode(
    scheduler: Any,
    state: _SchedulerTraceState,
    *,
    now_ns: Optional[int] = None,
) -> bool:
    if not _decode_and_prefill_are_pending(scheduler):
        state.decode_starvation_start_ns = None
        state.forced_decode_reason = None
        return False

    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if state.decode_starvation_start_ns is None:
        state.decode_starvation_start_ns = now_ns

    burst_due = (
        state.prefill_burst_limit > 0
        and state.consecutive_prefill_batches >= state.prefill_burst_limit
    )
    starvation_due = (
        state.decode_starvation_limit_ms > 0
        and now_ns - state.decode_starvation_start_ns
        >= state.decode_starvation_limit_ms * 1_000_000
    )
    if not (burst_due or starvation_due):
        state.forced_decode_reason = None
        return False

    state.forced_decode_reason = "starvation" if starvation_due else "burst"
    return True


def _summarize_schedule(scheduler: Any, batch: Any, now_ns: int) -> dict[str, Any]:
    state = _state(scheduler)
    reqs = list(getattr(batch, "reqs", ()) or ()) if batch is not None else []
    digests = [_rid_digest(getattr(req, "rid", "unknown")) for req in reqs]
    gap_pairs = []
    for digest in digests:
        previous = state.last_schedule_ns.get(digest)
        if previous is not None:
            gap_pairs.append((digest, (now_ns - previous) / 1_000_000.0))
        state.last_schedule_ns[digest] = now_ns

    gaps = [value for _, value in gap_pairs]
    gap_pairs.sort(key=lambda item: item[1], reverse=True)
    decoding_reqs = list(getattr(batch, "decoding_reqs", ()) or ()) if batch else []
    chunked_req = getattr(batch, "chunked_req", None) if batch else None
    prefill_tokens = sum(int(getattr(req, "extend_input_len", 0) or 0) for req in reqs)

    return {
        "batch_sequence": state.current_batch_sequence,
        "forward_ct": int(getattr(scheduler, "forward_ct", 0)),
        "mode": _mode_name(batch),
        "batch_size": len(reqs),
        "active_decode_batch_size": (
            len(reqs) if _mode_name(batch) == "DECODE" else len(decoding_reqs)
        ),
        "prefill_tokens": prefill_tokens,
        "waiting_queue_depth": len(getattr(scheduler, "waiting_queue", ()) or ()),
        "running_queue_depth": len(
            getattr(getattr(scheduler, "running_batch", None), "reqs", ()) or ()
        ),
        "running_batch_full": bool(
            getattr(getattr(scheduler, "running_batch", None), "batch_is_full", False)
        ),
        "chunked_request": (
            _rid_digest(getattr(chunked_req, "rid", "unknown"))
            if chunked_req is not None
            else None
        ),
        "request_progress": _request_progress(reqs),
        "reschedule_gap_ms": {
            "count": len(gaps),
            "p50": _percentile(gaps, 0.50),
            "p99": _percentile(gaps, 0.99),
            "max": max(gaps) if gaps else None,
            "gt_100ms": sum(value > 100.0 for value in gaps),
            "gt_1s": sum(value > 1000.0 for value in gaps),
            "top5": [[digest, value] for digest, value in gap_pairs[:5]],
        },
    }


def _wrap_profiler_lifecycle(profiler_mixin: type) -> None:
    if getattr(profiler_mixin, "_musa_scheduler_trace_lifecycle_patched", False):
        return

    original_start = profiler_mixin.start_profile
    original_stop = profiler_mixin.stop_profile

    @wraps(original_start)
    def start_profile_with_scheduler_trace(self, *args, **kwargs):
        state = _state(self)
        state.emit("profile_start_enter", force=True)
        try:
            result = original_start(self, *args, **kwargs)
        except Exception as exc:
            state.emit(
                "profile_start_error", force=True, error_type=type(exc).__name__
            )
            raise
        state.active = True
        state.emit("profile_start_exit", force=True)
        return result

    @wraps(original_stop)
    def stop_profile_with_scheduler_trace(self, *args, **kwargs):
        state = _state(self)
        state.emit("profile_stop_enter", force=True)
        try:
            result = original_stop(self, *args, **kwargs)
        except Exception as exc:
            state.emit(
                "profile_stop_error", force=True, error_type=type(exc).__name__
            )
            raise
        finally:
            state.active = False
        state.emit("profile_stop_exit", force=True)
        return result

    profiler_mixin.start_profile = start_profile_with_scheduler_trace
    profiler_mixin.stop_profile = stop_profile_with_scheduler_trace
    profiler_mixin._musa_scheduler_trace_lifecycle_patched = True


def _wrap_scheduler(scheduler_cls: type) -> None:
    if getattr(scheduler_cls, "_musa_scheduler_trace_patched", False):
        return

    original_get_next = scheduler_cls.get_next_batch_to_run
    original_get_new_batch_prefill = scheduler_cls.get_new_batch_prefill
    original_run_batch = scheduler_cls.run_batch
    original_process_result = scheduler_cls.process_batch_result

    @wraps(original_get_new_batch_prefill)
    def get_new_batch_prefill_with_fairness(self, *args, **kwargs):
        state = _state(self)
        state.forced_decode_pending = False
        state.forced_decode_reason = None
        if _should_force_decode(self, state):
            state.forced_decode_pending = True
            state.forced_decode_count += 1
            return None
        return original_get_new_batch_prefill(self, *args, **kwargs)

    @wraps(original_get_next)
    def get_next_batch_with_trace(self, *args, **kwargs):
        batch = original_get_next(self, *args, **kwargs)
        state = _state(self)
        mode = _mode_name(batch)
        consecutive_prefill_before = state.consecutive_prefill_batches
        if mode == "EXTEND":
            state.consecutive_prefill_batches += 1
        elif mode == "DECODE":
            state.consecutive_prefill_batches = 0
            state.decode_starvation_start_ns = None
        if state.active:
            state.batch_sequence += 1
            state.current_batch_sequence = state.batch_sequence
            now_ns = time.monotonic_ns()
            summary = _summarize_schedule(self, batch, now_ns)
            summary.update(
                prefill_burst_limit=state.prefill_burst_limit,
                decode_starvation_limit_ms=state.decode_starvation_limit_ms,
                consecutive_prefill_before=consecutive_prefill_before,
                consecutive_prefill_after=state.consecutive_prefill_batches,
                forced_decode=state.forced_decode_pending,
                forced_decode_reason=state.forced_decode_reason,
                forced_decode_count=state.forced_decode_count,
            )
            state.emit("schedule_decision", **summary)
        return batch

    @wraps(original_run_batch)
    def run_batch_with_trace(self, batch, *args, **kwargs):
        state = _state(self)
        state.emit(
            "run_batch_enter",
            batch_sequence=state.current_batch_sequence,
            mode=_mode_name(batch),
            batch_size=len(getattr(batch, "reqs", ()) or ()),
        )
        try:
            return original_run_batch(self, batch, *args, **kwargs)
        finally:
            state.emit(
                "run_batch_exit",
                batch_sequence=state.current_batch_sequence,
                mode=_mode_name(batch),
            )

    @wraps(original_process_result)
    def process_batch_result_with_trace(self, batch, result, *args, **kwargs):
        state = _state(self)
        state.emit(
            "process_result_enter",
            batch_sequence=state.current_batch_sequence,
            mode=_mode_name(batch),
            request_progress=_request_progress(getattr(batch, "reqs", ()) or ()),
        )
        try:
            return original_process_result(self, batch, result, *args, **kwargs)
        finally:
            state.emit(
                "process_result_exit",
                batch_sequence=state.current_batch_sequence,
                mode=_mode_name(batch),
                request_progress=_request_progress(
                    getattr(batch, "reqs", ()) or ()
                ),
            )

    scheduler_cls.get_next_batch_to_run = get_next_batch_with_trace
    scheduler_cls.get_new_batch_prefill = get_new_batch_prefill_with_fairness
    scheduler_cls.run_batch = run_batch_with_trace
    scheduler_cls.process_batch_result = process_batch_result_with_trace
    scheduler_cls._musa_scheduler_trace_patched = True


def apply_musa_scheduler_trace_patch() -> None:
    """Install diagnostic scheduler wrappers when the trace path is configured."""
    global _patches_applied
    if _patches_applied or not os.environ.get(_TRACE_PATH_ENV):
        return

    _require_sglang_0_5_11()
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin

    _wrap_profiler_lifecycle(SchedulerProfilerMixin)
    _wrap_scheduler(Scheduler)
    _patches_applied = True
    logger.warning(
        "MUSA scheduler trace enabled for diagnostic use only; timings from this "
        "run must not be reported as production performance"
    )
    if os.environ.get(_PREFILL_BURST_LIMIT_ENV, "0") != "0":
        logger.warning(
            "MUSA diagnostic prefill burst limit enabled: %s consecutive "
            "prefill batches",
            os.environ[_PREFILL_BURST_LIMIT_ENV],
        )
    if os.environ.get(_DECODE_STARVATION_LIMIT_MS_ENV, "0") != "0":
        logger.warning(
            "MUSA diagnostic decode starvation limit enabled: %s ms",
            os.environ[_DECODE_STARVATION_LIMIT_MS_ENV],
        )
