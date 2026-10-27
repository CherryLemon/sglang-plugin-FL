# SPDX-License-Identifier: Apache-2.0
"""Opt-in HY4 mRoPE orchestration compatibility for pinned SGLang 0.5.18.

Keep the existing graph-owned 3D input addresses, refresh them before replay,
and derive each draft step from its starting three-axis delta and the native
worker's current 1D position.  The native worker owns increment/reset of 1D
positions.  No 3D in-place increment is added, so capture warmups cannot drift.
No model math, cache format, or non-HY4 path changes here.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

_draft_delta = ContextVar("hy4_draft_mrope_delta", default=None)
_installation = None


def _is_hy4_mrope(model_runner):
    cfg = model_runner.model_config
    return getattr(cfg, "model_is_mrope", False) and getattr(
        cfg.hf_config, "architectures", ()
    ) == ["HYV4ForCausalLMNextN"]


def _validate_positions(mrope, positions):
    import torch

    if mrope is None:
        raise ValueError("HY4 mRoPE draft requires explicit 3D input positions")
    if mrope.shape != (3, positions.numel()):
        raise ValueError("HY4 mRoPE input must have shape [3, num_tokens]")
    if mrope.dtype != torch.int64 or positions.dtype != torch.int64:
        raise ValueError("HY4 mRoPE positions must retain int64 indices")
    if mrope.device != positions.device:
        raise ValueError("HY4 mRoPE and 1D positions must share the input device")


def _refresh_graph_mrope(runner, forward_batch):
    """D2D-copy actual axes; clear only the unused graph-owned tail.

    Copy before clearing the tail, permitting an already-canonical input view.
    No host reads of tensor values and no per-request pointer cache are used.
    """
    source = forward_batch.mrope_positions
    positions = forward_batch.positions
    destination = runner.buffers.mrope_positions
    if (
        source is None
        and forward_batch.forward_mode.is_idle()
        and positions.numel() == 0
    ):
        # Native ForwardBatch.init_new returns before computing mRoPE for idle.
        # Empty input has no MM delta; clear the existing dummy graph indices.
        destination.zero_()
        return
    _validate_positions(source, positions)
    n = positions.numel()
    if destination is None or destination.shape[0] != 3 or n > destination.shape[1]:
        raise ValueError("HY4 mRoPE graph buffer cannot hold this input")
    if destination.dtype != source.dtype or destination.device != source.device:
        raise ValueError("HY4 mRoPE graph buffer has a different index ABI")
    destination[:, :n].copy_(source)
    destination[:, n:].zero_()


def install(*, enabled=False):
    """Install once at plugin registration, before any graphs are captured.

    ``enabled`` is a registration-time boolean, not a per-layer policy lookup.
    The returned handle can undo exactly these private-subprocess wrappers.
    """
    global _installation
    if not enabled:
        return None
    if _installation is not None:
        return _installation

    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    from sglang_fl.models.hy_v4.nextn import HYV4ForCausalLMNextN

    replacements = []

    def wrap_execute(cls):
        original = cls.execute

        @wraps(original)
        def execute(self, forward_batch, *args, **kwargs):
            if _is_hy4_mrope(self.model_runner):
                _refresh_graph_mrope(self, forward_batch)
            return original(self, forward_batch, *args, **kwargs)

        replacements.append((cls, "execute", original, execute))

    wrap_execute(EAGLEDraftCudaGraphRunner)
    wrap_execute(EAGLEDraftExtendCudaGraphRunner)

    original_draft = EagleDraftWorker.draft_forward

    @wraps(original_draft)
    def draft_forward(self, forward_batch, *args, **kwargs):
        if not _is_hy4_mrope(self.draft_runner) or forward_batch.forward_mode.is_idle():
            return original_draft(self, forward_batch, *args, **kwargs)
        _validate_positions(forward_batch.mrope_positions, forward_batch.positions)
        # This tiny subtraction is captured once per draft graph invocation.
        # All three axes, including nonzero/negative MM deltas, are retained.
        delta = forward_batch.mrope_positions - forward_batch.positions.unsqueeze(0)
        token = _draft_delta.set(delta)
        try:
            return original_draft(self, forward_batch, *args, **kwargs)
        finally:
            _draft_delta.reset(token)

    replacements.append(
        (EagleDraftWorker, "draft_forward", original_draft, draft_forward)
    )
    original_nextn = HYV4ForCausalLMNextN.forward

    @wraps(original_nextn)
    def nextn_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        delta = _draft_delta.get()
        if (
            delta is None
            or not self.is_mrope_enabled
            or not forward_batch.forward_mode.is_decode()
        ):
            return original_nextn(
                self, input_ids, positions, forward_batch, *args, **kwargs
            )
        if delta.shape != (3, positions.numel()):
            raise ValueError("HY4 mRoPE draft scope does not match model input width")
        previous = forward_batch.mrope_positions
        forward_batch.mrope_positions = positions.unsqueeze(0) + delta
        try:
            return original_nextn(
                self, input_ids, positions, forward_batch, *args, **kwargs
            )
        finally:
            forward_batch.mrope_positions = previous

    replacements.append(
        (HYV4ForCausalLMNextN, "forward", original_nextn, nextn_forward)
    )

    class Installation:
        def close(self):
            global _installation
            for cls, name, original, patched in reversed(replacements):
                if getattr(cls, name) is not patched:
                    raise RuntimeError("HY4 mRoPE wrapper changed after installation")
                setattr(cls, name, original)
            _installation = None

    for cls, name, original, patched in replacements:
        setattr(cls, name, patched)
    _installation = Installation()
    return _installation


def draft_scope_is_clear():
    """A CPU/component diagnostic, never part of the serving hot path."""
    return _draft_delta.get() is None
