# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Eligibility checks for the SGLang shared-expert call bridge."""

from __future__ import annotations
import torch
from ..env import is_s5000

PRODUCT_M, PRODUCT_H = 4, 2048


def _storage_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return a conservative storage interval for fail-closed alias checks."""

    try:
        storage = tensor.untyped_storage()
        start = int(storage.data_ptr())
        return start, start + int(storage.nbytes())
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        return None


def _may_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively detect overlap, including views sharing one storage."""

    if left.device != right.device:
        return False
    left_span = _storage_span(left)
    right_span = _storage_span(right)
    if left_span is None or right_span is None:
        return True
    return max(left_span[0], right_span[0]) < min(left_span[1], right_span[1])


def candidate_guard_reason(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    gate_logits_out: torch.Tensor | None = None,
    gate_prob_out: torch.Tensor | None = None,
    enabled: bool | None = None,
    capturing: bool | None = None,
) -> str:
    """Explain why the candidate is or is not eligible.

    The reasons are intentionally conservative and stable enough for a
    harness to record.  ``enabled`` is injectable for CPU policy tests; the
    production default enables the guarded S5000 path unless explicitly disabled.
    """

    if enabled is None:
        enabled = True
    if not enabled:
        return "disabled"

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (hidden, gate_weight, shared_output)
    ):
        return "non_tensor_input"
    if hidden.ndim != 2 or shared_output.ndim != 2 or gate_weight.ndim != 2:
        return "rank_not_2"
    if hidden.shape != (PRODUCT_M, PRODUCT_H):
        return "hidden_shape_not_M4_H2048"
    if gate_weight.shape != (1, PRODUCT_H):
        return "gate_weight_shape_not_1xH"
    if shared_output.shape != (PRODUCT_M, PRODUCT_H):
        return "shared_output_shape_not_M4_H2048"
    if hidden.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        return "input_dtype_not_bfloat16"
    if shared_output.dtype != torch.bfloat16:
        return "shared_dtype_not_bfloat16"
    if hidden.device != gate_weight.device or hidden.device != shared_output.device:
        return "device_mismatch"
    if (
        not hidden.is_contiguous()
        or not gate_weight.is_contiguous()
        or not shared_output.is_contiguous()
    ):
        return "input_layout_not_contiguous"
    if tuple(hidden.stride()) != (PRODUCT_H, 1):
        return "hidden_stride_not_MajorH"
    if tuple(gate_weight.stride()) != (PRODUCT_H, 1):
        return "gate_stride_not_H_major"
    if tuple(shared_output.stride()) != (PRODUCT_H, 1):
        return "shared_stride_not_MajorH"
    if _may_alias(hidden, gate_weight):
        return "hidden_aliases_gate_weight"
    if _may_alias(hidden, shared_output):
        return "hidden_aliases_shared_output"
    if _may_alias(gate_weight, shared_output):
        return "gate_weight_aliases_shared_output"

    if out is not None:
        if not isinstance(out, torch.Tensor):
            return "output_not_tensor"
        if out.shape != (PRODUCT_M, PRODUCT_H):
            return "output_shape_not_M4_H2048"
        if out.dtype != torch.bfloat16:
            return "output_dtype_not_bfloat16"
        if out.device != hidden.device:
            return "output_device_mismatch"
        if not out.is_contiguous() or tuple(out.stride()) != (PRODUCT_H, 1):
            return "output_layout_not_contiguous"
        if any(
            _may_alias(out, tensor) for tensor in (hidden, gate_weight, shared_output)
        ):
            return "output_aliases_input"
    elif capturing is True:
        return "output_required_during_capture"

    debug_outputs = (
        ("gate_logits_debug", gate_logits_out),
        ("gate_prob_debug", gate_prob_out),
    )
    for debug_name, debug_output in debug_outputs:
        if debug_output is None:
            continue
        if not isinstance(debug_output, torch.Tensor):
            return f"{debug_name}_not_tensor"
        if debug_output.shape != (PRODUCT_M, 1):
            return f"{debug_name}_shape_not_Mx1"
        if debug_output.dtype != torch.bfloat16:
            return f"{debug_name}_dtype_not_bfloat16"
        if debug_output.device != hidden.device:
            return f"{debug_name}_device_mismatch"
        if not debug_output.is_contiguous() or tuple(debug_output.stride()) != (1, 1):
            return f"{debug_name}_layout_not_contiguous"
        if any(
            _may_alias(debug_output, tensor)
            for tensor in (hidden, gate_weight, shared_output, out)
            if tensor is not None
        ):
            return f"{debug_name}_aliases_input_or_output"
    if (
        gate_logits_out is not None
        and gate_prob_out is not None
        and _may_alias(gate_logits_out, gate_prob_out)
    ):
        return "debug_outputs_alias"

    if hidden.device.type != "musa":
        return "device_not_musa"
    if not is_s5000(hidden.device):
        return "device_not_s5000"
    return "eligible"
