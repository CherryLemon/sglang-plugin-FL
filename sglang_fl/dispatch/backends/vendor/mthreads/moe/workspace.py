# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned M16K scratch, reserved before KV sizing and capture."""

from __future__ import annotations
import logging
from typing import Any, List, Optional
import torch
from ..env import enabled as _enabled, is_s5000
from sglang.srt.utils import is_musa

logger = logging.getLogger(__name__)
_is_musa = is_musa()
_MUSA_M16K_MOE_DOWN_WORKSPACE_ENV = "SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE"
_MUSA_M16K_MOE_DOWN_WORKSPACE = None


def maybe_preallocate_musa_m16k_moe_down_workspace(
    model_config: Any,
    device: Any,
    *,
    tp_size: int,
    moe_ep_size: int,
    is_draft_worker: bool,
) -> bool:
    """Reserve the exact M16K routed-down output before sizing the KV pool.

    The persistent allocation follows the same lifetime ordering used by
    upstream workspace managers: account for process-lifetime scratch before
    the large KV backing allocation, then reuse a fixed-address tensor across
    sequential MoE layers and graph replays. The S5000/model/TP2/BF16 guards
    scope the default allocation; an explicit environment override disables it.
    """
    global _MUSA_M16K_MOE_DOWN_WORKSPACE

    if not _enabled(_MUSA_M16K_MOE_DOWN_WORKSPACE_ENV, default=True):
        return False

    text_config = getattr(model_config, "hf_text_config", None)
    rope_config = getattr(text_config, "rope_parameters", None) or getattr(
        text_config, "rope_scaling", None
    )
    mrope_section = (
        rope_config.get("mrope_section") if isinstance(rope_config, dict) else None
    )
    matches = (
        _is_musa
        and is_s5000(device)
        and not is_draft_worker
        and tp_size == 2
        and moe_ep_size == 1
        and getattr(model_config, "dtype", None) == torch.bfloat16
        and getattr(model_config, "vocab_size", None) == 248320
        and getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
        and getattr(text_config, "hidden_size", None) == 2048
        and getattr(text_config, "num_hidden_layers", None) == 40
        and getattr(text_config, "num_experts", None) == 256
        and getattr(text_config, "num_experts_per_tok", None) == 8
        and getattr(text_config, "moe_intermediate_size", None) == 512
        and getattr(text_config, "shared_expert_intermediate_size", None) == 512
        and tuple(mrope_section or ()) == (11, 11, 10)
    )
    if not matches:
        logger.warning(
            "%s ignored because the loaded model/runtime is outside the exact "
            "Qwen3.6 V248320 MRoPE TP2 BF16 contract",
            _MUSA_M16K_MOE_DOWN_WORKSPACE_ENV,
        )
        return False

    if _MUSA_M16K_MOE_DOWN_WORKSPACE is None:
        _MUSA_M16K_MOE_DOWN_WORKSPACE = torch.empty(
            (16384, 8, 2048),
            dtype=torch.bfloat16,
            device=device,
        )
        logger.info(
            "Pre-allocated exact M16K routed-MoE down workspace before KV pool: "
            "bytes=%s",
            _MUSA_M16K_MOE_DOWN_WORKSPACE.numel()
            * _MUSA_M16K_MOE_DOWN_WORKSPACE.element_size(),
        )
    return True


def _get_musa_m16k_moe_down_workspace(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fused_swiglu_epilogue: bool,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    activation: str,
    is_gated: bool,
    no_combine: bool,
    inplace: bool,
    apply_router_weight_on_input: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    filter_expert: bool,
    hooks: Optional[Any],
) -> Optional[torch.Tensor]:
    """Return the pre-reserved tensor only for the proven allocation shape."""
    workspace = _MUSA_M16K_MOE_DOWN_WORKSPACE
    if (
        workspace is None
        or hidden_states.ndim != 2
        or hidden_states.shape[0] != 16384
        or hidden_states.shape[1] != 2048
    ):
        return None

    tensors = (hidden_states, w1, w2, topk_weights, topk_ids, workspace)
    if not (
        _is_musa
        and tuple(w1.shape) == (256, 512, 2048)
        and tuple(w2.shape) == (256, 2048, 256)
        and tuple(topk_weights.shape) == (16384, 8)
        and tuple(topk_ids.shape) == (16384, 8)
        and tuple(workspace.shape) == (16384, 8, 2048)
        and hidden_states.dtype == torch.bfloat16
        and w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and topk_ids.dtype == torch.int32
        and workspace.dtype == torch.bfloat16
        and all(tensor.device == hidden_states.device for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and b1 is None
        and b2 is None
        and use_fused_swiglu_epilogue
        and not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        and not per_channel_quant
        and all(
            value is None
            for value in (
                w1_scale,
                w2_scale,
                w1_zp,
                w2_zp,
                a1_scale,
                a2_scale,
                block_shape,
            )
        )
        and activation == "silu"
        and is_gated
        and not no_combine
        and inplace
        and not apply_router_weight_on_input
        and routed_scaling_factor in (None, 1.0)
        and gemm1_alpha is None
        and gemm1_limit is None
        and not filter_expert
        and hooks is None
    ):
        return None
    return workspace
