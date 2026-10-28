# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Adapted from the pinned SGLang/MUSA Qwen3.6 campaign implementation.
# Upstream ancestry: vllm a6221a144af772fd1a68fe7e627935dc53e81738.
# Numerical kernels are provided through FlagGems-sglang public APIs.
"""SGLang graph registrations and guarded routed-MoE orchestration."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import torch
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as _core
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.custom_op import register_custom_op
from ..env import enabled as _enabled
from .workspace import _get_musa_m16k_moe_down_workspace

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)
_is_musa = _core._is_musa
padding_size = 0


def try_get_optimal_moe_config(*args, **kwargs):
    return _core.try_get_optimal_moe_config(*args, **kwargs)


def get_config_dtype_str(*args, **kwargs):
    return _core.get_config_dtype_str(*args, **kwargs)


def moe_align_block_size(*args, **kwargs):
    return _core.moe_align_block_size(*args, **kwargs)


def _down_moe_use_tma():
    return False


_MUSA_M4_W13_BN64_ENV = "SGLANG_MUSA_M4_W13_BN64"


@register_custom_op(
    op_name="musa_qwen36_inplace_fused_experts", mutates_args=["hidden_states"]
)
def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    fuse_swiglu_epilogue: bool = False,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        False,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
        filter_expert,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )


@register_custom_op(
    op_name="musa_qwen36_outplace_fused_experts", out_shape="hidden_states"
)
def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    fuse_swiglu_epilogue: bool = False,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        False,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    fuse_swiglu_epilogue: bool = False,
):
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )
    if moe_runner_config.inplace:
        if moe_runner_config.no_combine:
            raise ValueError("no combine + inplace makes no sense")
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            moe_runner_config.routed_scaling_factor,
            moe_runner_config.gemm1_alpha,
            moe_runner_config.gemm1_clamp_limit,
            filter_expert,
            fuse_swiglu_epilogue=fuse_swiglu_epilogue,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            no_combine=moe_runner_config.no_combine,
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            fuse_swiglu_epilogue=fuse_swiglu_epilogue,
        )


def _prepare_fused_moe_run(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]],
    fuse_swiglu_epilogue: bool = False,
):
    """Resolve config, down_config, TMA flag, and aligned expert routing ids.

    Shared by ``fused_experts_impl`` and ``pre_permute_standard_to_triton`` so
    both paths compute alignment from the same source.
    """
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None:
        padded_size = 0

    num_tokens = hidden_states.shape[0]
    E = w1.shape[0]
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    config, (down_config, _) = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        num_tokens,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.get("USE_TMA", False)
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    return (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _fused_moe_kernel_sequence(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: Dict[str, Any],
    down_config: Optional[Dict[str, Any]],
    down_moe_use_tma: bool,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
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
    hooks: Optional[Any] = None,
    fuse_swiglu_epilogue: bool = False,
) -> torch.Tensor:
    """Orchestrate public GEMMs, native activation and the existing reduction seam."""
    from flaggems_sglang import fused_moe_bf16_gemm

    num_tokens, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    width = w1.shape[1]
    fused = fuse_swiglu_epilogue and num_tokens >= 2048
    up_profile = (
        "swiglu_up"
        if fused
        else (
            "m4_up"
            if inplace and _enabled(_MUSA_M4_W13_BN64_ENV, default=True)
            else None
        )
    )
    intermediate = torch.empty(
        (num_tokens, topk, width // 2 if fused else width),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    fused_moe_bf16_gemm(
        hidden_states,
        w1,
        intermediate,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        top_k=topk,
        fuse_swiglu=fused,
        tuning_profile=up_profile,
    )
    activated = intermediate.view(-1, intermediate.shape[-1])
    if not fused:
        # Preserve the pinned vendor activation and its BF16 materialization.
        activated = _core._silu_and_mul_musa(activated)

    down = _get_musa_m16k_moe_down_workspace(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1=b1,
        b2=b2,
        use_fused_swiglu_epilogue=fused,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        apply_router_weight_on_input=False,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=False,
        hooks=None,
    )
    if down is None:
        down = torch.empty(
            (num_tokens, topk, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    fused_moe_bf16_gemm(
        activated,
        w2,
        down,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        down_config or config,
        top_k=1,
        mul_routed_weight=True,
        tuning_profile="swiglu_down" if fused else None,
    )
    output = hidden_states if inplace else torch.empty_like(hidden_states)
    reduce = (
        _core.moe_sum_reduce_torch_compile if num_tokens <= 32 else _core.moe_sum_reduce
    )
    reduce(
        down.view(*down.shape),
        output,
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
    )
    return output


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    fuse_swiglu_epilogue: bool = False,
):
    if (
        b1 is not None
        or b2 is not None
        or activation != "silu"
        or not is_gated
        or apply_router_weight_on_input
        or use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or per_channel_quant
        or no_combine
        or filter_expert
        or gemm1_alpha is not None
        or gemm1_limit is not None
        or any(
            v is not None
            for v in (w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape)
        )
    ):
        raise ValueError("unsupported guarded MUSA BF16 MoE mode")
    if (
        tuple(w1.shape) != (256, 512, 2048)
        or tuple(w2.shape) != (256, 2048, 256)
        or hidden_states.ndim != 2
        or hidden_states.shape[1] != 2048
        or topk_ids.shape != (hidden_states.shape[0], 8)
    ):
        raise ValueError("unsupported guarded MUSA BF16 MoE shape")
    if get_global_server_args().enable_fused_moe_sum_all_reduce:
        raise ValueError("atomic MoE all-reduce is outside the guarded contract")
    if (
        topk_weights.shape != topk_ids.shape
        or not all(t.is_contiguous() for t in (hidden_states, w1, w2))
        or any(t.dtype != torch.bfloat16 for t in (hidden_states, w1, w2))
    ):
        raise ValueError(
            "guarded MUSA MoE requires contiguous BF16 activations/weights"
        )

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        w1,
        w2,
        topk_ids,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )

    return _fused_moe_kernel_sequence(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        down_config,
        down_moe_use_tma,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        apply_router_weight_on_input=apply_router_weight_on_input,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        hooks=None,
        fuse_swiglu_epilogue=fuse_swiglu_epilogue,
    )
