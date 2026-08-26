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

"""MUSA FLA implementations and safe fallbacks."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_MATE_GDN_ENV = "SGLANG_MUSA_MATE_GDN"
_MATE_GDN_REQUIRED_PARAMETERS = {
    "q",
    "k",
    "v",
    "state",
    "A_log",
    "a",
    "dt_bias",
    "b",
    "state_layout",
    "state_indices",
    "scale",
    "output",
    "disable_state_update",
    "use_qk_l2norm",
}
_mate_gdn_decode: Optional[Callable] = None
_mate_gdn_import_attempted = False
_mate_gdn_match_logged = False


def _mate_gdn_enabled() -> bool:
    return os.environ.get(_MATE_GDN_ENV, "auto").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def _load_mate_gdn_decode() -> Optional[Callable]:
    """Load a compatible MATE GDN API once, without making MATE mandatory."""
    global _mate_gdn_decode, _mate_gdn_import_attempted
    if _mate_gdn_import_attempted:
        return _mate_gdn_decode

    _mate_gdn_import_attempted = True
    try:
        from mate.gdn_decode import gated_delta_rule_decode

        parameters = set(inspect.signature(gated_delta_rule_decode).parameters)
        missing = _MATE_GDN_REQUIRED_PARAMETERS - parameters
        if missing:
            logger.warning(
                "MATE GDN API is missing required parameters %s; using SGLang fallback",
                sorted(missing),
            )
            return None
        _mate_gdn_decode = gated_delta_rule_decode
    except (ImportError, OSError, TypeError, ValueError):
        logger.info("Compatible MATE GDN decode is unavailable; using SGLang fallback")
    return _mate_gdn_decode


def _is_s5000(device: torch.device) -> bool:
    try:
        return "S5000" in str(torch.musa.get_device_name(device)).upper()
    except (AttributeError, RuntimeError, TypeError):
        return False


def _supports_mate_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
) -> bool:
    """Match the MATE VK/FP32-state T=1 contract validated on MP31."""
    if not _mate_gdn_enabled() or not _is_s5000(mixed_qkv.device):
        return False
    if (
        mixed_qkv.ndim != 2
        or a.ndim != 2
        or b.ndim != 2
        or A_log.ndim != 1
        or dt_bias.ndim != 1
        or initial_state.ndim != 4
        or out.ndim != 4
        or ssm_state_indices.ndim != 1
    ):
        return False

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qk_dim = mixed_qkv.shape[1] - HV * V
    if qk_dim <= 0 or qk_dim % 2 or (qk_dim // 2) % K:
        return False
    H = (qk_dim // 2) // K

    tensors = (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    return (
        K == 128
        and V == K
        and H > 0
        and HV % H == 0
        and a.shape == (B, HV)
        and b.shape == (B, HV)
        and A_log.numel() == HV
        and dt_bias.numel() == HV
        and out.shape == (B, 1, HV, V)
        and ssm_state_indices.shape == (B,)
        and mixed_qkv.dtype in (torch.float16, torch.bfloat16)
        and a.dtype == mixed_qkv.dtype
        and b.dtype == mixed_qkv.dtype
        and A_log.dtype == torch.float32
        and dt_bias.dtype in (torch.float32, torch.bfloat16)
        and initial_state.dtype == torch.float32
        and out.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and ssm_state_indices.dtype in (torch.int32, torch.int64)
        and mixed_qkv.stride(-1) == 1
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        and initial_state.is_contiguous()
        and out.is_contiguous()
        and all(t.device == mixed_qkv.device for t in tensors)
    )


def _original(fn_name: str):
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(fn_name)
    if fn is None:
        raise RuntimeError(
            f"FLA original '{fn_name}' not available — fla_patch not applied yet"
        )
    return fn


def chunk_gated_delta_rule_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """chunk_gated_delta_rule — not yet implemented on MUSA. Current behavior: SGLang's original triton kernels."""
    return _original("chunk_gated_delta_rule")(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """fused_recurrent_gated_delta_rule — not yet implemented on MUSA. Current behavior: SGLang's original triton kernels."""
    return _original("fused_recurrent_gated_delta_rule")(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_musa(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use MATE's MP31 decode kernel for its validated packed-QKV contract.

    MATE consumes unpacked strided views, so this adapter introduces no QKV
    materialization.  Unsupported devices, dtypes, layouts, and MATE versions
    retain SGLang's original packed Triton implementation.  Set
    ``SGLANG_MUSA_MATE_GDN=off`` to force that fallback.
    """
    if _supports_mate_packed_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
    ):
        mate_gdn_decode = _load_mate_gdn_decode()
        if mate_gdn_decode is not None:
            global _mate_gdn_match_logged
            B = mixed_qkv.shape[0]
            HV, V, K = initial_state.shape[-3:]
            H = ((mixed_qkv.shape[1] - HV * V) // 2) // K

            q_end = H * K
            k_end = 2 * H * K
            q = mixed_qkv[:, :q_end].reshape(B, 1, H, K)
            k = mixed_qkv[:, q_end:k_end].reshape(B, 1, H, K)
            v = mixed_qkv[:, k_end:].reshape(B, 1, HV, V)

            if not _mate_gdn_match_logged:
                logger.info(
                    "Using MATE GDN packed decode on MTT S5000 "
                    "(B=%d, H=%d, HV=%d, K=V=%d)",
                    B,
                    H,
                    HV,
                    K,
                )
                _mate_gdn_match_logged = True

            mate_out, _ = mate_gdn_decode(
                q=q,
                k=k,
                v=v,
                state=initial_state,
                A_log=A_log,
                a=a.reshape(B, 1, HV),
                dt_bias=dt_bias,
                b=b.reshape(B, 1, HV),
                state_layout="VK",
                state_indices=ssm_state_indices,
                scale=scale,
                output=out,
                disable_state_update=False,
                use_qk_l2norm=use_qk_l2norm_in_kernel,
            )
            return mate_out, initial_state

    return _original("fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
