# SPDX-License-Identifier: Apache-2.0
"""Default-off Hy4-preview attention-post to MLP-pre orchestration.

Numerics belong to FlagGems-sglang. The original following RMSNorm remains the
consumer so its selected provider, rounding and hooks remain observable.
"""

import os

FLAG = "SGLANG_HY4_IHC_POST_PRE"
CANDIDATE_ROWS = (1, 4, 128, 208, 256)


def maybe_ihc_post_pre(owner, output, residual, post):
    if os.environ.get(FLAG, "0") != "1":
        return None

    import torch
    from torch.nn.modules import module as nn_module

    from .model import HYV4HCLayer, HYV4HCPreLayer, _hpc_ihc_available

    before, after = owner.hc_attn_layer, owner.hc_mlp_layer
    pre = after.hc_pre
    if (
        type(before) is not HYV4HCLayer
        or type(after) is not HYV4HCLayer
        or type(pre) is not HYV4HCPreLayer
        or before.hidden_size != 6144
        or after.hidden_size != 6144
        or before.hc_mult != 4
        or after.hc_mult != 4
        or pre.hidden_size != 6144
        or pre.hc_mult != 4
        or before._fused_ihc_post_disabled
        or pre._fused_ihc_pre_disabled
        or not output.is_cuda
        or output.shape[0] not in CANDIDATE_ROWS
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return None
    for obj, name, expected in (
        (before, "post", HYV4HCLayer.post),
        (after, "pre", HYV4HCLayer.pre),
        (after, "prepare_input", HYV4HCLayer.prepare_input),
        (pre, "forward", HYV4HCPreLayer.forward),
    ):
        if getattr(getattr(obj, name), "__func__", None) is not expected:
            return None
    for obj in (nn_module, before, after, pre, pre.hc_fn):
        if any(
            getattr(obj, name, None)
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        ):
            return None
    # This candidate qualifies the observed split Triton path. An installed
    # HPC pre/post provider is a separate baseline and keeps its native route.
    if (
        _hpc_ihc_available("fuse_ihc_pre", 4, 6144)
        or _hpc_ihc_available("fuse_ihc_post", 4, 6144)
        or torch.cuda.get_device_capability(output.device) != (9, 0)
    ):
        return None

    try:
        from flaggems_sglang.ops.ihc_post_pre import can_ihc_post_pre, ihc_post_pre
    except ImportError:
        return None

    args = (
        output,
        residual,
        post,
        pre.hc_fn.weight,
        pre.hc_scale,
        pre.hc_base,
        pre.magnitude,
        pre.hc_eps,
        pre.rms_norm_eps,
    )
    if not can_ihc_post_pre(*args):
        return None
    reduced, next_post, next_residual = ihc_post_pre(*args)
    normalized = owner.post_attention_layernorm(reduced)
    return normalized, next_post, next_residual
