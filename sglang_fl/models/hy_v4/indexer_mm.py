# SPDX-License-Identifier: Apache-2.0
"""Explicit per-HY4-indexer BF16 projection opt-in; default OFF.

Bind only the constructed wq_b instance. Keep module/parameter names, native
weight loading, RMS, INT8 q_b, indexer quantization, cache and RoPE unchanged.
"""

import os
import re
from types import MethodType

import torch

FLAG = "SGLANG_HY4_INDEXER_BF16_MM"


def provider_ok():
    try:
        from torch.overrides import _get_current_function_mode
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        if (
            torch.compiler.is_compiling()
            or _get_current_function_mode() is not None
            or _get_current_dispatch_mode() is not None
        ):
            return False
        table = torch._C._dispatch_dump_table("aten::mm")
        return any(
            re.fullmatch(
                "CUDA: registered at .*[/\\\\]RegisterCUDA(?:_\\d+)?\\.cpp:\\d+ \\[kernel\\]",
                line,
            )
            for line in table.splitlines()
        )
    except (AttributeError, RuntimeError, ImportError):
        return False


def eligible(layer, x):
    w = layer.weight
    return (
        getattr(layer, "_hy4_mm_enabled", False) is True
        and getattr(layer, "_hy4_mm_sm90", False) is True
        and (layer.bias is None)
        and (not layer.skip_bias_add)
        and x.is_cuda
        and w.is_cuda
        and (x.device == w.device)
        and (x.dtype == w.dtype == torch.bfloat16)
        and (x.ndim == w.ndim == 2)
        and (x.shape[0] in (208, 256))
        and (x.shape[1] == 2048)
        and (tuple(w.shape) == (4096, 2048))
        and (tuple(x.stride()) == (2048, 1))
        and (tuple(w.stride()) == (2048, 1))
        and (x.storage_offset() == w.storage_offset() == 0)
        and (not x.is_conj())
        and (not x.is_neg())
        and (not w.is_conj())
        and (not w.is_neg())
        and (not torch.is_grad_enabled())
        and (not x.requires_grad)
        and (not w.requires_grad)
        and (not torch.backends.cuda.matmul.allow_tf32)
        and (not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    )


def _forward(layer, x, *args, **kwargs):
    if args or kwargs or (not eligible(layer, x)) or (not provider_ok()):
        return layer._hy4_mm_native_forward(x, *args, **kwargs)
    return (torch.mm(x, layer.weight.T), None)


def install_indexer_mm(indexer):
    if os.environ.get(FLAG, "0") != "1":
        return
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    layer = indexer.wq_b
    if type(layer.quant_method) is not UnquantizedLinearMethod:
        return
    if getattr(layer, "_hy4_mm_enabled", False):
        raise RuntimeError("wq_b candidate already installed")
    layer._hy4_mm_sm90 = layer.weight.is_cuda and torch.cuda.get_device_capability(
        layer.weight.device
    ) == (9, 0)
    layer._hy4_mm_enabled = True
    layer._hy4_mm_native_forward = layer.forward
    layer.forward = MethodType(_forward, layer)
