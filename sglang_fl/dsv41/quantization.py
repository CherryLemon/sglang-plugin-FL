"""Explicit checkpoint formats and small, device-independent references.

These routines are correctness references, not production expert/attention
kernels. Quantized mode retains E4M3 rounding even on devices without FP8 GEMM.
BF16 mode deliberately omits activation quantization and is a different mode.
"""

from enum import Enum

import torch


class ScaleEncoding(str, Enum):
    FLOAT = "float"
    UE8M0 = "ue8m0"


class QuantizationMode(str, Enum):
    QUANTIZED = "quantized"
    BF16 = "bf16"


def decode_scale(scale: torch.Tensor, encoding: ScaleEncoding) -> torch.Tensor:
    encoding = ScaleEncoding(encoding)
    if encoding is ScaleEncoding.FLOAT:
        if not scale.is_floating_point():
            raise TypeError("numeric scales must have a floating dtype")
        return scale.float()
    if scale.dtype != torch.uint8:
        raise TypeError("UE8M0 scales must be uint8 exponent bytes")
    exponent = scale.to(torch.int32)
    value = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), exponent - 127)
    return torch.where(exponent == 255, float("nan"), value)


def decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Low nibble first; output has twice the last dimension, including -0."""
    if packed.dtype != torch.uint8 or packed.ndim == 0:
        raise TypeError("packed E2M1 must be a non-scalar uint8 tensor")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1)
    codes = codes.reshape(*packed.shape[:-1], 2 * packed.shape[-1])
    magnitude = codes & 7
    # E2M1's unsigned magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6.
    exponent = (magnitude >> 1).to(torch.int32)
    mantissa = (magnitude & 1).float()
    value = torch.where(
        exponent == 0,
        mantissa * 0.5,
        torch.ldexp(1.0 + mantissa * 0.5, exponent - 1),
    )
    return torch.where((codes & 8) != 0, -value, value)


def decode_e4m3(codes: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN bytes without requiring device float8 dtype support."""
    if codes.dtype != torch.uint8:
        raise TypeError("E4M3FN storage must be uint8")
    exponent = ((codes >> 3) & 15).to(torch.int32)
    mantissa = (codes & 7).float()
    value = torch.where(
        exponent == 0,
        mantissa * (2.0**-9),
        torch.ldexp(1.0 + mantissa / 8.0, exponent - 7),
    )
    value = torch.where((codes & 127) == 127, float("nan"), value)
    return torch.where((codes & 128) != 0, -value, value)


def round_e4m3(value: torch.Tensor) -> torch.Tensor:
    """Saturating, ties-to-even E4M3FN rounding, returned in FP32."""
    x = value.float()
    magnitude = x.abs().clamp(max=448.0)
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-6)))
    step = torch.exp2(exponent - 3.0)
    rounded = (torch.round(magnitude / step) * step).clamp(max=448.0)
    return torch.copysign(rounded, x)


def quantize_activations(x: torch.Tensor, group_k: int, *, ue8m0: bool):
    if group_k <= 0 or x.ndim != 2 or x.shape[1] % group_k:
        raise ValueError("activation matrix K must be divisible by group_k")
    groups = x.float().reshape(x.shape[0], x.shape[1] // group_k, group_k)
    scale = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return round_e4m3(groups / scale.unsqueeze(-1)).reshape_as(x), scale


def dequantize_block_weight(weight, scale, block_size, *, encoding=ScaleEncoding.FLOAT):
    if weight.ndim != 2 or len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError("expected matrix weight and positive (block_n, block_k)")
    n, k = weight.shape
    bn, bk = block_size
    expected = ((n + bn - 1) // bn, (k + bk - 1) // bk)
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale shape {tuple(scale.shape)} != {expected}")
    values = decode_e4m3(weight) if weight.dtype == torch.uint8 else weight.float()
    scales = decode_scale(scale, encoding)
    expanded = scales.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]
    return values * expanded


def block_fp8_linear(
    x,
    weight,
    scale,
    block_size,
    bias=None,
    *,
    mode=QuantizationMode.QUANTIZED,
    encoding=ScaleEncoding.FLOAT,
    ue8m0=True,
):
    """Small correctness reference with bounded per-K-group intermediates."""
    mode = QuantizationMode(mode)
    if x.shape[-1] != weight.shape[-1] or x.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("invalid activation dtype or linear K")
    if weight.ndim != 2 or len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError("expected matrix weight and positive block sizes")
    if weight.shape[-1] == 0:
        raise ValueError("K must be positive")
    rows = x.reshape(-1, x.shape[-1])
    bn, bk = block_size
    n, k = weight.shape
    if k % bk:
        raise ValueError("activation K must be divisible by the quantization group")
    numeric_scale = decode_scale(scale, encoding)
    if tuple(scale.shape) != ((n + bn - 1) // bn, k // bk):
        raise ValueError("invalid block scale layout")
    if mode is QuantizationMode.QUANTIZED:
        q, activation_scale = quantize_activations(rows, bk, ue8m0=ue8m0)
    else:
        q = rows.to(torch.bfloat16).float()
        activation_scale = torch.ones((rows.shape[0], k // bk), device=x.device)
    out = torch.zeros((rows.shape[0], n), dtype=torch.float32, device=x.device)
    for group, start in enumerate(range(0, k, bk)):
        w = weight[:, start : start + bk]
        w = decode_e4m3(w) if w.dtype == torch.uint8 else w.float()
        ws = numeric_scale[:, group].repeat_interleave(bn)[:n]
        if mode is QuantizationMode.BF16:
            w = (w * ws[:, None]).to(torch.bfloat16).float()
            out += q[:, start : start + bk] @ w.T
        else:
            dot = q[:, start : start + bk] @ w.T
            out += dot * activation_scale[:, group, None] * ws[None, :]
    out = out.to(x.dtype)
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], n)
