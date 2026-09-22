"""Opt-in numerical gates. This migration's code/build run does not execute them.

RUN_DSV41_GPU_TESTS=1 pytest tests/dsv41/test_gpu_ops.py
"""

import os

import pytest
import torch

if os.environ.get("RUN_DSV41_GPU_TESTS") != "1":
    pytest.skip("GPU validation is explicitly opt-in", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("No CUDA device", allow_module_level=True)

from flag_gems.fused.dsv41.block_fp8_linear import block_fp8_linear
from flag_gems.fused.dsv41.fp4_indexer import fp4_index_logits_decode
from flag_gems.fused.dsv41.mhc import hc_combine, hc_mix_stats
from sglang_fl.dsv41.quantization import block_fp8_linear as linear_reference
from sglang_fl.dsv41.quantization import decode_e2m1, decode_scale


@pytest.mark.parametrize("m,n", [(0, 35), (1, 32), (17, 35), (65, 64)])
@pytest.mark.parametrize(
    "mode,native", [("quantized", False), ("quantized", True), ("bf16", False)]
)
def test_block32_distinct_scales(mode, native, m, n):
    if native and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Native FP8 GEMM unavailable")
    torch.manual_seed(413)
    x = torch.randn(m, 128).bfloat16()
    w = torch.randn(n, 128).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.03125, 0.5, 2.0, 16.0], [8.0, 1.0, 0.25, 0.0625]])[
        : (n + 31) // 32
    ]
    expected = linear_reference(x, w, scales, [32, 32], mode=mode)
    actual = block_fp8_linear(
        x.cuda(),
        w.cuda(),
        [32, 32],
        scales.cuda(),
        mode=mode,
        native_fp8=native,
        act_scale_ue8m0=True,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.008, atol=0.25)


def test_fp4_packed_pages_visibility_and_mixed_requests():
    torch.manual_seed(517)
    b, h, length, page = 3, 32, 65, 64
    payload = torch.randint(0, 256, (128, 64), dtype=torch.uint8)
    scale = torch.randint(125, 129, (128, 4), dtype=torch.uint8)
    scale[37, 2] = 255  # Reserved UE8M0 NaN must propagate.
    table = torch.cat([payload.reshape(2, page * 64), scale.reshape(2, page * 4)], 1)
    keys = (
        decode_e2m1(payload) * decode_scale(scale, "ue8m0").repeat_interleave(32, 1)
    ).bfloat16()
    q = torch.randn(b, h, 128).bfloat16()
    weights = torch.randn(b, h).bfloat16()
    slots = torch.stack(
        [torch.arange(length), torch.arange(length) + 31, torch.arange(length) + 15]
    )
    lens = torch.tensor([0, 63, 65])
    reference = []
    for row in range(b):
        dot = (q[row].float() @ keys[slots[row]].float().T).bfloat16()
        scores = (
            (dot.relu().float() * weights[row, :, None].float())
            .bfloat16()
            .float()
            .sum(0)
            .bfloat16()
            .float()
        )
        scores[lens[row] :] = -torch.inf
        reference.append(scores)
    result = fp4_index_logits_decode(
        q.cuda(), weights.cuda(), slots.cuda(), lens.cuda(), table.cuda(), page
    )
    torch.testing.assert_close(
        result.cpu(), torch.stack(reference), rtol=0.008, atol=0.25, equal_nan=True
    )


def test_mhc_fp32_weights_and_predecessor_combine():
    torch.manual_seed(713)
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(24, 512, device="cuda", dtype=torch.float32) * 0.01
    expected = (x.float() @ weight.T) * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    actual = hc_mix_stats(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    previous = torch.rand(7, 4, device="cuda")
    expected_y = (x.reshape(7, 4, 128).float() * previous[..., None]).sum(1).bfloat16()
    torch.testing.assert_close(
        hc_combine(x, previous, 4, x.dtype), expected_y, rtol=0.008, atol=0.01
    )
