"""CPU numerical contracts; independent of device availability and FlagGems init."""

import pytest
import torch

from sglang_fl.dsv41.quantization import (
    block_fp8_linear,
    decode_e2m1,
    decode_e4m3,
    decode_scale,
    quantize_activations,
    round_e4m3,
)


def test_e2m1_every_code_and_nibble_order():
    packed = torch.arange(256, dtype=torch.uint8)
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    expected = torch.stack(
        (lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1
    ).flatten()
    actual = decode_e2m1(packed)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_e4m3_all_bytes_against_torch_decoder():
    codes = torch.arange(256, dtype=torch.uint8)
    expected = codes.view(torch.float8_e4m3fn).float()
    actual = decode_e4m3(codes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(
        torch.signbit(actual[~expected.isnan()]),
        torch.signbit(expected[~expected.isnan()]),
    )


def test_ue8m0_exponents_and_nan_are_not_integer_multipliers():
    codes = torch.arange(256, dtype=torch.uint8)
    values = decode_scale(codes, "ue8m0")
    assert values[0] == 2.0**-127 and values[127] == 1.0
    assert values[128] == 2.0 and values[254] == 2.0**127
    assert values[255].isnan()
    with pytest.raises(TypeError):
        decode_scale(codes, "float")
    with pytest.raises(TypeError):
        decode_scale(codes.float(), "ue8m0")


def test_e4m3_rounding_ties_subnormals_and_saturation():
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mid = (positive[:-1] + positive[1:]) / 2
    values = torch.cat([positive, mid, -mid, torch.tensor([1e-12, -0.0, 1e6, -1e6])])
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    actual = round_e4m3(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("ue8m0", [False, True])
def test_group32_unequal_scales_and_tail_n(ue8m0):
    torch.manual_seed(413)
    x = torch.randn(3, 128).bfloat16()
    w = torch.randn(35, 128).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.03125, 0.5, 2.0, 16.0], [8.0, 1.0, 0.25, 0.0625]])
    q, qs = quantize_activations(x, 32, ue8m0=ue8m0)
    # Independent dense dequantization detects accidental sharing of K scales.
    expected = (
        (q * qs.repeat_interleave(32, 1))
        @ (w.float() * scales.repeat_interleave(32, 0)[:35].repeat_interleave(32, 1)).T
    ).bfloat16()
    actual = block_fp8_linear(x, w, scales, [32, 32], ue8m0=ue8m0)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.25)


def test_modes_are_explicit_and_empty_input_is_supported():
    x = torch.linspace(-2, 2, 64).reshape(1, 64).bfloat16()
    w = torch.ones((1, 64))
    w[:, :32] = 2.0
    w = w.to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 8.0]])
    quantized = block_fp8_linear(x, w, scales, [32, 32], mode="quantized")
    bf16 = block_fp8_linear(x, w, scales, [32, 32], mode="bf16")
    assert not torch.equal(quantized, bf16)
    assert block_fp8_linear(x[:0], w, scales, [32, 32]).shape == (0, 1)
    with pytest.raises(ValueError):
        block_fp8_linear(x, w, scales, [32, 0])
    with pytest.raises(TypeError):
        block_fp8_linear(x, w, scales.to(torch.uint8), [32, 32])
