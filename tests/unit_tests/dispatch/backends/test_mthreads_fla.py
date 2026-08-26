# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the MTT S5000 FLA adapters."""

from __future__ import annotations

from unittest.mock import Mock

import torch


def _packed_inputs(*, state_dtype=torch.float32):
    batch, qk_heads, value_heads, dim = 2, 2, 4, 128
    packed_dim = 2 * qk_heads * dim + value_heads * dim
    return {
        "mixed_qkv": torch.randn(batch, packed_dim, dtype=torch.bfloat16),
        "a": torch.randn(batch, value_heads, dtype=torch.bfloat16),
        "b": torch.randn(batch, value_heads, dtype=torch.bfloat16),
        "A_log": torch.randn(value_heads, dtype=torch.float32),
        "dt_bias": torch.randn(value_heads, dtype=torch.float32),
        "scale": dim**-0.5,
        "initial_state": torch.randn(8, value_heads, dim, dim, dtype=state_dtype),
        "out": torch.empty(batch, 1, value_heads, dim, dtype=torch.bfloat16),
        "ssm_state_indices": torch.tensor([1, 5], dtype=torch.int64),
        "use_qk_l2norm_in_kernel": True,
    }


def test_mate_packed_decode_uses_strided_views_and_caller_buffers(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _packed_inputs()
    captured = {}

    def fake_mate_gdn_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(3)
        return kwargs["output"], kwargs["state"]

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_decode", lambda: fake_mate_gdn_decode)

    result, state = fla.fused_recurrent_gated_delta_rule_packed_decode_musa(**inputs)

    assert result is inputs["out"]
    assert state is inputs["initial_state"]
    assert captured["q"].shape == (2, 1, 2, 128)
    assert captured["k"].shape == (2, 1, 2, 128)
    assert captured["v"].shape == (2, 1, 4, 128)
    assert captured["a"].shape == (2, 1, 4)
    assert captured["b"].shape == (2, 1, 4)
    assert captured["q"]._base is not None
    assert captured["k"]._base is not None
    assert captured["v"]._base is not None
    assert captured["state_layout"] == "VK"
    assert captured["state_indices"] is inputs["ssm_state_indices"]
    assert captured["scale"] == inputs["scale"]
    assert captured["output"] is inputs["out"]
    assert captured["use_qk_l2norm"] is True
    assert torch.all(result == 3)


def test_mate_packed_decode_falls_back_for_unsupported_state(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _packed_inputs(state_dtype=torch.bfloat16)
    fallback_result = (torch.empty(1), torch.empty(1))
    original = Mock(return_value=fallback_result)

    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_decode", Mock())
    monkeypatch.setattr(fla, "_original", lambda name: original)

    assert (
        fla.fused_recurrent_gated_delta_rule_packed_decode_musa(**inputs)
        is fallback_result
    )
    fla._load_mate_gdn_decode.assert_not_called()
    original.assert_called_once_with(**inputs)


def test_mate_packed_decode_can_be_disabled(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.mthreads.impl import fla

    inputs = _packed_inputs()
    fallback_result = (torch.empty(1), torch.empty(1))
    original = Mock(return_value=fallback_result)

    monkeypatch.setenv("SGLANG_MUSA_MATE_GDN", "off")
    monkeypatch.setattr(fla, "_is_s5000", lambda device: True)
    monkeypatch.setattr(fla, "_load_mate_gdn_decode", Mock())
    monkeypatch.setattr(fla, "_original", lambda name: original)

    assert (
        fla.fused_recurrent_gated_delta_rule_packed_decode_musa(**inputs)
        is fallback_result
    )
    fla._load_mate_gdn_decode.assert_not_called()
    original.assert_called_once_with(**inputs)
