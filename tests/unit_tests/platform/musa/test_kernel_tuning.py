# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from sglang_fl.dispatch.backends.vendor.mthreads.patches import (
    moe_schedule,
    topk_schedule,
)
from sglang_fl.dispatch.config.utils import _load_kernel_tuning, get_kernel_tuning


def test_platform_default_is_copied_and_loaded_once(monkeypatch):
    monkeypatch.delenv("SGLANG_FL_CONFIG", raising=False)
    _load_kernel_tuning.cache_clear()
    first = get_kernel_tuning("fused_moe", "decode")
    first["num_warps"] = 16
    assert get_kernel_tuning("fused_moe", "decode")["num_warps"] == 4
    assert _load_kernel_tuning.cache_info().misses == 1


def test_custom_yaml_reaches_real_schedule_bridge(tmp_path, monkeypatch):
    path = tmp_path / "musa.yaml"
    selected = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=3,
        num_warps=8,
        num_stages=1,
    )
    path.write_text(
        yaml.safe_dump({"kernel_tuning": {"fused_moe": {"decode": selected}}})
    )
    monkeypatch.setenv("SGLANG_FL_CONFIG", str(path))
    monkeypatch.setattr(moe_schedule, "_on_s5000", lambda: True)
    native_calls = []
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(
        lambda *a, **k: native_calls.append(a) or "native"
    )
    assert wrapped((256, 512, 2048), (256, 2048, 256), 8, None, 64) == selected
    assert not native_calls
    # SGLANG_FL_CONFIG is a complete replacement. Missing prefill profile
    # retains the native selector instead of borrowing platform defaults.
    assert wrapped((256, 512, 2048), (256, 2048, 256), 8, None, 2048) == "native"
    assert len(native_calls) == 1


def test_topk_options_follow_yaml_without_mutating_shared_config(tmp_path, monkeypatch):
    path = tmp_path / "musa.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "kernel_tuning": {
                    "topk": {"qwen36": {"num_warps": 2, "num_ctas": 1, "num_stages": 1}}
                }
            }
        )
    )
    monkeypatch.setenv("SGLANG_FL_CONFIG", str(path))
    assert topk_schedule._pinned_options() == dict(
        num_warps=2, num_ctas=1, num_stages=1
    )


@pytest.mark.parametrize("value", [0, -1, True, "4", 3])
def test_invalid_profile_fails_before_native_or_kernel_execution(
    tmp_path, monkeypatch, value
):
    path = tmp_path / "musa.yaml"
    path.write_text(
        yaml.safe_dump(
            {"kernel_tuning": {"fused_moe": {"decode": {"num_warps": value}}}}
        )
    )
    monkeypatch.setenv("SGLANG_FL_CONFIG", str(path))
    monkeypatch.setattr(moe_schedule, "_on_s5000", lambda: True)
    wrapped = moe_schedule._wrap_try_get_optimal_moe_config(
        lambda *a, **k: pytest.fail("invalid config must fail before native selection")
    )
    with pytest.raises(ValueError):
        wrapped((256, 512, 2048), (256, 2048, 256), 8, None, 64)
