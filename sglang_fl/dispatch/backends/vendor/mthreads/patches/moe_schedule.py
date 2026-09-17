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

"""Robust fused-MoE scheduling for MTT S5000 decode and long prefill.

TorchAda's bundled Triton 3.2 configuration uses eight warps and a K=128
tile for the Qwen3.6-35B-A3B TP2 decode shape.  That configuration has a
large performance cliff on some S5000 systems with Triton 3.6.  A four-warp,
K=64 configuration is within a few percent of the old-system optimum and is
about three times faster on the affected systems.

Long-prefill profiling found that the generic M=64/N=64/K=32 configuration
leaves expert-padding and occupancy performance on the table.  The measured
M=32/N=128/K=64, eight-warp, one-stage schedule is 12-27% faster across
8192-token random, balanced-shuffled, and block-boundary routes, and across
2048/4096/6144/8192-token chunks.  A separate fixed-work confirmation covers
only the exact M=16384 shape with M=64/N=128/K=64; no intermediate or adjacent
token count is widened by this patch.

Keep these as vendor monkeypatches: SGLang and TorchAda remain unmodified, and
operators can disable decode and prefill independently. Patch both resolvers
because SGLang v0.5.11 carries its own copy while some MUSA integration
versions call TorchAda's runtime copy directly.
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_MOE_DECODE_SCHEDULE"
_PREFILL_ENV_NAME = "SGLANG_MUSA_MOE_PREFILL_SCHEDULE"
# Keep compiler backend optimization opt-in and confined to the measured M=64
# MUSA decode path.  The service candidate sets this to 1 explicitly.
_BACKEND_OPT_ENV_NAME = "SGLANG_MUSA_MOE_BACKEND_OPT"
_PATCH_MARKER = "_sglang_fl_musa_moe_schedule"
_decode_match_logged = False
_prefill_match_logged = False
_prefill_m16k_match_logged = False
_r2_m4_match_logged = False
# R2's core M4 BN64 selector requires this original SGLang baseline tile.
# The August image's TorchAda table instead selects BM128/G64 at M4.
# Explicit restoration only; do not change any other image table entry.
_R2_M4_BASELINE_ENV = "SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE"
_R2_M4_BASELINE_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
}
_S5000_DECODE_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
}
_S5000_PREFILL_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 1,
}
_S5000_PREFILL_M16K_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 1,
}


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "auto").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def _prefill_enabled() -> bool:
    return os.environ.get(_PREFILL_ENV_NAME, "auto").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def _device_name() -> str:
    try:
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            return str(torch.musa.get_device_name())
    except Exception:
        pass
    return ""


def _backend_opt_enabled() -> bool:
    """Enable Triton backend optimization only for an explicit candidate opt-in."""

    if os.environ.get(_BACKEND_OPT_ENV_NAME, "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
        "enable",
        "enabled",
    }:
        return False
    if "S5000" not in _device_name().upper():
        return False
    try:
        import triton

        major, minor = (int(part) for part in str(triton.__version__).split(".", 2)[:2])
    except (AttributeError, ImportError, TypeError, ValueError):
        return False
    return (major, minor) == (3, 6)


def _matches_s5000_decode(
    w1_shape,
    w2_shape,
    top_k: int,
    dtype,
    M: int,
    *,
    block_shape=None,
    per_channel_quant: bool = False,
) -> bool:
    """Match only the measured Qwen3.6-35B-A3B TP2 BF16 decode shape."""

    return (
        _enabled()
        and "S5000" in _device_name().upper()
        and len(w1_shape) == 3
        and len(w2_shape) == 3
        and w1_shape[0] == w2_shape[0] == 256
        and w1_shape[2] == w2_shape[1] == 2048
        and w1_shape[1] == 2 * w2_shape[2]
        and w2_shape[2] in (256, 512)
        and top_k == 8
        and dtype is None
        and M == 64
        and block_shape is None
        and not per_channel_quant
    )


def _matches_s5000_prefill(
    w1_shape,
    w2_shape,
    top_k: int,
    dtype,
    M: int,
    *,
    block_shape=None,
    per_channel_quant: bool = False,
) -> bool:
    """Match only the measured Qwen3.6 TP2 BF16 long-prefill range."""

    return (
        _prefill_enabled()
        and "S5000" in _device_name().upper()
        and len(w1_shape) == 3
        and len(w2_shape) == 3
        and w1_shape[0] == w2_shape[0] == 256
        and w1_shape[2] == w2_shape[1] == 2048
        and w1_shape[1] == 2 * w2_shape[2]
        and w2_shape[2] == 256
        and top_k == 8
        and dtype is None
        and 2048 <= M <= 8192
        and block_shape is None
        and not per_channel_quant
    )


def _matches_s5000_m16k_prefill(
    w1_shape,
    w2_shape,
    top_k: int,
    dtype,
    M: int,
    *,
    block_shape=None,
    per_channel_quant: bool = False,
) -> bool:
    """Match only the measured exact M=16384 prefill shape."""

    return (
        _prefill_enabled()
        and "S5000" in _device_name().upper()
        and len(w1_shape) == 3
        and len(w2_shape) == 3
        and w1_shape[0] == w2_shape[0] == 256
        and w1_shape[2] == w2_shape[1] == 2048
        and w1_shape[1] == 2 * w2_shape[2]
        and w2_shape[2] == 256
        and top_k == 8
        and dtype is None
        and M == 16384
        and block_shape is None
        and not per_channel_quant
    )


def _wrap_try_get_optimal_moe_config(original: Callable[..., Any]):
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(
        w1_shape,
        w2_shape,
        top_k,
        dtype,
        M,
        is_marlin=False,
        block_shape=None,
        per_channel_quant=False,
        return_down_config=False,
    ):
        global _decode_match_logged, _prefill_match_logged, _prefill_m16k_match_logged
        global _r2_m4_match_logged
        if (
            os.environ.get(_R2_M4_BASELINE_ENV, "0") == "1"
            and "S5000" in _device_name().upper()
            and tuple(w1_shape) == (256, 512, 2048)
            and tuple(w2_shape) == (256, 2048, 256)
            and top_k == 8
            and dtype is None
            and M == 4
            and not is_marlin
            and block_shape is None
            and not per_channel_quant
        ):
            if not _r2_m4_match_logged:
                logger.info("Restored R2 M4 baseline MoE tile BM16/BN32/BK64/G1; W13 BN64 remains a separate core opt-in")
                _r2_m4_match_logged = True
            config = dict(_R2_M4_BASELINE_CONFIG)
            # R2 baseline has no independent down schedule. The core uses
            # the unchanged baseline config for W2, and copies W13 for BN64.
            return (config, (None, None)) if return_down_config else config
        if _matches_s5000_decode(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            block_shape=block_shape,
            per_channel_quant=per_channel_quant,
        ):
            backend_opt = w2_shape[2] == 256 and _backend_opt_enabled()
            config = dict(_S5000_DECODE_CONFIG)
            if backend_opt:
                config["enable_backend_opt"] = True
            if not _decode_match_logged:
                logger.info(
                    "MUSA S5000 MoE decode schedule selected for "
                    "w1=%s, w2=%s, top_k=%s, M=%s, backend_opt=%s",
                    tuple(w1_shape),
                    tuple(w2_shape),
                    top_k,
                    M,
                    backend_opt,
                )
                _decode_match_logged = True
            if return_down_config:
                return config, (dict(config), config["BLOCK_SIZE_M"])
            return config
        if _matches_s5000_prefill(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            block_shape=block_shape,
            per_channel_quant=per_channel_quant,
        ):
            config = dict(_S5000_PREFILL_CONFIG)
            if not _prefill_match_logged:
                logger.info(
                    "MUSA S5000 MoE prefill schedule selected for "
                    "w1=%s, w2=%s, top_k=%s, M=%s",
                    tuple(w1_shape),
                    tuple(w2_shape),
                    top_k,
                    M,
                )
                _prefill_match_logged = True
            if return_down_config:
                return config, (dict(config), config["BLOCK_SIZE_M"])
            return config
        if _matches_s5000_m16k_prefill(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            block_shape=block_shape,
            per_channel_quant=per_channel_quant,
        ):
            config = dict(_S5000_PREFILL_M16K_CONFIG)
            if not _prefill_m16k_match_logged:
                logger.info(
                    "MUSA S5000 MoE exact M=16384 prefill schedule selected for "
                    "w1=%s, w2=%s, top_k=%s",
                    tuple(w1_shape),
                    tuple(w2_shape),
                    top_k,
                )
                _prefill_m16k_match_logged = True
            if return_down_config:
                return config, (dict(config), config["BLOCK_SIZE_M"])
            return config
        return original(
            w1_shape,
            w2_shape,
            top_k,
            dtype,
            M,
            is_marlin=is_marlin,
            block_shape=block_shape,
            per_channel_quant=per_channel_quant,
            return_down_config=return_down_config,
        )

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _patch_resolver(config_module_name: str, fused_moe_module_name: str) -> bool:
    try:
        config_module = importlib.import_module(config_module_name)
        fused_moe_module = importlib.import_module(fused_moe_module_name)
    except ImportError as exc:
        logger.debug("MUSA MoE resolver %s unavailable: %s", config_module_name, exc)
        return False

    wrapped = _wrap_try_get_optimal_moe_config(
        config_module.try_get_optimal_moe_config
    )
    config_module.try_get_optimal_moe_config = wrapped
    fused_moe_module.try_get_optimal_moe_config = wrapped
    return True


def apply_musa_moe_schedule_patch() -> bool:
    """Patch SGLang/TorchAda config resolvers and already-imported aliases."""

    if not (_enabled() or _prefill_enabled()):
        logger.info(
            "MUSA S5000 MoE schedule patch disabled by %s and %s",
            _ENV_NAME,
            _PREFILL_ENV_NAME,
        )
        return False

    patched = False
    patched |= _patch_resolver(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config",
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
    )
    patched |= _patch_resolver(
        "torchada.triton.runtime.fused_moe.config",
        "torchada.triton.runtime.fused_moe.fused_moe",
    )
    if not patched:
        logger.warning("MUSA S5000 MoE decode schedule patch skipped: no resolver")
        return False

    logger.info(
        "MUSA S5000 MoE schedules applied: decode M=64 "
        "(M32/N32/K64/G1/W4/S1); prefill M=2048..8192 "
        "(M32/N128/K64/G4/W8/S1); exact M=16384 "
        "(M64/N128/K64/G4/W8/S1)"
    )
    return True
