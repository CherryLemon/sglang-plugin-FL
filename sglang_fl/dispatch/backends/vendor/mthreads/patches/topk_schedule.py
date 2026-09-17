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

"""Skip cold softmax-TopK autotuning for measured MTT S5000 shapes.

The MUSA kernel carries fifteen launch configurations.  Autotuning them also
flushes a 256 MiB cache buffer between candidates, making the first large
prefill wave several seconds slower.  On MP31, ``warps=1, stages=1`` is the
validated choice for Qwen3.6-35B-A3B's ``E=256, K=8`` graph and prefill shapes.

Only those exact shapes are pinned.  Other models and shapes keep SGLang's
normal autotuner.  Set ``SGLANG_MUSA_TOPK_SCHEDULE=off`` to disable the patch.

The pinned path launches the kernel's underlying JIT function directly with
the pinned options instead of mutating the shared ``kernel.configs`` list:
the single-config autotuner branch would only forward the same arguments to
the same ``fn.run`` call while touching shared tuner state, so the direct
call is launch-equivalent with no global mutation.  See ``_pinned_launch``.

Guard verification happens once at wrapper/application startup
(``_verify_pinned_startup``), not on every target call.  The wrapper closes
over the verified inner JIT object and an immutable (``MappingProxyType``)
copy of the pinned option kwargs.  Per-call work is limited to the shape gate, the MUSA-device gate,
and a cheap ``kernel.fn`` identity check (runtime replacement fails closed
to the original path).  No ``configs[0] == selected`` constraint is imposed:
the pinned path never reads ``kernel.configs``.

Stream behavior (MUSA Triton 3.6, read-only inspection): ``JITFunction.run``
takes the caller current stream via ``driver.active.get_current_stream`` and
launches on it, so the pinned direct call inherits the caller stream exactly
like the autotuner branch would.  ``do_bench`` (``triton.testing.do_bench``
via ``MusaDriver.get_benchmarker``) takes only the benchmark closure with no
stream parameter and runs repeated calls on the caller stream; it does not
create a dedicated reusable stream.  This fix only covers the pinned target
path.  The raw fallback path still uses the stock autotuner/benchmark flow
and its first-key host-state race is NOT claimed fixed.
"""

from __future__ import annotations

import inspect
import logging
import os
from functools import wraps
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

_ENV_NAME = "SGLANG_MUSA_TOPK_SCHEDULE"
_PATCH_MARKER = "_sglang_fl_musa_topk_schedule"
_PINNED_OPTION_KWARGS = {"num_warps": 1, "num_ctas": 1, "num_stages": 1}
# Exact inner-JIT parameter names, arity, and order the pinned launcher
# passes positionally/by keyword.  Validated at startup via
# ``_inner_jit_abi_ok``; any drift fails closed.
_EXPECTED_JIT_ARG_NAMES = (
    "gating_output_ptr",
    "selected_expert_ptr",
    "moe_weights_ptr",
    "renormalize_flag",
    "num_experts",
    "num_tokens",
    "moe_softcapping",
    "correction_bias_ptr",
    "has_correction_bias",
    "K",
    "BLOCK_K",
    "BLOCK_WIDTH_SIZE_UP",
)
_TARGET_GRAPH_TOKEN_COUNTS = frozenset(
    {
        # CUDA-graph capture batches.
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    }
)
_TARGET_PREFILL_TOKEN_RANGE = range(1024, 16385)


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "auto").strip().lower() not in {
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
    except ImportError:
        return ""

    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return str(torch.musa.get_device_name())
    except RuntimeError:
        logger.debug("Unable to query MUSA device name", exc_info=True)
    return ""


def _is_target_shape(
    topk_weights: Any,
    gating_output: Any,
    moe_softcapping: float,
    correction_bias: Any,
) -> bool:
    try:
        num_tokens = int(gating_output.shape[0])
        return (
            gating_output.ndim == 2
            and (
                num_tokens in _TARGET_GRAPH_TOKEN_COUNTS
                or num_tokens in _TARGET_PREFILL_TOKEN_RANGE
            )
            and int(gating_output.shape[1]) == 256
            and int(topk_weights.shape[-1]) == 8
            and not moe_softcapping
            and correction_bias is None
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _is_musa_launch_eligible(
    topk_weights: Any, topk_ids: Any, gating_output: Any
) -> bool:
    """CPU-only per-call guard: pinned launch requires MUSA torch tensors.

    Returns True only when all three tensors are ``torch.Tensor`` on a
    ``musa`` device.  CPU tensors, non-tensor fakes, or unknown devices fail
    closed to the original autotuner path (no pinned launch, no mutation).
    """
    try:
        import torch

        for tensor in (topk_weights, topk_ids, gating_output):
            if not isinstance(tensor, torch.Tensor):
                return False
            try:
                device_type = tensor.device.type
            except Exception:
                return False
            if device_type != "musa":
                return False
        return True
    except ImportError:
        return False
    except Exception:
        logger.debug("MUSA TopK device guard failed", exc_info=True)
        return False


def _fn_run_signature_ok(fn_run: Any) -> bool:
    """Check ``fn.run`` supports the exact pinned call shape.

    Requires ``grid`` and ``warmup`` parameter names passable as keywords
    alongside ``*args``/``**kwargs``, then behaviorally binds the precise
    call the pinned launcher makes: nine dummy positional arguments plus
    keyword ``grid``, ``warmup=False``, ``K``, ``BLOCK_K``,
    ``BLOCK_WIDTH_SIZE_UP``, and the pinned option kwargs.  A ``TypeError``
    fails closed.  This rejects signatures that look structurally similar
    but are not call-compatible, e.g. ``run(grid, warmup, *args, **kwargs)``
    (the nine positionals collide with the keyword ``grid``) or
    ``run(*args, grid, warmup, required, **kwargs)`` (missing required
    argument).  Unknown callables such as ``lambda: None`` fail closed.
    """
    try:
        sig = inspect.signature(fn_run)
    except (TypeError, ValueError):
        return False
    try:
        params = sig.parameters
    except Exception:
        return False
    if "grid" not in params or "warmup" not in params:
        return False
    allowed = (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if params["grid"].kind not in allowed:
        return False
    if params["warmup"].kind not in allowed:
        return False
    kinds = [p.kind for p in params.values()]
    if inspect.Parameter.VAR_POSITIONAL not in kinds:
        return False
    if inspect.Parameter.VAR_KEYWORD not in kinds:
        return False
    try:
        sig.bind(
            *[object() for _ in range(9)],
            grid=(1,),
            warmup=False,
            K=8,
            BLOCK_K=8,
            BLOCK_WIDTH_SIZE_UP=256,
            **dict(_PINNED_OPTION_KWARGS),
        )
    except TypeError:
        return False
    except Exception:
        logger.debug("MUSA TopK run-signature bind failed", exc_info=True)
        return False
    return True


def _inner_jit_abi_ok(inner: Any) -> bool:
    """Check the inner JIT parameter names, arity, and order.

    The pinned launcher passes nine positional arguments followed by the
    ``grid``/``warmup`` keywords and the ``K``/``BLOCK_K``/
    ``BLOCK_WIDTH_SIZE_UP`` constexpr keywords, so the inner function must
    declare exactly ``_EXPECTED_JIT_ARG_NAMES`` in order.  Anything else
    (renamed, reordered, added, or removed parameters) fails closed.
    """
    try:
        names = getattr(inner, "arg_names", None)
    except Exception:
        return False
    if names is None:
        return False
    try:
        return list(names) == list(_EXPECTED_JIT_ARG_NAMES)
    except Exception:
        return False


def _verify_pinned_startup(kernel: Any, selected_config: Any) -> Optional[dict]:
    """Verify once at wrapper/application startup; return ctx or None.

    Fail-closed checks: explicit Triton runtime class/package identity for
    the tuner, the selected config, and the inner JIT function; the selected
    config's ``all_kwargs()`` shape; the inner JIT parameter names, arity,
    and order (``_inner_jit_abi_ok``) plus tuner/JIT ``arg_names``
    consistency; empty reset/restore with no user hooks; and an ``fn.run``
    signature supporting ``(*args, grid, warmup, **kwargs)``.  On success
    returns ``{"inner_fn", "pinned_kwargs"}`` where ``pinned_kwargs`` is an
    immutable ``MappingProxyType`` frozen copy.  ``kernel.configs`` is never
    scanned and no ``configs[0] == selected`` constraint is imposed.
    ``inner.pre_run_hooks`` is intentionally not a rejection reason: the
    direct ``inner.run`` call executes pre-run hooks internally, preserving
    them.
    """
    try:
        import triton
        from triton.runtime.autotuner import Autotuner, Config
        from triton.runtime.jit import JITFunction
    except ImportError as exc:
        logger.debug("MUSA TopK schedule skipped: %s", exc)
        return None
    try:
        if getattr(triton, "__name__", "") != "triton":
            return None
        if type(selected_config).__name__ != "Config":
            return None
        if not str(getattr(type(selected_config), "__module__", "")).startswith(
            "triton."
        ):
            return None
        if not isinstance(selected_config, Config):
            return None
        try:
            all_kwargs = selected_config.all_kwargs()
        except Exception:
            return None
        if not isinstance(all_kwargs, dict):
            return None
        if dict(all_kwargs) != dict(_PINNED_OPTION_KWARGS):
            return None
        if getattr(selected_config, "pre_hook", None) is not None:
            return None
        if type(kernel).__name__ != "Autotuner":
            return None
        if not str(getattr(type(kernel), "__module__", "")).startswith("triton."):
            return None
        if not isinstance(kernel, Autotuner):
            return None
        if list(getattr(kernel, "reset_to_zero", None) or []) != []:
            return None
        if list(getattr(kernel, "restore_value", None) or []) != []:
            return None
        if bool(getattr(kernel, "user_defined_pre_hook", False)):
            return None
        if bool(getattr(kernel, "user_defined_post_hook", False)):
            return None
        inner = getattr(kernel, "fn", None)
        if inner is None:
            return None
        if type(inner).__name__ != "JITFunction":
            return None
        if not str(getattr(type(inner), "__module__", "")).startswith("triton."):
            return None
        if not isinstance(inner, JITFunction):
            return None
        run = getattr(inner, "run", None)
        if not callable(run):
            return None
        if not _fn_run_signature_ok(run):
            return None
        if not _inner_jit_abi_ok(inner):
            return None
        try:
            tuner_names = list(getattr(kernel, "arg_names", None) or [])
        except Exception:
            return None
        if tuner_names != list(_EXPECTED_JIT_ARG_NAMES):
            return None
        frozen: Mapping[str, Any] = MappingProxyType(dict(all_kwargs))
        return {"inner_fn": inner, "pinned_kwargs": frozen}
    except Exception:
        logger.debug("MUSA TopK pinned startup guard failed", exc_info=True)
        return None


def _pinned_equivalence_ok(kernel: Any, selected_config: Any) -> bool:
    """Bool alias kept for compatibility; see ``_verify_pinned_startup``."""
    return _verify_pinned_startup(kernel, selected_config) is not None


def _pinned_launch_closed(
    ctx: dict,
    topk_weights: Any,
    topk_ids: Any,
    gating_output: Any,
    renormalize: bool,
    moe_softcapping: float,
    correction_bias: Any,
) -> None:
    """Launch the pinned config without touching shared tuner state.

    Proven equivalent to the single-config autotuner branch for these exact
    shapes: the ``[grid]`` proxy forwards ``grid``/``warmup`` into
    ``Autotuner.run``, whose single-config branch only forwards every
    argument plus the pinned options to ``inner.run`` while also writing
    shared ``best_config``/``nargs`` state.  The direct call below passes the
    identical positional/keyword arguments, grid, warmup flag, and the
    immutable options mapping to the identical inner function object, and
    returns ``None``
    like the wrapped function.  Pre-run hooks are preserved because
    ``inner.run`` executes them internally.  The pinned call inherits the
    caller current stream exactly like the autotuner branch would.  Skipped
    shared-state writes have no readers outside the tuner itself:
    per-config hooks are empty, the tuner carries no run hooks, and
    ``best_config`` is only consumed by the tuner's own log line.
    """
    import triton

    inner = ctx["inner_fn"]
    pinned_kwargs = ctx["pinned_kwargs"]
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[-1]
    inner.run(
        gating_output,
        topk_ids,
        topk_weights,
        renormalize,
        num_experts,
        num_tokens,
        moe_softcapping,
        correction_bias,
        correction_bias is not None,
        grid=(num_tokens,),
        warmup=False,
        K=topk,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_WIDTH_SIZE_UP=triton.next_power_of_2(num_experts),
        **pinned_kwargs,
    )
    return None


def _pinned_launch(
    kernel: Any,
    selected_config: Any,
    topk_weights: Any,
    topk_ids: Any,
    gating_output: Any,
    renormalize: bool,
    moe_softcapping: float,
    correction_bias: Any,
) -> None:
    """Compatibility entry point; verifies then launches, else fails closed."""
    ctx = _verify_pinned_startup(kernel, selected_config)
    if ctx is None:
        raise RuntimeError("MUSA TopK pinned launch not verified; fail closed")
    return _pinned_launch_closed(
        ctx,
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        moe_softcapping,
        correction_bias,
    )


def _make_topk_wrapper(
    original: Callable[..., Any], kernel: Any, selected_config: Any
) -> Callable[..., Any]:
    # Guard verification happens once here at wrapper creation, never per
    # call.  The wrapper closes over the verified inner JIT object and the
    # immutable pinned kwargs.
    ctx = _verify_pinned_startup(kernel, selected_config)

    @wraps(original)
    def wrapped(
        topk_weights: Any,
        topk_ids: Any,
        gating_output: Any,
        renormalize: bool = False,
        moe_softcapping: float = 0,
        correction_bias: Any = None,
    ) -> Any:
        if not _is_target_shape(
            topk_weights, gating_output, moe_softcapping, correction_bias
        ):
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )
        if ctx is None:
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )
        if not _is_musa_launch_eligible(topk_weights, topk_ids, gating_output):
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )
        try:
            live_inner = getattr(kernel, "fn", None)
        except Exception:
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )
        if live_inner is not ctx["inner_fn"]:
            # Runtime object replacement: fail closed to the original path.
            return original(
                topk_weights,
                topk_ids,
                gating_output,
                renormalize,
                moe_softcapping,
                correction_bias,
            )

        return _pinned_launch_closed(
            ctx,
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
            moe_softcapping,
            correction_bias,
        )

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def apply_musa_topk_schedule_patch() -> bool:
    """Pin only the measured MP31 softmax-TopK shapes to one launch config."""

    if not _enabled():
        logger.info("MUSA TopK schedule disabled by %s", _ENV_NAME)
        return False

    device_name = _device_name()
    if "S5000" not in device_name.upper():
        logger.info("MUSA TopK schedule skipped on device %s", device_name)
        return False

    try:
        import triton
        from sglang.srt.hardware_backend.musa.kernels import topk as musa_topk
        from sglang.srt.layers.moe import topk as layer_topk
    except ImportError as exc:
        logger.warning("MUSA TopK schedule skipped: %s", exc)
        return False

    if getattr(musa_topk.topk_softmax, _PATCH_MARKER, False):
        return True

    selected_config = triton.Config({}, num_warps=1, num_stages=1)
    kernel = musa_topk.topk_softmax_triton_kernel
    if _verify_pinned_startup(kernel, selected_config) is None:
        logger.warning("MUSA TopK schedule skipped: pinned startup guard failed")
        return False
    wrapped = _make_topk_wrapper(
        musa_topk.topk_softmax,
        kernel,
        selected_config,
    )
    musa_topk.topk_softmax = wrapped
    # This alias may already have been imported before vendor patches run.
    layer_topk.topk_softmax = wrapped
    logger.info(
        "MUSA S5000 softmax TopK schedule enabled for E=256, K=8 graph shapes "
        "and 1K-16K prefill: warps=1, stages=1"
    )
    return True
