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

"""Guarded eager/prefill shared-expert combine through unified dispatch.

The framework owns model context and the original reduction/collective seams.
FlagGems-sglang owns arithmetic. Native decode graphs keep the original model
forward. Errors after submitting any shared or routed compute are propagated.
"""

from __future__ import annotations

import importlib
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any, Callable

import torch
import torch.nn.functional as F

from sglang_fl.dispatch import resolve_op

logger = logging.getLogger(__name__)


_ENV_NAME = "SGLANG_MUSA_DETERMINISTIC_MOE_COMBINE"
_DISABLED_VALUES = {"0", "false", "no", "off", "disable", "disabled"}
_PATCH_MARKER = "_sglang_fl_musa_deterministic_moe_combine"
_MODEL_CONTRACT_MARKER = "_sglang_fl_musa_deterministic_moe_contract"

_HIDDEN = 2048
_TOPK = 8
_NUM_EXPERTS = 256
_SUPPORTED_TOKENS = frozenset((2048, 4096, 6144, 8192, 16384))


@dataclass(frozen=True)
class _MoeModelContract:
    """Immutable constructor-time contract, independent of the config object."""

    model_type: str | None
    hidden_size: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    moe_intermediate_size: int | None
    shared_expert_intermediate_size: int | None
    matches: bool


def _typed_config_field(config: Any, name: str, expected_type: type) -> Any:
    """Read one config field, keeping exact type constraints and no coercion."""

    value = getattr(config, name, None)
    return value if type(value) is expected_type else None


def _model_contract_from_config(config: Any) -> _MoeModelContract:
    """Copy only primitive contract fields; never retain the config object.

    All config reads are captured at this one extraction boundary: a missing
    or unreadable field yields the non-matching contract, so an optional
    optimization probe can never become a model construction failure.  Exact
    field types are preserved and no implicit conversion is applied.
    """

    try:
        model_type = _typed_config_field(config, "model_type", str)
        hidden_size = _typed_config_field(config, "hidden_size", int)
        num_experts = _typed_config_field(config, "num_experts", int)
        num_experts_per_tok = _typed_config_field(config, "num_experts_per_tok", int)
        moe_intermediate_size = _typed_config_field(
            config, "moe_intermediate_size", int
        )
        shared_expert_intermediate_size = _typed_config_field(
            config, "shared_expert_intermediate_size", int
        )
    except Exception:  # noqa: BLE001 - unreadable config must fall back
        logger.debug(
            "MUSA deterministic combine config contract unavailable", exc_info=True
        )
        return _MoeModelContract(None, None, None, None, None, None, False)

    matches = (
        model_type in {"qwen3_5_moe_text", "qwen3_5_moe"}
        and hidden_size == _HIDDEN
        and num_experts == _NUM_EXPERTS
        and num_experts_per_tok == _TOPK
        and moe_intermediate_size == 512
        and shared_expert_intermediate_size == 512
    )
    return _MoeModelContract(
        model_type=model_type,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        matches=matches,
    )


def _init_config(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Get Qwen2MoeSparseMoeBlock's config from positional or keyword args."""

    if len(args) >= 2:
        return args[1]
    return kwargs.get("config")


def _clear_model_contract(instance: Any) -> None:
    try:
        delattr(instance, _MODEL_CONTRACT_MARKER)
    except AttributeError:
        pass


def _make_qwen_init(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap construction and publish a marker only after original init succeeds."""

    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(self, *args: Any, **kwargs: Any):
        # A repeated __init__ must not retain a marker from an earlier success
        # if the new constructor invocation fails.
        _clear_model_contract(self)
        result = original(self, *args, **kwargs)
        try:
            marker = _model_contract_from_config(_init_config(args, kwargs))
            setattr(self, _MODEL_CONTRACT_MARKER, marker)
        except Exception:
            _clear_model_contract(self)
            raise
        return result

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


@dataclass
class MoeCombineContext:
    """Per-forward shared-expert inputs and consumption state."""

    shared_unweighted: torch.Tensor
    gate_logits: torch.Tensor
    routed_scaling_factor: float = 1.0
    used: bool = False


_ACTIVE_CONTEXT: ContextVar[MoeCombineContext | None] = ContextVar(
    "sglang_fl_moe_combine_context", default=None
)
_SUCCESS_LOGGED: set[str] = set()
_SUCCESS_LOG_LOCK = Lock()
_DEVICE_NAME_CACHE: dict[tuple[str | None, str], str] = {}
_DEVICE_NAME_CACHE_LOCK = Lock()


def _enabled() -> bool:
    return os.environ.get(_ENV_NAME, "auto").strip().lower() not in _DISABLED_VALUES


def _distributed_rank() -> int:
    """Return the local process rank without making distributed a dependency."""

    try:
        distributed = getattr(torch, "distributed", None)
        if (
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        ):
            return int(distributed.get_rank())
    except Exception as exc:  # noqa: BLE001 - rank evidence must never affect execution
        logger.debug("MUSA deterministic MoE combine rank unavailable: %s", exc)
    return -1


def _log_success_once(path: str) -> None:
    """Emit one launch-submitted marker for each combine execution path.

    This records that the launch was submitted, not that asynchronous
    execution or graph replay has been verified.
    """

    with _SUCCESS_LOG_LOCK:
        if path in _SUCCESS_LOGGED:
            return
        _SUCCESS_LOGGED.add(path)
    logger.info(
        "MUSA deterministic MoE %s combine launch submitted: rank=%s",
        path,
        _distributed_rank(),
    )


def _device_cache_key(device: Any) -> tuple[str | None, str]:
    return device.type, str(device)


def _device_name(tensor: torch.Tensor) -> str:
    device = tensor.device
    if device.type != "musa":
        return ""
    cache_key = _device_cache_key(device)
    with _DEVICE_NAME_CACHE_LOCK:
        if cache_key in _DEVICE_NAME_CACHE:
            return _DEVICE_NAME_CACHE[cache_key]
        try:
            musa = getattr(torch, "musa", None)
            if musa is None or not musa.is_available():
                name = ""
            else:
                name = str(musa.get_device_name(device))
        except Exception:  # noqa: BLE001 - a missing runtime must fall back safely
            name = ""
        # Cache both successful and unknown results.  A runtime/query failure
        # must not become a per-layer host-side cost or repeatedly emit work.
        _DEVICE_NAME_CACHE[cache_key] = name
        return name


def _is_capture_or_graph(qwen2_module: Any, forward_batch: Any) -> bool:
    """Conservatively reject every known capture/graph indication."""

    capture_fn = getattr(qwen2_module, "get_is_capture_mode", None)
    if capture_fn is None:
        return True
    try:
        if bool(capture_fn()):
            return True
    except Exception:  # noqa: BLE001 - unknown runtime state is unsafe to patch
        return True

    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return False
    for name in ("is_cuda_graph", "is_piecewise_cuda_graph", "is_capture"):
        value = getattr(mode, name, None)
        if value is None:
            continue
        try:
            if bool(value() if callable(value) else value):
                return True
        except Exception:  # noqa: BLE001 - unknown mode is unsafe to patch
            return True
    return False


def _is_deepep(qwen2_module: Any) -> bool:
    backend_fn = getattr(qwen2_module, "get_moe_a2a_backend", None)
    if backend_fn is None:
        return True
    try:
        backend = backend_fn()
        return bool(backend.is_deepep())
    except Exception:  # noqa: BLE001 - unknown backend must use the old path
        return True


def _is_bf16(tensor: Any) -> bool:
    return tensor.dtype == torch.bfloat16


def _is_contiguous(tensor: Any) -> bool:
    return bool(tensor.is_contiguous())


def _same_device(*tensors: Any) -> bool:
    devices = [tensor.device for tensor in tensors]
    return bool(devices) and all(device == devices[0] for device in devices[1:])


def _is_tensor(value: Any) -> bool:
    """Single entry object-category check before direct tensor attribute reads.

    The contract paths read ``shape``/``dtype``/``device``/``is_contiguous``
    directly.  Non-tensor inputs are rejected here and keep the original path,
    instead of raising from an attribute read that sits outside the launch
    guard.
    """

    return isinstance(value, torch.Tensor)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _contract_matches(
    routed: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float | None,
    context: MoeCombineContext,
) -> bool:
    """Check the complete measured combine contract before launching Triton."""

    if not _enabled():
        return False
    if not (
        _is_tensor(routed)
        and _is_tensor(output)
        and _is_tensor(context.shared_unweighted)
        and _is_tensor(context.gate_logits)
    ):
        return False
    routed_shape = _shape(routed)
    output_shape = _shape(output)
    shared_shape = _shape(context.shared_unweighted)
    gate_shape = _shape(context.gate_logits)
    token_num = routed_shape[0] if routed_shape else -1
    if (
        token_num not in _SUPPORTED_TOKENS
        or routed_shape != (token_num, _TOPK, _HIDDEN)
        or output_shape != (token_num, _HIDDEN)
        or shared_shape != (token_num, _HIDDEN)
        or gate_shape != (token_num, 1)
    ):
        return False
    if context.used or context.routed_scaling_factor != 1.0:
        return False
    if routed_scaling_factor is not None:
        try:
            if float(routed_scaling_factor) != 1.0:
                return False
        except (TypeError, ValueError):
            return False
    if not all(
        _is_bf16(tensor) and _is_contiguous(tensor)
        for tensor in (
            routed,
            output,
            context.shared_unweighted,
            context.gate_logits,
        )
    ):
        return False
    if not _same_device(routed, output, context.shared_unweighted, context.gate_logits):
        return False
    if routed.device.type != "musa":
        return False
    return "S5000" in _device_name(routed).upper()


def _shared_context_matches(
    hidden_states: torch.Tensor,
    context: MoeCombineContext,
) -> bool:
    """Validate model-side shared tensors without allocating a routed buffer."""

    hidden_shape = _shape(hidden_states)
    token_num = hidden_shape[0] if hidden_shape else -1
    return (
        _shape(context.shared_unweighted) == (token_num, _HIDDEN)
        and _shape(context.gate_logits) == (token_num, 1)
        and _is_bf16(context.shared_unweighted)
        and _is_bf16(context.gate_logits)
        and _is_contiguous(context.shared_unweighted)
        and _is_contiguous(context.gate_logits)
        and _same_device(hidden_states, context.shared_unweighted, context.gate_logits)
    )


def _launch_candidate(routed, output, context):
    from flaggems_sglang import deterministic_moe_combine

    deterministic_moe_combine(
        routed, context.gate_logits, context.shared_unweighted, output
    )


def _wrap_moe_sum_reduce(original: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(routed, output, routed_scaling_factor, *args, **kwargs):
        return resolve_op("moe_sum_reduce")(
            original, routed, output, routed_scaling_factor, *args, **kwargs
        )

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def moe_sum_reduce_musa(
    original, routed, output, routed_scaling_factor, *args, **kwargs
):
    """Select before launch; execution errors propagate without retry."""
    context = _ACTIVE_CONTEXT.get()
    if context is None or not _contract_matches(
        routed, output, routed_scaling_factor, context
    ):
        return original(routed, output, routed_scaling_factor, *args, **kwargs)
    _launch_candidate(routed, output, context)
    context.used = True
    _log_success_once("eager/prefill")
    return None


def _extract_linear_output(value: Any) -> Any:
    if isinstance(value, tuple):
        return value[0]
    return value


def _model_contract_matches(
    qwen2_module: Any,
    module: Any,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    use_reduce_scatter: bool = False,
    should_allreduce_fusion: bool = False,
) -> bool:
    if not _enabled():
        return False
    if _is_capture_or_graph(qwen2_module, forward_batch):
        return False
    if _is_deepep(qwen2_module):
        return False
    # The tested combine writes the complete TP-local output.  Keep the
    # reduce-scatter variant on the original path until it has its own proof.
    # ``should_allreduce_fusion`` is supported: the candidate preserves the
    # in-place destination and lets the downstream custom-AR owner retain it.
    if use_reduce_scatter:
        return False
    if not _is_tensor(hidden_states):
        return False
    # Qwen3.5/3.6 constructs ``alt_stream`` on MUSA even for eager execution.
    # The framework only consumes it in the capture-mode branch, which is
    # rejected above. Presence of the handle alone must therefore not make
    # the measured eager path miss the contract.
    hidden_shape = _shape(hidden_states)
    if hidden_shape[1:] != (_HIDDEN,):
        return False
    if hidden_shape[0] not in _SUPPORTED_TOKENS:
        return False
    if not _is_bf16(hidden_states) or not _is_contiguous(hidden_states):
        return False
    if hidden_states.device.type != "musa":
        return False
    if "S5000" not in _device_name(hidden_states).upper():
        return False

    marker = getattr(module, _MODEL_CONTRACT_MARKER, None)
    if not isinstance(marker, _MoeModelContract) or not marker.matches:
        return False
    if getattr(module, "tp_size", None) != 2:
        return False
    if getattr(module, "num_experts", None) != _NUM_EXPERTS:
        return False
    if getattr(module, "num_shared_experts", 0) <= 0:
        return False
    if getattr(module, "num_fused_shared_experts", 0) != 0:
        return False
    if getattr(module, "enable_shared_expert_fusion", False):
        return False
    if getattr(module, "shared_expert", None) is None:
        return False
    if getattr(module, "shared_expert_gate", None) is None:
        return False

    topk_config = getattr(getattr(module, "topk", None), "topk_config", None)
    if getattr(topk_config, "top_k", None) != _TOPK:
        return False
    experts = getattr(module, "experts", None)
    if getattr(experts, "num_experts", None) != _NUM_EXPERTS:
        return False
    runner_config = getattr(experts, "moe_runner_config", None)
    if getattr(runner_config, "inplace", None) is not True:
        return False
    if getattr(runner_config, "no_combine", False):
        return False
    scale = getattr(runner_config, "routed_scaling_factor", None)
    if scale is not None:
        try:
            if float(scale) != 1.0:
                return False
        except (TypeError, ValueError):
            return False
    # The existing SGLang all-reduce combine path writes routed results with
    # BF16 atomics and bypasses ``moe_sum_reduce``.  Do not alter that path or
    # precompute shared tensors when it is selected; this candidate is only
    # the measured non-atomic reduction branch.
    server_args_fn = getattr(qwen2_module, "get_global_server_args", None)
    if server_args_fn is None:
        return False
    try:
        if bool(getattr(server_args_fn(), "enable_fused_moe_sum_all_reduce", False)):
            return False
    except Exception:  # noqa: BLE001 - unknown server mode must use old path
        return False
    return True


def _weighted_shared(context: MoeCombineContext) -> torch.Tensor:
    return F.sigmoid(context.gate_logits) * context.shared_unweighted


def _make_qwen_forward(
    qwen2_module: Any,
    original: Callable[..., Any],
) -> Callable[..., Any]:
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Any = None,
        use_reduce_scatter: bool = False,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        if not _model_contract_matches(
            qwen2_module,
            self,
            hidden_states,
            forward_batch,
            use_reduce_scatter,
            should_allreduce_fusion,
        ):
            return original(
                self,
                hidden_states,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )

        num_tokens, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        shared_unweighted = self.shared_expert(flat_hidden_states)
        shared_gate_logits = _extract_linear_output(
            self.shared_expert_gate(flat_hidden_states)
        )
        if not isinstance(shared_unweighted, torch.Tensor) or not isinstance(
            shared_gate_logits, torch.Tensor
        ):
            raise TypeError("shared expert/gate did not return tensors")
        context = MoeCombineContext(
            shared_unweighted=shared_unweighted,
            gate_logits=shared_gate_logits,
            routed_scaling_factor=1.0,
        )
        if not _shared_context_matches(flat_hidden_states, context):
            raise ValueError("shared expert context is outside combine contract")

        token = _ACTIVE_CONTEXT.set(context)
        try:
            final_hidden_states = self._forward_router_experts(flat_hidden_states)
        finally:
            # Never leave one layer's shared tensors visible to a later layer,
            # coroutine, or request after the router returns or raises.
            _ACTIVE_CONTEXT.reset(token)

        if not context.used:
            # Candidate guard miss: the original fused-MoE reduction has
            # already populated final_hidden_states, so preserve the exact
            # in-place shared-expert ownership contract.
            final_hidden_states += _weighted_shared(context)

        if self.tp_size > 1 and not qwen2_module.should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = qwen2_module.tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _restore_installation(
    fused_module: Any,
    qwen_cls: Any,
    old_fused: Any,
    old_init: Any,
    old_forward: Any,
) -> bool:
    """Restore the three pre-install objects and report full success.

    The restored values are exactly the objects that were live immediately
    before installation, including any wrapper already in place.  The code
    never unwraps through ``__wrapped__`` to an earlier function.
    """

    restored = True
    try:
        fused_module.moe_sum_reduce = old_fused
    except Exception:
        restored = False
        logger.warning(
            "MUSA deterministic combine failed to restore fused_moe.moe_sum_reduce",
            exc_info=True,
        )
    try:
        qwen_cls.__init__ = old_init
    except Exception:
        restored = False
        logger.warning(
            "MUSA deterministic combine failed to restore Qwen2MoeSparseMoeBlock.__init__",
            exc_info=True,
        )
    try:
        qwen_cls.forward = old_forward
    except Exception:
        restored = False
        logger.warning(
            "MUSA deterministic combine failed to restore Qwen2MoeSparseMoeBlock.forward",
            exc_info=True,
        )
    return restored


def apply_musa_deterministic_moe_combine_patch() -> bool:
    """Apply the idempotent local combine patch when the MUSA runtime exists."""

    if not _enabled():
        logger.info("MUSA deterministic MoE combine disabled by %s", _ENV_NAME)
        return False
    if not hasattr(torch, "musa"):
        logger.info("MUSA deterministic MoE combine skipped: torch.musa unavailable")
        return False

    # Snapshot all three seams so a failed import/assignment cannot expose a
    # half-patched product path.  Existing wrappers are restored as-is, which
    # keeps repeated application idempotent.
    try:
        fused_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
        )
        qwen_module = importlib.import_module("sglang.srt.models.qwen2_moe")
        qwen_cls = qwen_module.Qwen2MoeSparseMoeBlock
        old_fused = fused_module.moe_sum_reduce
        old_init = qwen_cls.__init__
        old_forward = qwen_cls.forward
    except (ImportError, AttributeError) as exc:
        logger.warning("MUSA deterministic combine patch targets unavailable: %s", exc)
        return False

    # Build every wrapper before touching an install point.  The wrapper
    # builders are idempotent via their own marker check, so repeated
    # installation returns the existing wrappers and never adds a layer.
    try:
        new_fused = _wrap_moe_sum_reduce(old_fused)
        new_init = _make_qwen_init(old_init)
        new_forward = _make_qwen_forward(qwen_module, old_forward)
    except Exception as exc:
        logger.warning("MUSA deterministic combine wrapper build failed: %s", exc)
        return False

    # Publish the three wrappers together; restore the exact pre-install
    # objects if any assignment fails.
    try:
        fused_module.moe_sum_reduce = new_fused
        qwen_cls.__init__ = new_init
        qwen_cls.forward = new_forward
    except Exception as exc:
        if _restore_installation(
            fused_module, qwen_cls, old_fused, old_init, old_forward
        ):
            logger.warning(
                "MUSA deterministic combine rolled back after install error: %s",
                exc,
            )
        else:
            logger.error(
                "MUSA deterministic combine install failed and rollback was "
                "incomplete: %s",
                exc,
            )
        return False

    logger.info("MUSA eager/prefill deterministic MoE combine installed")
    return True


__all__ = [
    "MoeCombineContext",
    "apply_musa_deterministic_moe_combine_patch",
]
