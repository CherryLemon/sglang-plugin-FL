# SPDX-License-Identifier: Apache-2.0
"""Opt-in HY4 attention gate orchestration; all numerical code is in FlagGems.

The environment is read once at model construction. Dynamo/BCG, prefill,
other roles/providers/layouts and large M retain the original linear call.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import torch

_QUALIFIED_M = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class GatePolicy:
    enabled: bool


@lru_cache(maxsize=1)
def construction_policy():
    return GatePolicy(enabled=os.environ.get("HY4_SG_SMALL_M_BF16_GATE", "0") == "1")


def _provider_key(layer):
    cls = type(getattr(layer, "quant_method", None))
    return (cls.__module__, cls.__name__)


def _layer_contract(layer, prefix):
    cls = type(layer)
    return (
        prefix.endswith(".self_attn.linear_gate")
        and cls.__module__ == "sglang.srt.layers.linear"
        and (cls.__name__ == "ColumnParallelLinear")
        and (getattr(layer, "prefix", prefix) == prefix)
        and (getattr(layer, "input_size", None) == 6144)
        and (getattr(layer, "output_size_per_partition", None) == 1024)
        and (not getattr(layer, "gather_output", True))
        and (not getattr(layer, "skip_bias_add", True))
        and (getattr(layer, "bias", None) is None)
        and (
            _provider_key(layer)
            == ("sglang.srt.layers.quantization.unquant", "UnquantizedLinearMethod")
        )
    )


class SmallMGateProjector:
    def __init__(self, layer, prefix, sm, compute, supports, is_bcg):
        self.layer = layer
        self.prefix = prefix
        self.device_sm = sm
        self.provider_key = _provider_key(layer)
        self.compute = compute
        self.supports = supports
        self.is_bcg = is_bcg

    def maybe_project(self, hidden_states, forward_batch):
        if torch.compiler.is_compiling() or self.is_bcg() or forward_batch is None:
            return None
        mode = forward_batch.forward_mode
        if not (
            mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2()
        ):
            return None
        layer = self.layer
        weight = getattr(layer, "weight", None)
        if (
            not _layer_contract(layer, self.prefix)
            or _provider_key(layer) != self.provider_key
        ):
            return None
        if weight is None or hidden_states.ndim != 2 or weight.ndim != 2:
            return None
        if (
            getattr(hidden_states, "fake_mode", None) is not None
            or getattr(weight, "fake_mode", None) is not None
        ):
            return None
        selected = (
            hidden_states.shape[0] in _QUALIFIED_M
            and hidden_states.device.type == "cuda"
            and (weight.device == hidden_states.device)
            and (weight.shape == (1024, 6144))
            and self.supports(
                m=hidden_states.shape[0],
                n=weight.shape[0],
                k=hidden_states.shape[1],
                input_dtype=str(hidden_states.dtype).removeprefix("torch."),
                weight_dtype=str(weight.dtype).removeprefix("torch."),
                input_stride=hidden_states.stride(),
                weight_stride=weight.stride(),
                device_sm=self.device_sm,
                bias_present=layer.bias is not None,
                same_device=True,
            )
        )
        if not selected:
            return None
        return self.compute(hidden_states, weight, device_sm=self.device_sm)


def make_small_m_gate_projector(layer, prefix):
    policy = construction_policy()
    if not policy.enabled or not _layer_contract(layer, prefix):
        return None
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )
    from sglang.srt.runtime_context import get_lora
    from sglang.srt.utils import get_device_capability

    major, minor = get_device_capability()
    if (
        (major, minor) != (9, 0)
        or get_lora().enable_lora
        or (not check_cuda_graph_backend(Phase.DECODE, Backend.FULL))
    ):
        return None
    from flag_gems.fused.bf16_small_m_linear import (
        bf16_small_m_linear,
        supported_metadata,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    return SmallMGateProjector(
        layer,
        prefix,
        90,
        bf16_small_m_linear,
        supported_metadata,
        is_in_breakable_cuda_graph,
    )
