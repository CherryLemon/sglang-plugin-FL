# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Register model-specific hooks before worker initialization and Graph capture."""

from types import FunctionType

_registered = False


class _NativeProtocolView:
    def __init__(self, config):
        self._config = config

    @property
    def disaggregation_transfer_backend(self):
        return "mooncake"

    def __getattr__(self, name):
        return getattr(self._config, name)


def _validate_features(original, server_args):
    # FlagCX reuses the native control protocol. Reuse its model validation
    # with a local protocol view; never change the actual transport setting,
    # the module's resolving_view or any ServerArgs/resolution state.
    resolving_view = original.__globals__["resolving_view"]
    if resolving_view(server_args).disaggregation_transfer_backend != "flagcx":
        return original(server_args)

    def protocol_view(args):
        config = resolving_view(args)
        if config.disaggregation_transfer_backend == "flagcx":
            return _NativeProtocolView(config)
        return config

    scope = dict(original.__globals__, resolving_view=protocol_view)
    validate = FunctionType(
        original.__code__,
        scope,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    validate.__kwdefaults__ = original.__kwdefaults__
    return validate(server_args)


def apply_dsv41_patches():
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from sglang_fl.disaggregation.patch import apply_disaggregation_patch

    from .prefill import candidate_block_ids, candidate_blocks, extend_indexer

    apply_disaggregation_patch()
    indexer = "sglang.srt.layers.attention.dsv4.indexer"
    HookRegistry.register(
        f"{indexer}.select_candidate_blocks", candidate_blocks, HookType.REPLACE
    )
    HookRegistry.register(
        f"{indexer}.select_candidate_block_ids", candidate_block_ids, HookType.REPLACE
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.deepseek_v4_backend."
        "DeepseekV4AttnBackend._low_ratio_index_topk_sm90_extend",
        extend_indexer,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v41_features",
        _validate_features,
        HookType.AROUND,
    )
    _registered = True
