# SPDX-License-Identifier: Apache-2.0
"""HYV4 registration and narrowly scoped v0.5.18 compatibility hooks."""

import sys

_registered = False


def _hy4(config):
    return getattr(config, "model_type", None) in ("hy_v4", "hy_v4_vl")


def _replace_imported_function(module, name, replacement):
    previous = getattr(module, name)
    setattr(module, name, replacement)
    for loaded in tuple(sys.modules.values()):
        if loaded is None or not getattr(loaded, "__name__", "").startswith("sglang."):
            continue
        for key, value in tuple(vars(loaded).items()):
            if value is previous:
                setattr(loaded, key, replacement)


def register_hy_v4():
    global _registered
    if _registered:
        return
    from sglang.srt.configs import model_config as mc
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.models.registry import ModelRegistry
    from transformers import AutoConfig
    from .config import HYV4Config, HYV4VLConfig
    from .model import HYV4ForCausalLM
    from .nextn import HYV4ForCausalLMNextN
    from .nextn_utils import configure_hy4_draft, configure_hy4_hidden_shapes
    from .vl_model import HYV4VLForConditionalGeneration

    AutoConfig.register("hy_v4", HYV4Config, exist_ok=True)
    AutoConfig.register("hy_v4_vl", HYV4VLConfig, exist_ok=True)
    ModelRegistry.models.update(
        {
            "HYV4ForCausalLM": HYV4ForCausalLM,
            "HYV4VLForConditionalGeneration": HYV4VLForConditionalGeneration,
            "HYV4ForCausalLMNextN": HYV4ForCausalLMNextN,
        }
    )
    previous_draft = mc.ModelConfig._config_draft_model

    def config_draft(self):
        previous_draft(self)
        configure_hy4_draft(self)

    mc.ModelConfig._config_draft_model = config_draft
    previous_dsa = mc.is_deepseek_dsa
    previous_skip = mc.dsa_layer_skips_topk

    def is_dsa(config):
        return _hy4(config) or previous_dsa(config)

    def skips_topk(config, layer_id):
        if _hy4(config):
            return (
                0 <= layer_id < config.num_hidden_layers
                and config.indexer_types[layer_id] == "shared"
            )
        return previous_skip(config, layer_id)

    _replace_imported_function(mc, "is_deepseek_dsa", is_dsa)
    _replace_imported_function(mc, "dsa_layer_skips_topk", skips_topk)
    previous_shapes = mc.ModelConfig._derive_model_shapes

    def derive_shapes(self):
        previous_shapes(self)
        if _hy4(self.hf_config):
            config = self.hf_text_config
            self.attention_arch = mc.AttentionArch.MLA
            self.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            for key in (
                "kv_lora_rank",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
                "index_head_dim",
            ):
                setattr(self, key, getattr(config, key))
            self._init_mla_scaling(config.rope_parameters)
            configure_hy4_hidden_shapes(self)

    mc.ModelConfig._derive_model_shapes = derive_shapes
    previous_backend = ATTENTION_BACKENDS["dsa"]

    def create_backend(runner):
        if _hy4(runner.model_config.hf_config):
            from .attention_backend import HYV4SparseAttnBackend

            return HYV4SparseAttnBackend(runner)
        return previous_backend(runner)

    ATTENTION_BACKENDS["dsa"] = create_backend
    _registered = True
    import os
    from .mrope_compat import install

    install(enabled=os.environ.get("SGLANG_HY4_MROPE_COMPAT", "0") == "1")
    from .draft_attention_backend import install as install_draft_sink_backend

    install_draft_sink_backend(
        enabled=os.environ.get("SGLANG_HY4_DRAFT_SINK_BACKEND", "0") == "1"
    )
    from .draft_causal_lengths import install as install_draft_causal_lengths

    install_draft_causal_lengths(
        enabled=os.environ.get("SGLANG_HY4_DRAFT_CAUSAL_LENGTHS", "0") == "1"
    )
