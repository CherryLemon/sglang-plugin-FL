# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; these tests do not initialize SGLang or a CUDA device."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL = Path(__file__).resolve().parents[3] / "sglang_fl/models/hy_v4"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"hy4_contract_{name}", MODEL / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "role,expected",
    [
        (False, "target"),
        (True, "draft"),
        (None, "unknown"),
        (0, "unknown"),
        (1, "unknown"),
    ],
)
def test_role_is_per_runner_and_strict_bool(role, expected):
    policy = load("attention_policy")
    runner = SimpleNamespace(is_draft_worker=role)
    env = {policy.TARGET_FLAG: "1", policy.DRAFT_FLAG: "1"}
    assert policy.freeze_role_policy(runner, env) == (expected, expected != "unknown")
    assert policy.freeze_role_policy(runner, {}) == (expected, False)


@pytest.mark.parametrize(
    "mode,accepted",
    [
        ("DECODE", True),
        ("TARGET_VERIFY", True),
        ("EXTEND", False),
        ("DRAFT_EXTEND_V2", False),
    ],
)
def test_prefill_and_draft_extend_remain_native(mode, accepted):
    policy = load("attention_policy")
    backend = SimpleNamespace(
        _hy4_mla_opted_in=True,
        _hy4_mla_role="target",
        _hy4_forward_mode=SimpleNamespace(name=mode),
    )
    assert policy.eligible_forward(backend) is accepted
    backend._hy4_mla_opted_in = False
    assert not policy.eligible_forward(backend)


def test_nextn_keeps_target_checkpoint_depth():
    mod = load("nextn_utils")
    text = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=78)
    config = SimpleNamespace(
        model_type="hy_v4_vl", architectures=["HYV4VLForConditionalGeneration"]
    )
    runner = SimpleNamespace(
        is_draft_model=True,
        hf_config=config,
        hf_text_config=text,
        hidden_size=6144,
        num_nextn_predict_layers=1,
    )
    mod.configure_hy4_draft(runner)
    mod.configure_hy4_hidden_shapes(runner)
    assert config.architectures == ["HYV4ForCausalLMNextN"]
    assert text.num_hidden_layers == 78
    assert runner.num_hidden_layers == runner.num_attention_layers == 1
    assert runner.spec_hidden_size == 6144
    assert runner.hc_hidden_size is None


@pytest.mark.parametrize("layers", [0, 2])
def test_nextn_rejects_missing_or_multiple_mtp(layers):
    mod = load("nextn_utils")
    runner = SimpleNamespace(
        is_draft_model=True,
        hf_config=SimpleNamespace(
            model_type="hy_v4", architectures=["HYV4ForCausalLM"]
        ),
        hf_text_config=SimpleNamespace(num_nextn_predict_layers=layers),
    )
    with pytest.raises(ValueError):
        mod.configure_hy4_draft(runner)


class QuantConfig:
    target_scheme_map = None
    sparsity_scheme_map = None
    sparsity_ignore_list = None

    def get_name(self):
        return "compressed_tensors"


def test_ct_aliases_copy_without_mutating_target():
    mod = load("nextn_utils")
    original = QuantConfig()
    original.target_scheme_map = {
        "model.mtp_layers.0.eh_proj": {"weight": "int8"},
        "unrelated": {"weight": "bf16"},
    }
    original.ignore = ["model.mtp_layers.0.enorm"]
    draft = mod.remap_hy4_mtp_quant_config(original)
    assert draft.target_scheme_map["model.eh_proj"] == {"weight": "int8"}
    assert "model.enorm" in draft.ignore
    assert "model.eh_proj" not in original.target_scheme_map
    assert original.ignore == ["model.mtp_layers.0.enorm"]
    draft.target_scheme_map["model.eh_proj"]["weight"] = "changed"
    assert original.target_scheme_map["model.mtp_layers.0.eh_proj"]["weight"] == "int8"


def test_only_bundled_mtp_weights_are_forwarded():
    mod = load("nextn_utils")
    a, b, unrelated = object(), object(), object()
    values = list(
        mod.iter_hy4_nextn_weights(
            [
                ("model.mtp_layers.0.eh_proj.weight", a),
                ("model.mtp_layers.0.final_layernorm.weight", b),
                ("model.layers.0.weight", unrelated),
                ("model.mtp_layers.1.weight", unrelated),
            ],
            78,
        )
    )
    assert values == [
        ("model.layers.78.eh_proj.weight", a),
        ("model.layers.78.shared_head.norm.weight", b),
    ]


def test_indexer_mm_cpu_fallback_preserves_call_and_math():
    mod = load("indexer_mm")
    seen = []

    def native(x, *args, **kwargs):
        seen.append((x, args, kwargs))
        return "native"

    layer = SimpleNamespace(
        weight=torch.zeros(4096, 2048, dtype=torch.bfloat16),
        _hy4_mm_enabled=True,
        _hy4_mm_sm90=True,
        bias=None,
        skip_bias_add=False,
        _hy4_mm_native_forward=native,
    )
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    value = torch.zeros(1, 2048, dtype=torch.bfloat16)
    assert mod._forward(layer, value, tag="retain") == "native"
    assert seen == [(value, (), {"tag": "retain"})]
    assert flags == (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )


def test_hpc_absence_uses_explicit_model_fallback(monkeypatch):
    mod = load("ihc_kernels")
    monkeypatch.setattr(mod, "_hpc_ihc_op", lambda *args: None)
    with pytest.raises(NotImplementedError):
        mod.fused_hy4_ihc_post(
            torch.zeros(1, 4), torch.zeros(1, 4, 4), torch.zeros(1, 4)
        )


def test_checkpoint_wrapper_config_is_explicitly_text_only():
    mod = load("config")
    config = mod.HYV4VLConfig(
        text_config={"architectures": ["HYV4ForCausalLM"]},
        vision_config={"unused_metadata": 7},
    )
    assert config.architectures == ["HYV4VLForConditionalGeneration"]
    assert config.language_model_only is True
    assert config.text_config.architectures == ["HYV4ForCausalLM"]
    assert config.vision_config == {"unused_metadata": 7}
    assert isinstance(config.vision_config, dict)


def test_checkpoint_wrapper_rejects_media_enable():
    mod = load("config")
    with pytest.raises(ValueError, match="text-only"):
        mod.HYV4VLConfig(language_model_only=False)
