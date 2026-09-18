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

from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads.patches import topk_schedule


def _tensor_shape(*shape):
    return SimpleNamespace(ndim=len(shape), shape=shape)


def test_topk_schedule_is_enabled_by_default_but_can_be_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_MUSA_TOPK_SCHEDULE", raising=False)
    assert topk_schedule._enabled()

    monkeypatch.setenv("SGLANG_MUSA_TOPK_SCHEDULE", "off")
    assert not topk_schedule._enabled()


def test_target_shape_requires_measured_dimensions():
    weights = _tensor_shape(64, 8)
    assert topk_schedule._is_target_shape(weights, _tensor_shape(64, 256), 0, None)
    assert not topk_schedule._is_target_shape(weights, _tensor_shape(65, 256), 0, None)
    assert topk_schedule._is_target_shape(
        _tensor_shape(4095, 8), _tensor_shape(4095, 256), 0, None
    )
    assert topk_schedule._is_target_shape(
        _tensor_shape(15360, 8), _tensor_shape(15360, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(
        _tensor_shape(16385, 8), _tensor_shape(16385, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(weights, _tensor_shape(64, 128), 0, None)
    assert not topk_schedule._is_target_shape(
        _tensor_shape(64, 4), _tensor_shape(64, 256), 0, None
    )
    assert not topk_schedule._is_target_shape(
        weights, _tensor_shape(64, 256), 1.0, None
    )
    assert not topk_schedule._is_target_shape(
        weights, _tensor_shape(64, 256), 0, object()
    )


def test_fn_run_signature_ok_accepts_jit_shape_and_rejects_fakes():
    class _Good:
        def run(self, *args, grid, warmup, **kwargs):
            return None

    class _MissingWarmup:
        def run(self, *args, grid, **kwargs):
            return None

    class _NoVarKw:
        def run(self, *args, grid, warmup):
            return None

    assert topk_schedule._fn_run_signature_ok(_Good().run) is True
    assert topk_schedule._fn_run_signature_ok(_MissingWarmup().run) is False
    assert topk_schedule._fn_run_signature_ok(_NoVarKw().run) is False
    assert topk_schedule._fn_run_signature_ok(lambda: None) is False


def test_fn_run_signature_bind_rejects_positional_grid_warmup():
    class _PositionalGridWarmup:
        def run(self, grid, warmup, *args, **kwargs):
            return None

    assert topk_schedule._fn_run_signature_ok(_PositionalGridWarmup().run) is False


def test_fn_run_signature_bind_rejects_extra_required():
    class _ExtraRequired:
        def run(self, *args, grid, warmup, required, **kwargs):
            return None

    assert topk_schedule._fn_run_signature_ok(_ExtraRequired().run) is False


def test_verify_pinned_startup_rejects_unknown_fake_kernel():
    fake_kernel = SimpleNamespace(configs=[object()], fn=SimpleNamespace(run=lambda: None))
    assert topk_schedule._verify_pinned_startup(fake_kernel, object()) is None

    triton = pytest.importorskip("triton")
    real_selected = triton.Config({}, num_warps=1, num_stages=1)
    assert topk_schedule._verify_pinned_startup(fake_kernel, real_selected) is None


def _target_tensors():
    return _tensor_shape(64, 8), object(), _tensor_shape(64, 256)


def test_wrapper_falls_back_when_startup_not_verified(monkeypatch):
    configs = [object(), object()]
    kernel = SimpleNamespace(configs=configs, fn=SimpleNamespace(run=lambda: None))
    monkeypatch.setattr(topk_schedule, "_verify_pinned_startup", lambda k, s: None)

    seen = []

    def original(*args):
        seen.append(args)
        return "result"

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, object())
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result == "result"
    assert len(seen) == 1
    assert kernel.configs is configs


def test_wrapper_falls_back_on_runtime_fn_replacement(monkeypatch):
    configs = [object(), object()]

    class _FakeInner:
        def run(self, *args, grid=None, warmup=None, **kwargs):
            raise AssertionError("replaced inner must not run")

    inner = _FakeInner()
    kernel = SimpleNamespace(configs=configs, fn=inner)
    ctx = {"inner_fn": inner, "pinned_kwargs": {"num_warps": 1, "num_ctas": 1, "num_stages": 1}}
    monkeypatch.setattr(topk_schedule, "_verify_pinned_startup", lambda k, s: ctx)
    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", lambda *a: True)

    seen = []

    def original(*args):
        seen.append(args)
        return "fallback"

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, object())
    # Runtime replacement after wrapper creation must fail closed.
    kernel.fn = SimpleNamespace(run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result == "fallback"
    assert len(seen) == 1
    assert kernel.configs is configs


def test_wrapper_falls_back_for_non_musa_tensors():
    configs = [object(), object()]
    kernel = SimpleNamespace(configs=configs, fn=SimpleNamespace(run=lambda: None))

    seen = []

    def original(*args):
        seen.append(args)
        return "fallback"

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, object())
    # Plain shape-only fakes are not MUSA torch tensors, so the device
    # guard fails closed to the original path (no triton needed).
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result == "fallback"
    assert len(seen) == 1
    assert kernel.configs is configs


def test_wrapper_leaves_unmeasured_shape_on_autotuner():
    configs = [object(), object()]
    kernel = SimpleNamespace(configs=configs)
    seen = []

    def original(*args):
        seen.append(kernel.configs)

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, object())
    wrapped(_tensor_shape(65, 8), object(), _tensor_shape(65, 256))

    assert seen == [configs]
    assert kernel.configs is configs


def test_inner_jit_abi_ok_with_fakes():
    abi_names = list(topk_schedule._EXPECTED_JIT_ARG_NAMES)
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=abi_names)) is True
    swapped = list(abi_names)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=swapped)) is False
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=abi_names[:-1])) is False
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace(arg_names=abi_names + ["EXTRA"])) is False
    assert topk_schedule._inner_jit_abi_ok(SimpleNamespace()) is False
    assert topk_schedule._inner_jit_abi_ok(object()) is False
    assert topk_schedule._inner_jit_abi_ok(None) is False


def test_wrapper_passthrough_with_mappingproxy_ctx(monkeypatch):
    from types import MappingProxyType

    configs = [object(), object()]
    configs_before = list(configs)
    calls = {}

    class _FakeInner:
        def run(self, *args, grid=None, warmup=None, **kwargs):
            calls["args"] = args
            calls["grid"] = grid
            calls["warmup"] = warmup
            calls["kwargs"] = kwargs
            return None

    inner = _FakeInner()
    kernel = SimpleNamespace(configs=configs, fn=inner)
    ctx = {
        "inner_fn": inner,
        "pinned_kwargs": MappingProxyType({"num_warps": 1, "num_ctas": 1, "num_stages": 1}),
    }
    monkeypatch.setattr(topk_schedule, "_verify_pinned_startup", lambda k, s: ctx)
    monkeypatch.setattr(topk_schedule, "_is_musa_launch_eligible", lambda *a: True)

    def original(*args):
        raise AssertionError("original must not be called on the pinned path")

    wrapped = topk_schedule._make_topk_wrapper(original, kernel, object())
    weights, ids, gating = _target_tensors()
    result = wrapped(weights, ids, gating)

    assert result is None
    assert calls["grid"] == (64,)
    assert calls["warmup"] is False
    assert calls["kwargs"]["K"] == 8
    assert calls["kwargs"]["num_warps"] == 1
    assert calls["kwargs"]["num_ctas"] == 1
    assert calls["kwargs"]["num_stages"] == 1
    assert kernel.configs is configs
    assert configs == configs_before
