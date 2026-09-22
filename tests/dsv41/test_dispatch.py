from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dsv41.backend import Capabilities, Dsv41Backend


def test_launch_failure_never_retries_vendor():
    backend = Dsv41Backend("hybrid", capabilities=Capabilities("cpu"))
    calls = []

    def failed(*args):
        raise RuntimeError("kernel failure after launch")

    backend._implementations["linear.block_fp8"] = failed
    with pytest.raises(RuntimeError, match="kernel failure after launch"):
        backend("linear.block_fp8", lambda *a: calls.append("vendor"))
    assert calls == []
    assert backend.snapshot()["selected"] == {"linear.block_fp8": "flaggems"}


def test_vendor_allowlist_and_strict_mode():
    backend = Dsv41Backend("hybrid", capabilities=Capabilities("cpu"))
    assert backend("attention.sparse", lambda x: x, 17) == 17
    with pytest.raises(KeyError):
        backend("new.unreviewed_op", lambda: 1)
    strict = Dsv41Backend("flagos", capabilities=Capabilities("cpu"))
    with pytest.raises(NotImplementedError, match="attention.sparse"):
        strict("attention.sparse", lambda: pytest.fail("vendor called"))


@pytest.mark.parametrize("mode", ["flagos", "reference"])
def test_fixture_profiles_do_not_admit_whole_models(mode):
    backend = Dsv41Backend(
        mode, capabilities=Capabilities("cuda", True, False, True, True)
    )
    with pytest.raises(RuntimeError, match="operator-fixture"):
        backend.validate_model(SimpleNamespace(model_type="deepseek_v41"))


def test_bf16_and_non_cuda_models_fail_before_allocating_weights():
    for backend in [
        Dsv41Backend(
            "hybrid", "bf16", capabilities=Capabilities("cuda", True, False, True, True)
        ),
        Dsv41Backend("hybrid", capabilities=Capabilities("cpu")),
    ]:
        with pytest.raises(RuntimeError):
            backend.validate_model(SimpleNamespace(model_type="deepseek_v41"))


def test_reference_linear_uses_engine_argument_order():
    backend = Dsv41Backend("reference", capabilities=Capabilities("cpu"))
    x = torch.ones((1, 32), dtype=torch.bfloat16)
    w = torch.ones((2, 32)).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0]])
    out = backend(
        "linear.block_fp8",
        None,
        x,
        w,
        [32, 32],
        scale,
        None,
        torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        True,
    )
    torch.testing.assert_close(
        out, torch.tensor([[65.0, 66.0]], dtype=torch.bfloat16), rtol=0, atol=0
    )


def test_engine_abi_and_required_registration(monkeypatch):
    from sglang.srt.layers import dsv41_ops

    monkeypatch.setattr(dsv41_ops, "_dispatch", None)
    monkeypatch.setenv("SGLANG_FL_DSV41_REQUIRED", "1")

    @dsv41_ops.dsv41_op("linear.block_fp8")
    def native(x):
        return x

    with pytest.raises(RuntimeError, match="initialization"):
        native(1)
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        dsv41_ops.register_backend(lambda *a: None, abi_version=999)
    dsv41_ops.register_backend(
        lambda name, fn, *a, **kw: fn(*a, **kw) + 1, abi_version=1
    )
    assert native(1) == 2


def test_mhc_keeps_predecessor_pre_and_returns_new_coefficients(monkeypatch):
    import sys
    from sglang_fl.dsv41.backend import _mix_and_combine

    new_pre = torch.tensor([[7.0, 8.0]])
    old_pre = torch.tensor([[2.0, 3.0]])
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    def combine(flat, pre, hc, dtype):
        assert pre is old_pre
        return (flat.reshape(1, hc, -1) * pre[..., None]).sum(1).to(dtype)

    monkeypatch.setitem(
        sys.modules,
        "flag_gems.fused.dsv41.mhc",
        SimpleNamespace(
            hc_combine=combine, hc_mix_stats_sinkhorn=lambda *a: (new_pre, None, None)
        ),
    )
    layer = SimpleNamespace(
        hc_mult=2, hc_sinkhorn_iters=20, rms_norm_eps=1e-6, hc_eps=1e-6
    )
    y, pre, _, _ = _mix_and_combine(layer, x, None, None, None, old_pre, lambda a: a)
    assert pre is new_pre
    torch.testing.assert_close(y, torch.tensor([[11.0, 16.0]]))
