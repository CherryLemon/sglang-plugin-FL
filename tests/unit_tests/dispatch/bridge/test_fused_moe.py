from types import SimpleNamespace

import torch

import sglang_fl.dispatch.bridge.fused_moe as bridge


def test_fused_moe_bridge_forwards_layer_and_dispatch_output(monkeypatch) -> None:
    obj = SimpleNamespace()
    layer = torch.nn.Linear(4, 4)
    dispatch_output = object()
    expected = object()
    calls = []

    def selected(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        bridge, "resolve_op", lambda name: selected if name == "fused_moe" else None
    )

    result = bridge.fused_moe_bridge(obj, layer, dispatch_output)

    assert result is expected
    assert calls == [((obj, layer, dispatch_output), {})]


def test_execution_failure_does_not_retry_another_backend(monkeypatch):
    import os

    import pytest

    from sglang_fl.dispatch.manager import OpManager
    from sglang_fl.dispatch.policy import SelectionPolicy, policy_context
    from sglang_fl.dispatch.types import BackendImplKind, OpImpl

    manager = OpManager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    hidden = torch.zeros(1)
    error = RuntimeError("failed after mutation")
    calls = []

    def candidate(*args):
        calls.append("candidate")
        hidden.fill_(7)
        raise error

    def alternative(*args):
        calls.append("alternative")
        hidden.zero_()

    for name, fn, priority in (("first", candidate, 100), ("second", alternative, 0)):
        manager.registry.register_impl(
            OpImpl(
                op_name="fused_moe",
                impl_id=name,
                kind=BackendImplKind.DEFAULT,
                fn=fn,
                priority=priority,
            )
        )
    monkeypatch.setattr(bridge, "resolve_op", manager.resolve)
    with policy_context(SelectionPolicy.from_dict(strict=False)), pytest.raises(
        RuntimeError
    ) as caught:
        bridge.fused_moe_bridge(object(), object(), object())
    assert caught.value is error
    assert calls == ["candidate"]
    assert hidden.item() == 7
