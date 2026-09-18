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

"""Enable-policy tests for the vendor-neutral stream-completion patch."""

import sys
import types

import pytest

from sglang_fl.runtime.stream_completion import policy


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(policy.ENV_NAME, raising=False)
    monkeypatch.delenv(policy.LEGACY_ENV_NAME, raising=False)


def _inject_policy(monkeypatch, vendor, auto_enable):
    module = types.ModuleType(
        f"sglang_fl.dispatch.backends.vendor.{vendor}.eventfd_policy"
    )
    module.auto_enable = auto_enable
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_auto_is_off_without_vendor_hook(monkeypatch):
    monkeypatch.setattr(policy, "resolve_vendor", lambda: "nvidia")
    assert not policy.is_enabled()


def test_auto_uses_vendor_hook(monkeypatch):
    monkeypatch.setattr(policy, "resolve_vendor", lambda: "fakevendor")
    _inject_policy(monkeypatch, "fakevendor", lambda: True)
    assert policy.is_enabled()

    _inject_policy(monkeypatch, "fakevendor", lambda: False)
    assert not policy.is_enabled()


def test_on_overrides_missing_auto_policy(monkeypatch):
    monkeypatch.setenv(policy.ENV_NAME, "on")
    monkeypatch.setattr(policy, "resolve_vendor", lambda: None)
    assert policy.is_enabled()


def test_explicit_off_wins_over_vendor_hook(monkeypatch):
    monkeypatch.setenv(policy.ENV_NAME, "off")
    monkeypatch.setattr(policy, "resolve_vendor", lambda: "fakevendor")
    _inject_policy(monkeypatch, "fakevendor", lambda: True)
    assert not policy.is_enabled()


def test_legacy_musa_env_is_honoured(monkeypatch):
    monkeypatch.setenv(policy.LEGACY_ENV_NAME, "1")
    monkeypatch.setattr(policy, "resolve_vendor", lambda: None)
    assert policy.is_enabled()


def test_invalid_value_raises(monkeypatch):
    monkeypatch.setenv(policy.ENV_NAME, "sometimes")
    with pytest.raises(ValueError):
        policy.is_enabled()
