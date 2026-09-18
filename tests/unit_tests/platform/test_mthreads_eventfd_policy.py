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

"""Auto-enable policy test for MUSA stream completion."""

from types import SimpleNamespace

import torch

from sglang_fl.dispatch.backends.vendor.mthreads import eventfd_policy


def test_auto_enable_only_on_s5000(monkeypatch):
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(get_device_name=lambda: "MTT S5000"),
        raising=False,
    )
    assert eventfd_policy.auto_enable()

    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(get_device_name=lambda: "MTT S4000"),
        raising=False,
    )
    assert not eventfd_policy.auto_enable()


def test_auto_enable_is_false_when_probe_fails(monkeypatch):
    def _boom():
        raise RuntimeError("no musa")

    monkeypatch.setattr(
        torch, "musa", SimpleNamespace(get_device_name=_boom), raising=False
    )
    assert not eventfd_policy.auto_enable()
