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

"""Unit tests for the vendor-neutral stream-completion provider."""

from types import SimpleNamespace

from sglang_fl.runtime.stream_completion import provider
from sglang_fl.runtime.stream_completion.patch import (
    _wrap_copy_to_cpu,
)
import sglang_fl.runtime.stream_completion as stream_completion


class _FallbackEvent:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class _Completion:
    def __init__(self, error=None):
        self.error = error
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        if self.error is not None:
            raise self.error


class _Result:
    def __init__(self, device_type="musa"):
        self.next_token_ids = SimpleNamespace(
            device=SimpleNamespace(type=device_type)
        )
        self.copy_done = None


def test_copy_wrapper_replaces_recorded_event_with_completion_proxy(monkeypatch):
    fallback = _FallbackEvent()
    completion = _Completion()

    def original(result, return_logprob):
        assert return_logprob is False
        result.next_token_ids = "cpu-output"
        result.copy_done = fallback
        return "copied"

    monkeypatch.setattr(
        stream_completion,
        "try_enqueue_completion",
        lambda device: completion,
    )
    result = _Result()
    value = _wrap_copy_to_cpu(original)(result, False)

    assert value == "copied"
    assert isinstance(result.copy_done, provider.CompletionEventProxy)
    result.copy_done.synchronize()
    result.copy_done.synchronize()
    assert completion.wait_calls == 1
    assert fallback.sync_calls == 0


def test_completion_wait_failure_uses_original_event():
    fallback = _FallbackEvent()
    completion = _Completion(RuntimeError("wait failed"))
    proxy = provider.CompletionEventProxy(completion, fallback)

    proxy.synchronize()
    proxy.synchronize()

    assert completion.wait_calls == 1
    assert fallback.sync_calls == 1


def test_pool_exhaustion_keeps_original_event(monkeypatch):
    fallback = _FallbackEvent()

    def original(result, _return_logprob):
        result.copy_done = fallback

    monkeypatch.setattr(
        stream_completion,
        "try_enqueue_completion",
        lambda device: None,
    )
    result = _Result()
    _wrap_copy_to_cpu(original)(result, False)

    assert result.copy_done is fallback


def test_proxy_delegates_unknown_attributes_to_fallback():
    fallback = _FallbackEvent()
    proxy = provider.CompletionEventProxy(_Completion(), fallback)
    assert proxy.sync_calls == 0
    proxy.synchronize()


def test_unsupported_device_yields_no_pool():
    device = SimpleNamespace(type="not-a-real-device")
    assert provider._get_pool(device) is None


def test_stream_pointer_prefers_vendor_attribute():
    from sglang_fl.runtime.stream_completion.vendors import stream_pointer

    spec = provider.get_vendor_spec("cuda")
    stream = SimpleNamespace(cuda_stream=0xABCD)
    assert stream_pointer(stream, spec) == 0xABCD
