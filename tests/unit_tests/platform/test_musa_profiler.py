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

import logging

import sglang_fl.profiler as musa_profiler


class _FakeFunction:
    def __init__(self, name, calls, result):
        self.name = name
        self.calls = calls
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self):
        self.calls.append(self.name)
        return self.result


class _FakeMusart:
    def __init__(self, result=801):
        self.calls = []
        self.musaProfilerStart = _FakeFunction("musaProfilerStart", self.calls, result)
        self.musaProfilerStop = _FakeFunction("musaProfilerStop", self.calls, result)
        self.musaGetLastError = _FakeFunction("musaGetLastError", self.calls, result)


def test_musa_profiler_api_clears_sticky_runtime_error(monkeypatch, caplog):
    library = _FakeMusart()
    monkeypatch.setattr(musa_profiler.ctypes, "CDLL", lambda _name: library)

    controller = musa_profiler.MusaProfilerApi()
    with caplog.at_level(logging.WARNING, logger="sglang_fl.profiler"):
        assert controller.start() == 801
        assert controller.stop() == 801

    assert library.calls == [
        "musaProfilerStart",
        "musaGetLastError",
        "musaProfilerStop",
        "musaGetLastError",
    ]
    assert caplog.text.count("cleared error 801") == 2


def test_legacy_profile_endpoint_maps_gpu_and_msys_to_musa(monkeypatch):
    events = []

    class FakeTorchProfiler:
        def start(self):
            events.append("torch_start")

        def stop(self):
            events.append("torch_stop")

    captured_kwargs = {}

    def fake_profile(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeTorchProfiler()

    class FakeApi:
        def start(self):
            events.append("api_start")
            return 801

        def stop(self):
            events.append("api_stop")
            return 801

    class FakeSchedulerProfilerMixin:
        def start_profile(self, stage=None):
            self.activities_seen_by_original_start = list(self.profiler_activities)
            return "start-result"

        def stop_profile(self, stage=None):
            self.activities_seen_by_original_stop = list(self.profiler_activities)
            if self.torch_profiler is not None:
                self.torch_profiler.stop()
            self.profile_in_progress = False
            return "stop-result"

    monkeypatch.setattr(musa_profiler.torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(musa_profiler, "_MUSA_PROFILER_API", FakeApi())
    monkeypatch.setattr(musa_profiler, "_is_first_rank_in_node", lambda _self: True)
    musa_profiler._wrap_legacy_scheduler_profiler(FakeSchedulerProfilerMixin)

    scheduler = FakeSchedulerProfilerMixin()
    scheduler.profiler_activities = ["CPU", "GPU", "MSYS"]
    scheduler.torch_profiler_with_stack = False
    scheduler.torch_profiler_record_shapes = True
    scheduler.torch_profiler = None
    scheduler.profile_in_progress = False

    assert scheduler.start_profile() == "start-result"
    assert scheduler.activities_seen_by_original_start == []
    assert scheduler.profiler_activities == ["CPU", "GPU", "MSYS"]
    assert captured_kwargs["activities"] == [
        musa_profiler.torch.profiler.ProfilerActivity.CPU,
        musa_profiler._torch_musa_activity(),
    ]
    assert captured_kwargs["with_stack"] is False
    assert captured_kwargs["record_shapes"] is True
    assert scheduler.profile_in_progress is True
    assert events == ["api_start", "torch_start"]

    assert scheduler.stop_profile() == "stop-result"
    assert scheduler.activities_seen_by_original_stop == ["CPU", "GPU"]
    assert scheduler.profiler_activities == ["CPU", "GPU", "MSYS"]
    assert events == ["api_start", "torch_start", "api_stop", "torch_stop"]


def test_profile_v2_maps_musa_activities(monkeypatch):
    events = []
    captured_create = {}
    captured_torch = {}

    class FakeTorchProfiler:
        def start(self):
            events.append("torch_start")

    def fake_profile(**kwargs):
        captured_torch.update(kwargs)
        return FakeTorchProfiler()

    class FakeProfilerBase:
        @staticmethod
        def create(activities, with_stack, record_shapes, **kwargs):
            captured_create.update(
                activities=activities,
                with_stack=with_stack,
                record_shapes=record_shapes,
                kwargs=kwargs,
            )
            return "create-result"

    class FakeProfilerTorch:
        def start(self):
            events.append("original_torch_start")

    class FakeProfilerCudart:
        pass

    class FakeApi:
        def start(self):
            events.append("api_start")

        def stop(self):
            events.append("api_stop")

    monkeypatch.setattr(musa_profiler.torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(musa_profiler, "_MUSA_PROFILER_API", FakeApi())
    musa_profiler._wrap_profile_v2(
        FakeProfilerBase, FakeProfilerTorch, FakeProfilerCudart
    )

    assert (
        FakeProfilerBase.create(
            ["CPU", "MUSA", "GPU", "MSYS", "MUSA_PROFILER"],
            False,
            True,
            output_dir="/tmp/profile",
        )
        == "create-result"
    )
    assert captured_create == {
        "activities": ["CPU", "GPU", "CUDA_PROFILER"],
        "with_stack": False,
        "record_shapes": True,
        "kwargs": {"output_dir": "/tmp/profile"},
    }

    torch_profiler = FakeProfilerTorch()
    torch_profiler.activities = ["CPU", "GPU"]
    torch_profiler.with_stack = False
    torch_profiler.record_shapes = True
    torch_profiler.start()
    assert captured_torch["activities"] == [
        musa_profiler.torch.profiler.ProfilerActivity.CPU,
        musa_profiler._torch_musa_activity(),
    ]

    api_profiler = FakeProfilerCudart()
    api_profiler.first_rank_in_node = True
    api_profiler.start()
    api_profiler.stop()
    assert events == ["torch_start", "api_start", "api_stop"]
