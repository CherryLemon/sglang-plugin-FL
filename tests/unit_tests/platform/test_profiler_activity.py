"""The out-of-tree platform must identify its PyTorch profiler activity."""

import torch

import sglang.srt.platforms  # Initialize the entry-point platform before direct import.
from sglang_fl.platform import PlatformFL


def test_cuda_profiler_activity():
    platform = PlatformFL.__new__(PlatformFL)
    platform._device_type = "cuda"

    assert platform.get_torch_profiler_activity_str() == "CUDA"
    assert platform.get_torch_profiler_activity() == torch.profiler.ProfilerActivity.CUDA


def test_optional_device_profiler_activity_name():
    platform = PlatformFL.__new__(PlatformFL)
    platform._device_type = "musa"

    assert platform.get_torch_profiler_activity_str() == "MUSA"
