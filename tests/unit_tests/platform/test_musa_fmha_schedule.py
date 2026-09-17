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

from sglang_fl.dispatch.backends.vendor.mthreads.patches import fmha_schedule


CURRENT_CONFIG = (192, 64, 1, 1, 256, 256, 3, 3, False)


def _original(*args, **kwargs):
    return CURRENT_CONFIG


def test_s5000_8k_prefill_enables_pack_gqa(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", raising=False)

    wrapped = fmha_schedule._wrap_get_fwd_kernel_config(_original)
    tuned = wrapped(8192, 8, 256, 256, 2)

    assert tuned[:-1] == CURRENT_CONFIG[:-1]
    assert tuned[-1] is True


def test_fmha_schedule_is_narrow_and_respects_explicit_choice(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    wrapped = fmha_schedule._wrap_get_fwd_kernel_config(_original)

    assert wrapped(4096, 8, 256, 256, 2) == CURRENT_CONFIG
    assert wrapped(8192, 4, 256, 256, 2) == CURRENT_CONFIG
    assert wrapped(8192, 8, 128, 128, 2) == CURRENT_CONFIG
    assert wrapped(8192, 8, 256, 256, 2, False) == CURRENT_CONFIG
    assert wrapped(8192, 8, 256, 256, 2, None, True) == CURRENT_CONFIG

    monkeypatch.setenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", "off")
    assert wrapped(8192, 8, 256, 256, 2) == CURRENT_CONFIG


def test_apply_patches_all_mate_aliases(monkeypatch):
    monkeypatch.setattr(fmha_schedule, "_device_name", lambda: "MTT S5000")
    monkeypatch.delenv("SGLANG_MUSA_FMHA_PREFILL_PACK_GQA", raising=False)
    utils = SimpleNamespace(_get_fwd_kernel_config=_original)
    fwd = SimpleNamespace(_get_fwd_kernel_config=_original)
    metadata = SimpleNamespace(_get_metadata_kernel_config=_original)
    modules = iter((utils, fwd, metadata))
    monkeypatch.setattr(
        fmha_schedule.importlib, "import_module", lambda _name: next(modules)
    )

    assert fmha_schedule.apply_musa_fmha_schedule_patch()
    assert utils._get_fwd_kernel_config is fwd._get_fwd_kernel_config
    assert utils._get_fwd_kernel_config is metadata._get_metadata_kernel_config
    assert getattr(utils._get_fwd_kernel_config, fmha_schedule._PATCH_MARKER)
