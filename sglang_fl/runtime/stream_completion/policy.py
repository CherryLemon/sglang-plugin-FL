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

"""Enable policy for the generic stream-completion patch.

``SGLANG_FL_EVENTFD_COMPLETION=auto|on|off`` (default ``auto``) is the single
switch.  In ``auto`` mode the decision is delegated to an optional per-vendor
hook ``sglang_fl.dispatch.backends.vendor.<vendor>.eventfd_policy.auto_enable``
so that a vendor can restrict the optimisation to the hardware it was validated
on.  Without a hook, ``auto`` stays off.

The legacy ``SGLANG_MUSA_EVENTFD_COMPLETION`` variable is still honoured for
backward compatibility.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ENV_NAME = "SGLANG_FL_EVENTFD_COMPLETION"
LEGACY_ENV_NAME = "SGLANG_MUSA_EVENTFD_COMPLETION"
_POLICY_MODULE = "sglang_fl.dispatch.backends.vendor.{vendor}.eventfd_policy"

_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_ON_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_value() -> str:
    for name in (ENV_NAME, LEGACY_ENV_NAME):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return "auto"


def resolve_vendor() -> Optional[str]:
    """Return the FlagGems vendor name, or ``None`` when detection fails."""
    try:
        try:
            # FlagGems<=5.0.2: DeviceDetector lives in device.
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            # FlagGems>5.0.2: DeviceDetector lives in device_finder.
            from flag_gems.runtime.backend.device_finder import DeviceDetector

        return DeviceDetector().vendor_name
    except Exception as exc:
        logger.debug("eventfd completion: vendor detection failed (%s)", exc)
        return None


def _auto_enable(vendor: Optional[str]) -> bool:
    if not vendor:
        return False
    module_name = _POLICY_MODULE.format(vendor=vendor)
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.debug("eventfd completion: no auto policy for %s", vendor)
        return False
    hook = getattr(module, "auto_enable", None)
    if hook is None:
        return False
    try:
        return bool(hook())
    except Exception as exc:
        logger.warning(
            "eventfd completion: %s.auto_enable() failed (%s)", module_name, exc
        )
        return False


def is_enabled(vendor: Optional[str] = None) -> bool:
    """Resolve the switch, consulting the vendor hook only in ``auto``."""
    value = _env_value().lower()
    if value in _OFF_VALUES:
        return False
    if value in _ON_VALUES:
        return True
    if value != "auto":
        raise ValueError(f"Unsupported {ENV_NAME} value: {value!r}")
    if vendor is None:
        vendor = resolve_vendor()
    return _auto_enable(vendor)


__all__ = [
    "ENV_NAME",
    "LEGACY_ENV_NAME",
    "is_enabled",
    "resolve_vendor",
]
