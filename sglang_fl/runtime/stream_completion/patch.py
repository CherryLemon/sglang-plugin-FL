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

"""Replace accelerator output Event waits with native callback + eventfd waits."""

from __future__ import annotations

import logging
import os
from functools import wraps

from .policy import ENV_NAME, is_enabled, resolve_vendor

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_sglang_fl_eventfd_completion"


def _wrap_copy_to_cpu(original):
    if getattr(original, _PATCH_MARKER, False):
        return original

    @wraps(original)
    def wrapped(result, return_logprob: bool):
        next_token_ids = getattr(result, "next_token_ids", None)
        copy_device = getattr(next_token_ids, "device", None)
        value = original(result, return_logprob)
        fallback_event = getattr(result, "copy_done", None)
        if copy_device is None or fallback_event is None:
            return value
        from sglang_fl.runtime.stream_completion import (
            CompletionEventProxy,
            try_enqueue_completion,
        )

        completion = try_enqueue_completion(copy_device)
        if completion is not None:
            result.copy_done = CompletionEventProxy(completion, fallback_event)
        return value

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def apply_eventfd_completion_patch() -> bool:
    vendor = resolve_vendor()
    if not is_enabled(vendor):
        logger.info("%s disabled (vendor=%s)", ENV_NAME, vendor or "unknown")
        return False
    try:
        from sglang.srt.managers import utils as manager_utils
    except Exception as exc:
        logger.warning("eventfd completion patch skipped: %s", exc)
        return False

    original = manager_utils.GenerationBatchResult.copy_to_cpu
    if getattr(original, _PATCH_MARKER, False):
        return True
    manager_utils.GenerationBatchResult.copy_to_cpu = _wrap_copy_to_cpu(original)
    logger.info(
        "eventfd output completion enabled with Event fallback "
        "(vendor=%s, pool size=%s)",
        vendor or "unknown",
        os.environ.get("SGLANG_FL_EVENTFD_COMPLETION_POOL_SIZE", "64"),
    )
    return True


__all__ = ["apply_eventfd_completion_patch"]
