# SPDX-License-Identifier: Apache-2.0
"""Construction-time, per-ModelRunner opt-ins. No tensor or numerical work."""

import os

TARGET_FLAG = "SGLANG_HY4_BF16_SPARSE_MLA_ACTIVE_SPLIT_TARGET"
DRAFT_FLAG = "SGLANG_HY4_BF16_SPARSE_MLA_ACTIVE_SPLIT_DRAFT"


def freeze_role_policy(model_runner, environ=None):
    """Only an actual bool is a known role; legacy process-global flag is ignored."""
    flag = getattr(model_runner, "is_draft_worker", None)
    role = "draft" if flag is True else "target" if flag is False else "unknown"
    source = os.environ if environ is None else environ
    name = DRAFT_FLAG if role == "draft" else TARGET_FLAG if role == "target" else None
    return role, bool(name is not None and source.get(name, "0") == "1")


def eligible_forward(backend):
    # Python attributes are frozen before warm/capture. Never infer policy from
    # SQ alone or use a process-global role: target and draft coexist.
    return (
        getattr(backend, "_hy4_mla_opted_in", False) is True
        and getattr(backend, "_hy4_mla_role", "unknown") in ("target", "draft")
        and getattr(getattr(backend, "_hy4_forward_mode", None), "name", None)
        in ("TARGET_VERIFY", "DECODE")
    )
