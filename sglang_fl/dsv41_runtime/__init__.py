# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 runtime hooks, enabled with SGLANG_FL_DSV41=1."""

import os


def load_plugin():
    if os.getenv("SGLANG_FL_DSV41") != "1":
        return
    from .patch import apply_dsv41_patches

    apply_dsv41_patches()
