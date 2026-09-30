# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit opt-in controls shared by MThreads integration hooks."""

import os


def enabled(name):
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes", "on")
