#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Run in the pinned S5000 vendor runtime; see tests/musa/README.md.
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.6-35B-A3B weights}"
source "$(dirname "${BASH_SOURCE[0]}")/qwen36_runtime.env"
: "${FLAGCX_PATH:?Set FLAGCX_PATH to the FlagCX installation directory}"
if [[ ! -r "$FLAGCX_PATH/build/lib/libflagcx.so" ||
      ! -r "$FLAGCX_PATH/plugin/interservice/flagcx_wrapper.py" ]]; then
  printf 'FlagCX build not found under %s\n' "$FLAGCX_PATH" >&2
  exit 1
fi
exec "${PYTHON:-python}" -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 2 --dtype bfloat16 \
  --page-size 1 --context-length 32768 --chunked-prefill-size 16384 \
  --mem-fraction-static 0.90 --disable-radix-cache \
  --disable-piecewise-cuda-graph \
  --cuda-graph-bs 1 2 4 8 12 16 24 32 40 48 56 64 \
  --trust-remote-code "$@"
