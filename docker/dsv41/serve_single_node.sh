#!/usr/bin/env bash
# Run inside the migration image after GPU device/driver preflight.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the pinned DeepSeek V4.1 Flash checkpoint}"
test -f "${MODEL_PATH}/config.json"

export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_DSV41_REQUIRED=1
export SGLANG_FL_DSV41_BACKEND="${SGLANG_FL_DSV41_BACKEND:-hybrid}"
export SGLANG_FL_DSV41_QUANTIZATION=quantized
export USE_FLAGGEMS=0
export SGLANG_FL_OOT_ENABLED=0
export SGLANG_FL_STRICT=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_ATTN_TP_REDUCE=1
export SGLANG_OPT_DSV41_INDEXER_SKIP_INVALID_TILES=1
export SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name deepseek-v4.1-flash \
    --trust-remote-code \
    --tp 8 --ep-size 8 --moe-a2a-backend none \
    --context-length "${CONTEXT_LENGTH:-65536}" \
    --chunked-prefill-size 4096 \
    --max-total-tokens "${MAX_TOTAL_TOKENS:-131072}" \
    --max-running-requests 16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --cuda-graph-max-bs-decode 16 \
    --cuda-graph-bs-decode 1 2 3 4 6 8 12 16 \
    --reasoning-parser auto --tool-call-parser auto \
    --enable-metrics --decode-log-interval 10 \
    --host "${SERVE_HOST:-0.0.0.0}" --port "${SERVE_PORT:-31818}" \
    "$@"
