#!/usr/bin/env bash
set -euo pipefail

pid_file=/work/results/router.pid
if test -s "$pid_file"; then
    pid="$(cat "$pid_file")"
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    if test -n "$state" && test "${state#Z}" = "$state"; then
        echo "Router is already running: $pid" >&2
        exit 1
    fi
fi
nohup python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://10.8.2.1:31819 \
    --decode http://10.8.2.3:31820 \
    --mini-lb \
    --port 31821 \
    > /work/results/router.log 2>&1 < /dev/null &
echo "$!" > "$pid_file"
echo "FlagCX PD router PID: $(cat "$pid_file")"
