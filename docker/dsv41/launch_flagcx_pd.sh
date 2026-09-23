#!/usr/bin/env bash
set -euo pipefail

: "${PD_ROLE:?Set PD_ROLE=prefill or decode}"
pid_file="/work/results/${PD_ROLE}.pid"
log_file="/work/results/${PD_ROLE}.log"
if test -s "$pid_file"; then
    pid="$(cat "$pid_file")"
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    if test -n "$state" && test "${state#Z}" = "$state"; then
        echo "Recorded $PD_ROLE server is already running: $pid" >&2
        exit 1
    fi
fi
nohup bash /work/scripts/serve_flagcx_pd.sh > "$log_file" 2>&1 < /dev/null &
echo "$!" > "$pid_file"
echo "$PD_ROLE server PID: $(cat "$pid_file")"
