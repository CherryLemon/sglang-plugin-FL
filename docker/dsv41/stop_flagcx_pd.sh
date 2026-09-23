#!/usr/bin/env bash
set -euo pipefail

: "${PD_ROLE:?Set PD_ROLE=prefill or decode}"
pid_file="/work/results/${PD_ROLE}.pid"
pid="$(cat "$pid_file")"
cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
case "$cmd" in
    *sglang.launch_server*--disaggregation-mode*"$PD_ROLE"*) kill -TERM "$pid" ;;
    *) echo "Recorded PID is not our $PD_ROLE SGLang server" >&2; exit 1 ;;
esac
for _ in $(seq 1 30); do
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    if test -z "$state" || test "${state#Z}" != "$state"; then
        echo "Own $PD_ROLE server stopped"
        exit 0
    fi
    sleep 1
done
echo "Own $PD_ROLE server did not stop within 30 seconds" >&2
exit 1
