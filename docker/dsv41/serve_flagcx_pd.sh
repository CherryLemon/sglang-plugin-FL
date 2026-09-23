#!/usr/bin/env bash
set -euo pipefail

: "${PD_ROLE:?Set PD_ROLE=prefill or decode}"
case "$PD_ROLE" in
    prefill) : "${SERVE_PORT:=31819}" ;;
    decode) : "${SERVE_PORT:=31820}" ;;
    *) echo "Unknown PD_ROLE: $PD_ROLE" >&2; exit 2 ;;
esac
export SERVE_PORT
export SGLANG_FL_DIST_BACKEND=nccl
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"
# FlagCX's RPC listener chooses an interface independently of SGLang's
# advertised get_local_ip_auto() address. Both must resolve to bond0 here.
export FLAGCX_SOCKET_IFNAME="${FLAGCX_SOCKET_IFNAME:-bond0}"
export FLAGCX_P2P_SLICE_SIZE="${FLAGCX_P2P_SLICE_SIZE:-65536}"
export FLAGCX_P2P_WORKERS_PER_POOL="${FLAGCX_P2P_WORKERS_PER_POOL:-2}"
export FLAGCX_P2P_QPS_PER_CONN="${FLAGCX_P2P_QPS_PER_CONN:-2}"

decode_layout_args=()
if [[ "$PD_ROLE" == decode && "${PD_DECODE_DPA:-0}" == 1 ]]; then
    export SGLANG_DSV41_EXPERIMENTAL_DPA_PD=1
    export CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-40}"
    export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-1 2 3 4 6 8 12 16 24 32 40}"
    decode_layout_args=(--enable-dp-attention --dp-size 4 --enable-dp-lm-head)
fi

exec bash /work/scripts/serve_single_node.sh \
    --disaggregation-mode "$PD_ROLE" \
    --disaggregation-transfer-backend flagcx \
    --disaggregation-ib-device mlx5_bond_0 \
    "${decode_layout_args[@]}" \
    "$@"
