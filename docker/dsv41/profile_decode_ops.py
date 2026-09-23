#!/usr/bin/env python3
"""Compare GPU kernel time in matching DSV4.1 steady-decode profiler traces.

Inputs are SGLang's per-rank ``.trace.json.gz`` files from the D worker.
The categories below are tied to the DSV4.1 hybrid/vendor implementations;
unrelated CUDA kernels are kept out of the operator gap calculation.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
from statistics import median


def category(mode, name):
    if name == "w8a8_block_fp8_matmul_kernel" or name in (
        "_w8a8_block_fp8_matmul_hopper_static",
        "_w8a8_block_fp8_matmul",
    ):
        return "w8a8_gemm_main"
    if name == "_reduce_block_fp8_split_k":
        return "w8a8_splitk_reduce"
    if (mode == "hybrid" and name == "_quantize") or name.startswith(
        "void sglang::per_token_group_quant_8bit_v2_kernel"
    ):
        return "w8a8_quant"
    if name == "_fp4_index_logits_grouped_kernel":
        return "fp4_grouped"
    if name == "_fp4_index_logits_kernel":
        return "fp4_plain"
    if name == "_hc_mix_stats_partial_kernel":
        return "mhc_stats"
    if name in (
        "_hc_mix_reduce_sinkhorn_kernel",
        "_hc_mix_stats_reduce_kernel",
        "hc_split_sinkhorn_kernel__kernel",
    ):
        return "mhc_sinkhorn"
    if name == "_hc_combine_kernel":
        return "mhc_combine"
    if name in ("mhc_post_kernel_hc_mult_4", "_mhc_post_split_h_tilelang_kernel"):
        return "mhc_post"
    if "cutlass" in name and "GemmUniversal" in name:
        return "control_moe_gemm"
    if "sglang::all_reduce_kernel" in name:
        return "control_tp_allreduce"
    if "sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel" in name:
        return "control_sparse_attention"
    return None


COMPOSITES = {
    "w8a8_total": ("w8a8_gemm_main", "w8a8_splitk_reduce", "w8a8_quant"),
    "fp4_total": ("fp4_grouped", "fp4_plain"),
    "mhc_total": ("mhc_stats", "mhc_sinkhorn", "mhc_combine", "mhc_post"),
}


def summarize_trace(path, mode):
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    kernels = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    if not kernels:
        raise ValueError(f"No GPU kernels in {path}")
    families = defaultdict(lambda: {"count": 0, "ms": 0.0})
    for event in kernels:
        family = category(mode, event["name"])
        if family is not None:
            families[family]["count"] += 1
            families[family]["ms"] += event["dur"] / 1000
    for family, parts in COMPOSITES.items():
        families[family] = {
            "count": sum(families[part]["count"] for part in parts),
            "ms": sum(families[part]["ms"] for part in parts),
        }
    return {
        "cuda_graph_launch_count": sum(
            event.get("name") == "cudaGraphLaunch" for event in trace["traceEvents"]
        ),
        "gpu_kernel_count": len(kernels),
        "gpu_kernel_sum_ms": sum(event["dur"] for event in kernels) / 1000,
        "gpu_kernel_span_ms": (
            max(event["ts"] + event["dur"] for event in kernels)
            - min(event["ts"] for event in kernels)
        )
        / 1000,
        "families": dict(families),
    }


def summarize_dir(path, mode):
    files = sorted(path.glob("*.trace.json.gz"))
    if not files:
        raise ValueError(f"No SGLang trace files in {path}")
    result = {}
    for file in files:
        match = re.search(r"-TP-(\d+)", file.name)
        if match is None:
            raise ValueError(f"Missing TP rank in {file.name}")
        rank = int(match.group(1))
        if rank in result:
            raise ValueError(f"Duplicate TP rank {rank} in {path}")
        result[rank] = summarize_trace(file, mode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    data = {
        "hybrid": summarize_dir(args.hybrid, "hybrid"),
        "vendor": summarize_dir(args.vendor, "vendor"),
    }
    if data["hybrid"].keys() != data["vendor"].keys():
        raise ValueError("Hybrid and vendor trace ranks differ")
    launches = {
        mode: {sample["cuda_graph_launch_count"] for sample in ranks.values()}
        for mode, ranks in data.items()
    }
    if len(launches["hybrid"]) != 1 or launches["hybrid"] != launches["vendor"]:
        raise ValueError(f"CUDA Graph launch counts differ: {launches}")
    print(f"ranks={len(data['hybrid'])} steps={args.steps} graph_launches={launches['hybrid']}")
    print("| Family | Hybrid calls | Vendor calls | Hybrid ms/step | Vendor ms/step | Hybrid/vendor | Delta ms/step |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    families = (
        "w8a8_gemm_main",
        "w8a8_splitk_reduce",
        "w8a8_quant",
        "w8a8_total",
        "fp4_grouped",
        "fp4_plain",
        "fp4_total",
        "mhc_stats",
        "mhc_sinkhorn",
        "mhc_combine",
        "mhc_post",
        "mhc_total",
        "control_moe_gemm",
        "control_tp_allreduce",
        "control_sparse_attention",
    )
    for family in families:
        counts = {
            mode: {rank["families"].get(family, {}).get("count", 0) for rank in ranks.values()}
            for mode, ranks in data.items()
        }
        times = {
            mode: median(
                rank["families"].get(family, {}).get("ms", 0) for rank in ranks.values()
            )
            / args.steps
            for mode, ranks in data.items()
        }
        hybrid, vendor = times["hybrid"], times["vendor"]
        ratio = f"{hybrid / vendor:.2f}×" if vendor else "—"
        print(
            f"| {family} | {sorted(counts['hybrid'])} | {sorted(counts['vendor'])} | "
            f"{hybrid:.3f} | {vendor:.3f} | {ratio} | {hybrid - vendor:+.3f} |"
        )
    for metric in ("gpu_kernel_span_ms", "gpu_kernel_sum_ms"):
        values = {
            mode: median(rank[metric] for rank in ranks.values()) / args.steps
            for mode, ranks in data.items()
        }
        print(f"{metric}/step: hybrid={values['hybrid']:.3f} ms vendor={values['vendor']:.3f} ms")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
