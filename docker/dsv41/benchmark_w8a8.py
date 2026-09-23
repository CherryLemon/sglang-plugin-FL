"""Reproducible H100 W8A8 operator benchmark; no model/service required.

The baseline directory contains untouched sources from the pinned r7 image.
Input shapes/frequencies come from paired kernel grids in all eight decode traces.
All timings use CUDA Graph replay, include allocation/workspace/reduction kernels,
and exclude JIT and Python dispatch. Replays keep inputs fixed (warm-cache bench).
"""

import argparse
import importlib
import json
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import triton


def quant_ref(x, pow2=True):
    groups = x.float().reshape(x.shape[0], -1, 32)
    scale = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
    if pow2:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (groups / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), scale


def gemm_ref(q, w, qs, ws):
    a = q.float() * qs.repeat_interleave(32, 1)
    b = w.float() * ws.repeat_interleave(32, 0)[: w.shape[0]].repeat_interleave(32, 1)
    return a @ b.T


def capture(fn, warmup=25, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure_set(functions, iterations=100, rounds=7):
    captures = {
        name: capture(fn, iterations=iterations) for name, fn in functions.items()
    }
    samples = {name: [] for name in functions}
    rng = random.Random(431)
    for _ in range(rounds):
        names = list(captures)
        rng.shuffle(names)
        for name in names:
            graph, _ = captures[name]
            torch.cuda.synchronize()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            samples[name].append(a.elapsed_time(b) * 1000 / iterations)
    return {
        name: {
            "p50_us": statistics.median(v),
            "min_us": min(v),
            "max_us": max(v),
            "samples_us": v,
        }
        for name, v in samples.items()
    }


def errors(y, ref):
    err = y.float() - ref
    return {
        "rmse": err.square().mean().sqrt().item(),
        "max_abs": err.abs().max().item(),
    }


def verify_gemm(y, ref, baseline_error, rms, dtype):
    e = errors(y, ref)
    noise = 2e-5 if dtype == torch.bfloat16 else 5e-6
    assert e["rmse"] <= baseline_error["rmse"] * 1.05 + rms * noise, (e, baseline_error)
    assert e["max_abs"] <= baseline_error["max_abs"] * 1.05 + rms * 2e-5, (
        e,
        baseline_error,
    )
    torch.testing.assert_close(
        y.float(),
        ref,
        rtol=0.004 if dtype == torch.bfloat16 else 2e-5,
        atol=baseline_error["max_abs"] * 1.1 + rms * 2e-5,
    )
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--shapes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["tune_quant", "benchmark", "verify"], default="benchmark"
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--weight-pool-mib", type=int, default=0)
    args = p.parse_args()
    sys.path.insert(0, str(args.baseline))
    base = importlib.import_module("w8a8_baseline")
    base_linear = importlib.import_module("linear_baseline")
    opt = importlib.import_module("flag_gems.ops.w8a8_block_fp8_matmul")
    linear = importlib.import_module("flag_gems.fused.dsv41.block_fp8_linear")
    from sglang.kernels.ops.quantization import fp8_kernel as vendor
    from sglang.srt.layers.quantization.fp8_utils import (
        triton_w8a8_block_fp8_linear as vendor_linear,
    )

    # Select the native vendor implementation explicitly, bypassing OOT dispatch.
    assert vendor_linear.__dsv41_op__ == "linear.block_fp8"
    vendor_linear = vendor_linear.__wrapped__
    logging.getLogger("w8a8_baseline").setLevel(logging.ERROR)
    logging.getLogger("flag_gems.ops.w8a8_block_fp8_matmul").setLevel(logging.ERROR)
    torch.manual_seed(431)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (9, 0)
    shapes = json.loads(args.shapes.read_text())["shapes"]
    if args.limit:
        shapes = shapes[: args.limit]
    report = {
        "mode": args.mode,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "triton_path": triton.__file__,
        "seed": 431,
        "weight_pool_mib": args.weight_pool_mib,
        "protocol": "25 warmup; 100 calls/captured graph; 7 shuffled timing rounds; fixed inputs; CUDA events; includes split workspace/reduction; no profiling",
        "results": [],
        "started": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    def qbase(x, pow2=True):
        m, k = x.shape
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = torch.empty((m, k // 32), device=x.device, dtype=torch.float32)
        base_linear._quantize[(m * k // 32,)](x, q, s, k, 32, pow2, False, num_warps=1)
        return q, s

    def qopt(x, groups=None, warps=None, pow2=True):
        m, k = x.shape
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = torch.empty((m, k // 32), device=x.device, dtype=torch.float32)
        if groups is None:
            groups, warps = linear._quantize_group_config(m * k // 32)
        linear._quantize_groups[(triton.cdiv(m * k // 32, groups),)](
            x, q, s, m * k // 32, 32, pow2, groups, num_warps=warps
        )
        return q, s

    if args.mode == "verify":
        # Group-count tails, zeros, saturation, exponent boundaries, and nonfinite parity.
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for pow2 in [False, True]:
                for groups in [
                    1,
                    15,
                    16,
                    17,
                    31,
                    32,
                    33,
                    159,
                    160,
                    161,
                    8191,
                    8192,
                    8193,
                    32767,
                    32768,
                    32769,
                    92160,
                ]:
                    x = torch.randn(1, groups * 32, device="cuda", dtype=dtype)
                    if groups > 15:
                        special = torch.tensor(
                            [
                                0.0,
                                -0.0,
                                448.0,
                                -448.0,
                                1e-12,
                                -1e-12,
                                float("inf"),
                                -float("inf"),
                                float("nan"),
                            ],
                            device="cuda",
                            dtype=dtype,
                        )
                        x[0, 32 : 32 + len(special)] = special
                    x[0, :32] = 0
                    bq, bs = qbase(x, pow2)
                    oq, os = qopt(x, pow2=pow2)
                    # NaN payloads do not carry semantic meaning; all other FP8 bytes must match.
                    bnan = bq.float().isnan()
                    assert torch.equal(bnan, oq.float().isnan())
                    assert torch.equal(
                        bq.view(torch.uint8)[~bnan], oq.view(torch.uint8)[~bnan]
                    )
                    torch.testing.assert_close(bs, os, rtol=0, atol=0, equal_nan=True)
                    if groups <= 161:
                        _refq, refs = quant_ref(x, pow2)
                        # PyTorch and the original Triton amax differ for NaNs.
                        # Preserve original nonfinite semantics; use the independent
                        # PyTorch oracle on groups whose inputs are finite.
                        finite = x.reshape(1, groups, 32).isfinite().all(-1)
                        torch.testing.assert_close(
                            os[finite], refs[finite], rtol=2e-7, atol=0
                        )
                    graph, (gq, gs) = capture(
                        lambda pow2=pow2, x=x: qopt(x, pow2=pow2), iterations=1
                    )
                    x.copy_(torch.randn_like(x))
                    rq, rs = qopt(x, pow2=pow2)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(gq.view(torch.uint8), rq.view(torch.uint8))
                    assert torch.equal(gs, rs)
                    report["results"].append(
                        {
                            "dtype": str(dtype),
                            "pow2": pow2,
                            "groups": groups,
                            "passed": True,
                        }
                    )
        save()
        print("quant validation passed", len(report["results"]), flush=True)
        return

    if args.mode == "tune_quant":
        seen = set()
        for row in shapes:
            m, k = row["m"], row["k"]
            if (m, k) in seen:
                continue
            seen.add((m, k))
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            bq, bs = qbase(x)
            funcs = {
                "baseline": lambda x=x: qbase(x),
                "vendor": lambda x=x: vendor.sglang_per_token_group_quant_fp8(
                    x, 32, scale_ue8m0=True
                ),
            }
            for g in [8, 16, 32, 64]:
                for w in [1, 4, 8]:
                    oq, os = qopt(x, g, w)
                    assert torch.equal(oq.view(torch.uint8), bq.view(torch.uint8))
                    assert torch.equal(os, bs)
                    funcs[f"g{g}_w{w}"] = lambda g=g, w=w, x=x: qopt(x, g, w)
            times = measure_set(funcs)
            record = {
                "m": m,
                "k": k,
                "times": times,
                "best": min(times, key=lambda n: times[n]["p50_us"]),
            }
            report["results"].append(record)
            save()
            print(
                json.dumps({k: v for k, v in record.items() if k != "times"}),
                flush=True,
            )
        return

    for shape in shapes:
        m, n, k = (shape[t] for t in ["m", "n", "k"])
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        ws = torch.exp2(
            torch.randint(-4, 4, ((n + 31) // 32, k // 32), device="cuda").float()
        )
        q, qs = qbase(x)
        oq, os = qopt(x)
        assert torch.equal(q.view(torch.uint8), oq.view(torch.uint8)) and torch.equal(
            qs, os
        )
        vq, vs = vendor.sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
        assert torch.equal(q.view(torch.uint8), vq.view(torch.uint8)) and torch.equal(
            qs, vs
        )
        ref = gemm_ref(q, w, qs, ws)
        rms = ref.square().mean().sqrt().item()
        check = {}
        for dtype in [torch.float32, torch.bfloat16]:
            by = base.w8a8_block_fp8_matmul(q, w, qs, ws, [32, 32], dtype)
            be = errors(by, ref)
            for name, fn in [
                ("candidate", opt.w8a8_block_fp8_matmul),
                ("vendor", vendor.w8a8_block_fp8_matmul_triton),
            ]:
                y = fn(q, w, qs, ws, [32, 32], dtype)
                check[f"{name}_{dtype}"] = verify_gemm(y, ref, be, rms, dtype)
            check[f"baseline_{dtype}"] = be
        gemms = {
            "baseline": lambda q=q, qs=qs, w=w, ws=ws: base.w8a8_block_fp8_matmul(
                q, w, qs, ws, [32, 32]
            ),
            "candidate": lambda q=q, qs=qs, w=w, ws=ws: opt.w8a8_block_fp8_matmul(
                q, w, qs, ws, [32, 32]
            ),
            "vendor": lambda q=q, qs=qs, w=w, ws=ws: (
                vendor.w8a8_block_fp8_matmul_triton(q, w, qs, ws, [32, 32])
            ),
        }
        chains = {
            "baseline": lambda w=w, ws=ws, x=x: base_linear.block_fp8_linear(
                x, w, [32, 32], ws, native_fp8=True, act_scale_ue8m0=True
            ),
            "candidate": lambda w=w, ws=ws, x=x: linear.block_fp8_linear(
                x, w, [32, 32], ws, native_fp8=True, act_scale_ue8m0=True
            ),
            "vendor": lambda w=w, ws=ws, x=x: vendor_linear(
                x, w, [32, 32], ws, act_scale_ue8m0=True
            ),
        }
        base_error = check["baseline_torch.bfloat16"]
        for name, fn in chains.items():
            verify_gemm(fn(), ref, base_error, rms, torch.bfloat16)
        # Refresh data in place after capture: both quantization and GEMM must read new input.
        graph, gy = capture(chains["candidate"], iterations=1)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            ey = chains["candidate"]()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(ey, gy)
        for _ in range(10):
            assert torch.equal(chains["candidate"](), ey)
        record = dict(
            **shape, correctness=check, graph_refresh_passed=True, deterministic=True
        )
        record["quant"] = measure_set(
            {
                "baseline": lambda x=x: qbase(x),
                "candidate": lambda x=x: qopt(x),
                "vendor": lambda x=x: vendor.sglang_per_token_group_quant_fp8(
                    x, 32, scale_ue8m0=True
                ),
                "pytorch": lambda x=x: quant_ref(x),
            }
        )
        if args.weight_pool_mib:
            # Rotate physical weight buffers to exceed H100 L2. Each copy has the
            # same values, so the numerical oracle above applies to every call.
            count = max(2, math.ceil(args.weight_pool_mib * 1024**2 / w.numel()))
            weight_pool = [w.clone() for _ in range(count)]

            def rotating(fn, weight_pool=weight_pool):
                index = 0

                def call():
                    nonlocal index
                    weight = weight_pool[index % len(weight_pool)]
                    index += 1
                    return fn(weight)

                return call

            gemms = {
                name: rotating(
                    lambda weight, fn=fn, q=q, qs=qs, ws=ws: fn(
                        q, weight, qs, ws, [32, 32]
                    )
                )
                for name, fn in [
                    ("baseline", base.w8a8_block_fp8_matmul),
                    ("candidate", opt.w8a8_block_fp8_matmul),
                    ("vendor", vendor.w8a8_block_fp8_matmul_triton),
                ]
            }
            chains = {
                "baseline": rotating(
                    lambda weight, x=x, ws=ws: base_linear.block_fp8_linear(
                        x, weight, [32, 32], ws, native_fp8=True, act_scale_ue8m0=True
                    )
                ),
                "candidate": rotating(
                    lambda weight, x=x, ws=ws: linear.block_fp8_linear(
                        x, weight, [32, 32], ws, native_fp8=True, act_scale_ue8m0=True
                    )
                ),
                "vendor": rotating(
                    lambda weight, x=x, ws=ws: vendor_linear(
                        x, weight, [32, 32], ws, act_scale_ue8m0=True
                    )
                ),
            }
            record["weight_pool_count"] = count
            record["weight_pool_bytes"] = count * w.numel()
        record["gemm"] = measure_set(gemms)
        record["linear"] = measure_set(chains)
        record["pytorch_gemm"] = measure_set(
            {"pytorch": lambda q=q, qs=qs, w=w, ws=ws: gemm_ref(q, w, qs, ws)}
        )["pytorch"]
        for group in ["quant", "gemm", "linear"]:
            t = record[group]
            record[group + "_speedup"] = (
                t["baseline"]["p50_us"] / t["candidate"]["p50_us"]
            )
        report["results"].append(record)
        save()
        print(
            json.dumps(
                {
                    "shape": [m, n, k],
                    **{
                        g: {a: round(b["p50_us"], 3) for a, b in record[g].items()}
                        for g in ["quant", "gemm", "linear"]
                    },
                }
            ),
            flush=True,
        )
        del graph, gy, ey, ref, w, ws, x, q, qs, oq, os, vq, vs, gemms, chains
        if args.weight_pool_mib:
            del weight_pool
        torch.cuda.empty_cache()
    report["weighted_us_per_step"] = {
        g: {
            name: sum(
                r["calls_per_step"] * r[g][name]["p50_us"] for r in report["results"]
            )
            for name in ["baseline", "candidate", "vendor"]
        }
        for g in ["quant", "gemm", "linear"]
    }
    report["finished"] = time.time()
    save()
    print(json.dumps(report["weighted_us_per_step"]), flush=True)


if __name__ == "__main__":
    main()
