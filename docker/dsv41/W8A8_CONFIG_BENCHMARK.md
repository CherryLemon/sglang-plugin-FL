# H100 block32 W8A8: config-driven integration

The FlagGems implementation is in [PR #6631](https://github.com/flagos-ai/FlagGems/pull/6631).
Exact M/N/K rows in `fp8_w8a8-32-32.yaml` select the Hopper kernel, transposed
MMA, and split-K count. The public W8A8 entry allocates FP32 partials and runs
the split-K reduction. Legacy six-value YAML rows retain nearest-M selection.
The new YAML and existing W8A8 tuning files are included in the FlagGems wheel.

The operator benchmark used one H100 80 GB, 15 shapes taken from the eight
steady-decode traces, and 230 weighted W8A8 linear calls per decode step.
It replayed 100 calls in each CUDA Graph, with 25 warmups and seven shuffled
timing rounds. Timings include quantization, GEMM, workspace, and reduction
kernels, but exclude JIT and Python launch overhead. Both BF16 and FP32 outputs
were checked against the generic implementation and an independent FP32
reference. All 15 shapes passed numerical, input-refresh CUDA Graph, and
deterministic replay checks.

| Weight access | Version | Generic baseline (ms/step) | Optimized FlagGems (ms/step) | SGLang vendor (ms/step) |
| --- | --- | ---: | ---: | ---: |
| Fixed weights, warm cache | r8 prototype | 21.071 | 11.663 | 11.555 |
| Fixed weights, warm cache | r10 config-driven | 21.045 | 11.653 | 11.536 |
| 128 MiB rotating weight pool | r8 prototype | 23.590 | 12.184 | 11.891 |
| 128 MiB rotating weight pool | r10 config-driven | 23.608 | 12.188 | 11.882 |

The config-driven dispatch preserves the prototype's weighted performance
within run-to-run variation. The standalone results and validation records
are under `/public-nvme/yjwu/sglang-fl-0518-build/w8a8-r8-runtime-node1/node1/refactor-bench/`.
The prototype reference is under
`/public-nvme/yjwu/sglang-fl-w8a8-opt-20260923/`. The checked benchmark
script is [benchmark_w8a8.py](benchmark_w8a8.py).

The official-base r10 image is
`sglang-fl-dsv41:0.5.18-flagcx-pd-flagtree-w8a8-r10`
(`sha256:07ee2d22ca37ca668b3996a7a7cc62dc03c3b18c4435ec4a75dc00506c4f5979`).
Its build passed 92 CPU tests, skipped two, and passed 63 subtests. The wheel
contains the new block32 YAML and both W8A8 modules. Build log:
`/public-nvme/yjwu/sglang-fl-0518-build/w8a8-r10-build.log`.

Before the config refactor, the r8 prototype completed two FlagCX PD rounds
with P on `.1`, D on `.3`, TP8/EP8, D attention TP2×DP4, DSpark block5, and
decode CUDA Graph. All 160 requests used 131072 input and 8192 output tokens
and finished with `length`. A second qualified only if all four DP groups
reported 20 running requests throughout that second, zero preallocated,
transferring, or retracted requests, and `cuda graph: True`. The median sum
of four DP-group generation rates was **6828.18 token/s** across 144 qualified
seconds (round medians 6799.70 and 6883.35). The earlier FlagTree hybrid
baseline was 5751.73 token/s and the vendor reference was 6890.21 token/s
under the same metric. The r8 client receipt is
`/public-nvme/yjwu/sglang-fl-0518-build/w8a8-r8-runtime-node1/node1/benchmark-80x128k8192.json`;
its scheduler analysis is `node1/steady-tps.json` in the same artifact tree.

The r10 config-driven image then repeated the same two-node PD workload with
the same prompt SHA256
`e3696762465c6a4503ffe8681de376575de9ff6fa91fe55cc14b60827c2009e9`.
All 160 requests again had exactly 131072 input tokens, 8192 output tokens,
and finish reason `length`. The scheduler reached 20 active requests in each
DP group with Graph enabled, no preallocation, no transfer, and no retractions.

| D implementation | Qualified seconds | Steady decode median (token/s) |
| --- | ---: | ---: |
| Earlier FlagTree hybrid | 168 | 5751.73 |
| SGLang vendor reference | 194 | 6890.21 |
| r8 prototype | 144 | 6828.18 |
| **r10 config-driven** | **147** | **6824.26** |

The two r10 round medians were 6794.52 and 6877.61 token/s, versus 6799.70
and 6883.35 for r8. The aggregate median is 0.06% below the prototype,
18.65% above the earlier FlagTree hybrid run, and 0.96% below the vendor
reference. These small r8/r10 differences are within run-to-run variation;
the comparison supports performance parity for the refactor. The r10 client
receipt and strict scheduler analysis are
`/public-nvme/yjwu/sglang-fl-0518-build/w8a8-r10-runtime-node1/node1/benchmark-80x128k8192.json`
and `node1/steady-tps.json` in the same artifact tree. The matching D log is
`/public-nvme/yjwu/sglang-fl-0518-build/w8a8-r10-runtime-node3/node3/decode.log`.
