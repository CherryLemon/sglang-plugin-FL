# DeepSeek V4.1 Flash / FlagCX PD on SGLang 0.5.18

Prefill runs on `aiops-10-8-2-1` (`10.8.2.1`) and decode on
`aiops-10-8-2-3` (`10.8.2.3`), each with eight H100 80 GB GPUs, global TP8/EP8,
DSpark block 5 and decode CUDA Graph. The default FlagCX deployment is
homogeneous attention TP8/DP1. Set `PD_DECODE_DPA=1` only on D to opt into the
historical attention TP2/DP4 layout; the V4.1 topology guard admits this exact
decode configuration with a TP8/DP1 prefill peer.

The build lock in [manifest.json](manifest.json) pins the official
`lmsysorg/sglang:v0.5.18` image, the `dev/0.5.18` based plugin branch, FlagGems,
FlagCX source and NVIDIA library SHA256. FlagCX is built from the pinned source:

```bash
make USE_NVIDIA=1 \
  CCL_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/nccl \
  DEVICE_HOME=/usr/local/cuda -j16
python docker/dsv41/build_image.py \
  --flaggems /absolute/path/FlagGems-dsv41 \
  --flagcx /absolute/path/FlagCX-dsv41 \
  --wheelhouse /absolute/path/wheelhouse \
  --output /absolute/path/new-build-context \
  --tag sglang-fl-dsv41:0.5.18-flagcx-pd
```

`--flagcx` copies the locked `libflagcx.so` and Python wrapper into the image.
The FlagCX PD transfer uses `FLAGCX_PATH=/opt/FlagCX`; TP/EP collectives stay on
NCCL through `SGLANG_FL_DIST_BACKEND=nccl`. The recipe sets
`FLAGCX_SOCKET_IFNAME=bond0` so FlagCX's RPC listener binds to the same
`10.8.2.x` address that SGLang advertises, and uses `mlx5_bond_0` for RDMA
on both hosts. `bond0` is the TCP control interface; `mlx5_bond_0` is the
ACTIVE RDMA device with an Ethernet link layer (RoCE), so the two names serve
different purposes. Both sides must use the same model revision, page size,
DSPARK block size, global TP and V4.1 cache layout.

The deployment scripts are [prepare_flagcx_pd.py](prepare_flagcx_pd.py),
[sync_flagcx_pd_runtime.py](sync_flagcx_pd_runtime.py),
[launch_flagcx_pd.sh](launch_flagcx_pd.sh),
[launch_flagcx_router.sh](launch_flagcx_router.sh) and
[smoke_flagcx_pd.py](smoke_flagcx_pd.py). The two-node GPU bring-up uses the
previously imported OCI runtime image plus `sync_flagcx_pd_runtime.py` on each
host because those hosts' older Docker cannot import the canonical official-base
image. The new official-base Docker image is built and validated separately;
runtime source hashes are checked after sync. FlagCX itself is mounted read-only
from the pinned checkout for this bring-up.

The endpoint layout is prefill `http://10.8.2.1:31819`, decode
`http://10.8.2.3:31820`, router `http://10.8.2.1:31821`. Send client requests
through the router. The launch scripts use `serve_single_node.sh` with the PD
role and `--disaggregation-transfer-backend flagcx`, so the existing MTP and
decode graph recipe remains enabled. The router's `--mini-lb` is adequate for
this 1P1D experiment.

The V4.1-specific changes are: allow FlagCX under the DSpark/static-verify
topology guard, with opt-in TP2/DP4 decode; register contiguous indexer pages for transfer;
ship C128 and SWA-ring state at the correct physical strides; reject incompatible
C2 receiver strides and state index counts; and report failed state writes as a
failed PD transfer instead of a successful one.

Validation receipts and logs are under `/public-nvme/yjwu/sglang-fl-0518-pd`.
The original official-base PD image build log is `image-build-r5.log` (92 CPU tests passed,
2 skipped; 63 subtests passed). Its tag is
`sglang-fl-dsv41:0.5.18-flagcx-pd-r5`. Its
`/opt/FlagCX/build/lib/libflagcx.so` SHA256 matches the manifest. The first startup failure
caused by the missing indexer registration method is preserved as
`node1/prefill-attempt1-indexer-missing.log`.

The first homogeneous run inherited the small single-node limit
(`max_total_tokens=131072`, `max_running_requests=16`): a 16-client 32K burst
had only about three active decode requests, so its TPS does not measure
16-way decode capacity. With `context_length=655360`,
`max_total_tokens=1048576`, and `max_running_requests=128`, the homogeneous D
started with 6.89 GB free per GPU after CUDA Graph capture. In a 16-client
32K/128-output-token burst, its scheduler reached `#running-req: 16`,
`#prealloc-req: 0`, `cuda graph: True`; all 16 requests completed. The receipt
is `node1/pd-decode-trial-cap1m-tp8.json`, and the server log is
`node3/decode-attempt5-tp8dp1-cap1m.log`. This proves the earlier three-active
limit was a capacity configuration error; it does not establish the historical
80-way 128K result, which uses attention TP2/DP4 on D.

With D attention TP2/DP4 and the same 1M-token pool, one exact 131072-token
request passed, then an 80-client burst with fixed 8192-token output completed
80/80 requests. This is **not** an 80-active-request result: D peaked at seven
running requests and 13 preallocated requests in each DP group (28 active,
52 waiting), with zero retractions. The minimum per-request streamed decode rate
was 94.86 tokens/s and the median was 110.00 tokens/s; those rates mix different
occupancy phases. The receipt is `node1/pd-decode-80x128k8192-dpa.json`, and the
log is `node3/decode-attempt6-dpa-ratio-cap1m.log`.

The public V11 manifest's 1M-token setting belongs to its 32K workload; the
V12 80×128K report does not publish an equivalent pool setting. The capacity
regression seen here was in the V4.1 SWA pool budget. Its compressed pool
config declares a 128-token sliding window, but the generic runner reports no
window for this model. The model override also replaces the CLI default SWA
ratio with 0.1, unintentionally disabling the request-capped pool mode. A 3M
full-token attempt with that ratio failed CUDA Graph capture with GPU OOM; its
log is `node3/decode-attempt7-dpa-ratio3m-oom.log`. Patch
`0014-dsv41-flagcx-dpa-capacity.patch` preserves the CLI default only for the
opt-in TP2/DP4 decode configuration, uses the model's declared window, and
excludes disabled decode-radix prefix tails. An explicit operator ratio still
selects ratio sizing.

At `max_total_tokens=3145728`, `max_running_requests=128`, and
`mem_fraction_static=0.96`, D started with `mode=cap`, a 3,145,728-token full
pool and 61,952-token SWA pool per worker. Target and draft CUDA Graph capture
completed with about 1.2 GB free per GPU. One 131072-token input/128-token
output request passed (`node1/pd-decode-trial-128k-cap3m.json`). Two 80-client
rounds with the same exact 131072-token input and 8192-token fixed output
passed 160/160 requests. The D log repeatedly records 20 active requests in
each DP group (80 total), zero preallocated waiting requests, zero retractions,
`cuda graph: True`, and about 88% full-token pool use at peak. Receipts are
`node1/pd-decode-80x128k8192-dpa-cap3m.json` and
`node3/decode-attempt8-dpa-cap3m-hybrid.log`.

| Round | Min request decode TPS | Median request decode TPS | Aggregate decode TPS |
|---|---:|---:|---:|
| 1 | 67.20 | 76.66 | 5088.37 |
| 2 | 67.20 | 75.98 | 5076.85 |

The per-request rate uses `(output_tokens - 1)/(last_content - first_content)`.
The original V12 experiment reported minimum 101.47/103.19 and median
106.17/106.75 TPS at the same concurrency. This migration restores 80-active
capacity but has a measured decode performance gap. The coding prompt here is
deterministic but not the private V12 prompt, and this benchmark forces 8192
output tokens with `ignore_eos=True` rather than natural EOS. Both differences
must be kept in mind when comparing the rates.

To locate the regression, D alone was restarted with
`SGLANG_FL_DSV41_BACKEND=vendor`; P, FlagCX, TP2/DP4, 3M full pool and graph
settings remained the same. One initial 80-client round and two following
rounds all completed at 20 requests per DP group, zero preallocated waiting
requests, zero retractions and `cuda graph: True`:

| D backend / round | Min request decode TPS | Median request decode TPS | Aggregate decode TPS |
|---|---:|---:|---:|
| vendor initial | 78.87 | 97.69 | 5940.74 |
| vendor subsequent 1 | 88.61 | 98.02 | 6597.30 |
| vendor subsequent 2 | 84.32 | 97.88 | 5652.46 |

The receipts are `node1/pd-decode-80x128k8192-dpa-cap3m-vendor.json`,
`node1/pd-decode-80x128k8192-dpa-cap3m-vendor-round2.json` and
`node1/pd-decode-80x128k8192-dpa-cap3m-vendor-round3.json`; the server log is
`node3/decode-attempt9-dpa-cap3m-vendor.log`. Vendor dispatch improves median
TPS from about 76 to 98, but still misses the V12 minimum-per-request
threshold. The hybrid log contains 88 missing FP8 GEMM tuning-config warnings;
the FlagGems H100 block32 path currently uses default tiles for those shapes,
while the source V12 branch carries shape-specific and Split-K configurations.
That is a concrete tuning lead, not a per-kernel attribution. D was restarted
back in the default hybrid mode after this A/B.

A follow-up [steady-decode operator profile](PROFILE_STEADY_DECODE.md) captured
10 Graph/MTP forward steps in both modes at 80 active requests. Across eight
D ranks, FlagGems W8A8 quantization and GEMM cost 25.14 ms/step versus
13.84 ms/step for the V4.1 vendor path; FP4 scoring differs by only
0.27 ms/step and mHC by 0.39 ms/step. The profile uses the same prompt and
capacity configuration as the receipts above. The out-of-tree platform's
PyTorch profiler activity hook was fixed so `/start_profile` can capture
the running service.
The follow-up image `sglang-fl-dsv41:0.5.18-flagcx-pd-r6` contains that hook;
its image ID is `sha256:6cac4057cd850a5bcfeaf416553ae28469251b3663cfa00aa7e53f2cafb2a56a`.
The build passed 92 CPU tests (2 skipped, 63 subtests) and the new image's
profiler activity mapping was verified directly. Build log and receipt are
`/public-nvme/yjwu/sglang-fl-0518-build/image-build-r6.log` and
`build-result-r6.json` in the same directory.
