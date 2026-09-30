# Qwen3.6 S5000 validation handoff

Run from the repository root with the pinned image's TorchMUSA interpreter.
The supported stack and historical results are recorded in
[the MoE integration notes](../../sglang_fl/dispatch/backends/vendor/mthreads/moe/README.md).
These commands validate the current checkout; the historical hardware results
at `f4bec31` do not validate eventfd removal or the later dispatch refactor.
Save the command, full logs, installed plugin commit, model revision, MATE revision and image digest alongside each result.

The current S5000 acceptance requires the isolated MATE GDN H-ready patch
described [below](#mate-gdn-prefill-synchronization-check). Both tested MATE
versions exhibit C4 drift without it. Installing this plugin alone does not
apply that dependency patch; retain its source hash with the runtime identity.

## Install and run regression checks

```bash
python -m pip install --no-deps --no-build-isolation .
python -m pytest -q tests/unit_tests/
python -m compileall -q sglang_fl/dispatch/backends/vendor/mthreads tests/musa
python -m pytest -v -rs tests/functional_tests/ops/test_musa_qwen36_paths.py
```

The six functional cases require S5000. On the target image a missing Triton API
or a failed guard is a failure, not a successful fallback. Confirm the GPU cases
actually ran; a skip on a CPU host is not GPU acceptance.

- TopK uses real Config/Autotuner/JITFunction objects at M4/M64/M2048, observes
  a pinned launch, and compares against the original kernel. IDs must be exact;
  weights use `rtol=1e-6, atol=1e-7` against the autotuned path. Replay must match
  the pinned eager output exactly after fixed-address input refresh (0/1/0).
- Combine covers eager M2048 and the two-stream reduction seam at M40/M64.
  It checks actual consumption, exact dyadic arithmetic, fixed-address refresh,
  repeated replay, explicit disable fallback and context cleanup. The simple
  inputs isolate indexing/stream behavior; this is not the full random-input,
  shared gate-tail, NaN/Inf, M16K workspace or model accuracy matrix.

## TP2 full-graph model smoke

Use these candidate controls in the candidate runtime only. Apply the retained
reference's own launch environment when collecting reference records. Keep the
same physical cards, model weights, memory fraction and request protocol for
both arms.

```bash
export SGLANG_FL_FLAGOS_BLACKLIST=mul,sort
export SGLANG_MUSA_MATE_GDN=auto
export SGLANG_MUSA_MATE_GDN_PREFILL=auto
export SGLANG_MUSA_TOPK_SCHEDULE=auto
export SGLANG_MUSA_MOE_DECODE_SCHEDULE=auto
export SGLANG_MUSA_MOE_PREFILL_SCHEDULE=auto
export SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE=1
export SGLANG_MUSA_M4_R2_BASELINE_SCHEDULE=1
export SGLANG_MUSA_M4_W13_BN64=1
export SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE=1
export SGLANG_MUSA_SHARED_EXPERT_GATE_TAIL_FUSED=1
export SGLANG_MUSA_DETERMINISTIC_MOE_COMBINE=auto
export SGLANG_MUSA_DETERMINISTIC_MOE_COMBINE_DECODE_GRAPH=1
export SGLANG_MUSA_CUSTOM_AR_FUSED_RMSNORM=1

python tests/run.py --platform musa --device s5000 --scope e2e \
  --model qwen3_6 --case 35b_a3b_tp2_graph --task concurrent
```

The new YAML enables BF16/TP2, full decode graphs, all 12 production capture
buckets and custom all-reduce. Piecewise graphs and radix cache are disabled;
the latter is required by the MATE prefill adapter. The 0.90 memory fraction
is a conservative smoke setting, not a performance claim. If adjusted, use
the same value in both comparison arms. The YAML model path can be edited for
the host; the repeatability script below also accepts `--model-path`/`MODEL_PATH`.

Capture logs should identify TopK installation, routed-MoE selection, shared
gate-tail selection and both combine paths. Check actual path hits as well as
installation. Short prompts do not exercise long-prefill combine or M16K
workspace reuse; run the retained fixed-work/long-prompt checks for those gates.
Launch-submitted logs alone do not establish replay correctness.

## Record C1/C4 repeats and compare

`validate_qwen36.py` launches an Engine in the current interpreter/environment.
It does **not** change plugin selection based on `--label`: a retained reference
may itself require the plugin and core overlays. `candidate` additionally checks
that the plugin entry point activated. Run each arm separately so monkeypatches,
graphs and process state cannot leak between them.

Set `IMAGE_DIGEST`, `MATE_REVISION`, and `PLUGIN_COMMIT` to the actual installed
runtime identities for **each** arm. `PLUGIN_COMMIT` is distinct from the git
commit containing this validation script. For an editable candidate checkout it
can be obtained with `git rev-parse HEAD`; for a wheel use its build record.

```bash
# Run in the retained reference image with its retained launch environment.
python tests/musa/validate_qwen36.py record \
  --label reference \
  --engine-config tests/models/qwen3_6/35b_a3b_tp2_graph.yaml \
  --image-digest "$IMAGE_DIGEST" --mate-revision "$MATE_REVISION" \
  --plugin-commit "$PLUGIN_COMMIT" \
  --rounds 3 --concurrency 1 4 --output /tmp/qwen36-reference.json

# Run in the candidate image with the controls above.
python tests/musa/validate_qwen36.py record \
  --label candidate \
  --engine-config tests/models/qwen3_6/35b_a3b_tp2_graph.yaml \
  --image-digest "$IMAGE_DIGEST" --mate-revision "$MATE_REVISION" \
  --plugin-commit "$PLUGIN_COMMIT" \
  --rounds 3 --concurrency 1 4 --output /tmp/qwen36-candidate.json

python tests/musa/validate_qwen36.py compare \
  /tmp/qwen36-reference.json /tmp/qwen36-candidate.json
```

Defaults use four short raw prompts, one excluded warmup and greedy decoding
with exactly 32 output tokens (`ignore_eos=True`). This is a repeatability smoke
protocol, not the historical 1024/32 workload. To reproduce that workload, supply
the same `--prompts /path/to/prompts.json` (a JSON list of rendered strings) to
both arms and verify input lengths with the pinned tokenizer. `--concurrency`
is the maximum simultaneously submitted requests; it does not prove the GPU
scheduler used that exact batch size. The final wave can be smaller than the
requested concurrency. Use `--eager` and new output filenames for a separate
eager comparison; compare files with identical engine settings.

The script stores native token IDs/logprobs, submitted and returned request IDs,
round/concurrency/prompt indices, package versions and selected environment
controls. Completed waves are saved incrementally. Existing output files are
never overwritten, and partial/error/missing-token records cannot pass comparison.
Output IDs and logprobs are compared exactly, independently for reference
self-drift, candidate self-drift and between-service differences. Exit 0 means
these records matched, 1 means drift/mismatch, and 2 means incomplete records or
an invalid comparison. Self-drift in the reference remains visible even if both
arms exhibit the same drift; it does not establish candidate equivalence.

Keep failed C4 records. Formal model accuracy, the full fixed-work matrix and
final-image performance acceptance remain separate gates.

## MATE GDN prefill synchronization check

The unpatched pinned MATE 0.2.4 compatibility kernel and official MATE 0.2.7
both exhibit C4 nondeterminism on the tested S5000 stack. A dependency upgrade
alone does not resolve it. See [the acceptance record](acceptance_20260920.md)
for controlled experiments, source hashes and the current acceptance status.

[mate-gdn-h-ready.patch](mate-gdn-h-ready.patch) separates consecutive H-ready
publications across two physical barriers. Both consumers can otherwise wait
on different phases of the same physical barrier. The patch preserves the
shared-state buffer and computation; it does not add a compute rendezvous.
It is a MATE source patch, not an automatic plugin monkeypatch.

Apply it only to an isolated copy of one of the recorded MATE sources, using
that source's matching TileLang/TVM-FFI dependencies. The parent directory below
must contain `mate/gdn_kernels/tilelang/gdn_prefill.py`. Verify the source hash
against the acceptance record before applying it, and retain the patched hash
and package import origins with the test output.

```bash
MATE_PACKAGE_PARENT=/path/to/isolated/mate-copy
MATE_PATCH_PATH="$(realpath tests/musa/mate-gdn-h-ready.patch)"
(
  cd "$MATE_PACKAGE_PARENT"
  git apply --check "$MATE_PATCH_PATH"
  git apply "$MATE_PATCH_PATH"
  sha256sum mate/gdn_kernels/tilelang/gdn_prefill.py
)

PYTHONPATH="$MATE_PACKAGE_PARENT" \
TILELANG_CACHE_DIR=/path/to/fresh/tilelang-cache \
python tests/musa/repro_mate_gdn_prefill.py --output /tmp/mate-gdn-patched.json
```

The reproducer uses seeded synthetic inputs and an independent CPU FP32
recurrence. It covers chunk boundaries, variable lengths, split-QKV layouts,
initial states and 100 repeats per case after a 128 MiB device fill. It checks
output/state repeatability, untouched inputs and complete overwrite of poisoned
outputs. `--smoke` selects C4/I1024; `--kernel-file` selects a standalone
candidate without changing the installed package. Existing result files are
refused. This is a correctness probe, and its elapsed time is not a benchmark.
The full-model and matched performance gates must still pass with the exact
dependency source that will be delivered.

## Current validation status

The current code uses the core Event completion path. It has not been
revalidated on S5000 after eventfd removal and the unified-dispatch refactor.
Historical benchmark detail and previous local checks are preserved in the
[immutable record](https://github.com/CherryLemon/sglang-plugin-FL/blob/b3cd3b1d099d0a2e315680d43ab63f2cbae4a2ea/tests/musa/README.md).
See [the evidence summary](acceptance_20260920.md) for tested identities, C4
findings, dependency hashes and limits.

`qwen36_perf.env` retains the opt-in runtime controls for reproducing the
campaign configuration, now without eventfd. Set the model path, visible
devices, communication interface and isolated patched MATE path separately.
The explicit environment blacklist replaces the YAML list. Its filename and
historical origin do not establish current performance acceptance.

On the target image, verify common dispatch preference/per-op/vendor filters
for `fused_moe`, `shared_expert_gate_tail`, `moe_sum_reduce` and
`allreduce_rms_norm`. The last three use the common resolver before execution,
so a failed in-place launch/collective is not retried by the dispatch manager.
Keep the full existing CPU and GPU test suites; GPU skips are not acceptance.
