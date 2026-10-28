# Qwen3.6-35B-A3B on MTT S5000

This integration targets BF16, TP2 and the MUSA SGLang 0.5.11 vendor runtime
with Torch/TorchMUSA 2.9, FlagTree's MUSA Triton 3.6 compiler, TorchAda, MATE
and FlagCX. It does not install or replace the vendor Torch/compiler stack.
Install the plugin's `musa` extra for the pinned FlagGems-sglang dependency.
Native custom all-reduce/JIT, IPC registration and graph buffers remain in the
plugin; FlagGems-sglang supplies only Triton operators.

## Run

```bash
python -m pip install '.[musa]'
FLAGCX_PATH=/sgl-workspace/FlagCX MODEL_PATH=/models/Qwen3.6-35B-A3B \
  bash examples/musa/qwen36_tp2.sh --port 30000
```

The example sources the vendor runtime compatibility settings in
`examples/musa/qwen36_runtime.env`. Set visible cards, communication interface
and any isolated dependency paths before launching. `FLAGCX_PATH` must point
to the built FlagCX tree; selecting `flagcx` alone does not load that dependency.
When FlagCX is unavailable, the MUSA all-reduce hook preserves the upstream
coordinator's custom-AR/PyMCCL selection, including its graph context. Add `--disable-cuda-graph`
for eager execution. Normal decode uses the upstream model forward; the
experimental decode-graph combine path has been removed. Eager long-prefill
combine and the guarded M4 shared expert/MoE paths select automatically.

## Tuning and dispatch

`sglang_fl/dispatch/config/musa.yaml` owns upstream SGLang/TorchAda MoE and
TopK launch profiles under `kernel_tuning`. Shape, dtype and S5000 guards still
apply. Backend preference, per-op order and vendor filters remain independent
and continue to use the common dispatch policy.

`SGLANG_FL_CONFIG` replaces the whole platform configuration, including tuning.
A missing tuning profile retains the upstream selector; an invalid selected
profile raises before launch. Configuration is cached by path, so restart the
process after editing a file. The loader returns copies to callers.

Migrated kernels have their defaults in FlagGems-sglang's
`runtime/backend/_mthreads/tune_configs.yaml`; the plugin passes semantic
profile names through public APIs. GEMM routing block size is fixed before
metadata preparation and cannot be changed by an operator epilogue profile.
Shared/combine APIs also support explicit `launch_options` overrides.

## Regression checks

Run with the vendor runtime's Python interpreter:

```bash
python -m pytest -q tests/unit_tests/
python -m pytest -v -rs tests/functional_tests/ops/test_musa_qwen36_paths.py
```

The functional cases exercise real TopK launches and eager/graph reduction
with fixed-address input refresh and poisoned outputs. Run the three migrated
operator tests and benchmarks in the matching FlagGems-sglang checkout as well.
GPU skips do not establish GPU correctness. Record current commits, model
revision, image digest and dependency origins alongside results in PR/CI
artifacts; historical measurements do not validate a new checkout.

`validate_qwen36.py record` captures token IDs/logprobs for repeated C1/C4
requests. Run reference and candidate in separate processes with identical
model, dependencies, engine options and sampling settings, then use `compare`.
Use `--eager` for a separate eager comparison. See `record --help` for required
runtime identity arguments. This repeatability check is separate from model
accuracy or serving performance evaluation.

## MATE GDN synchronization dependency

The retained MATE compatibility implementation requires the explicit
[H-ready barrier patch](mate-gdn-h-ready.patch) for concurrent prefill
repeatability. A MATE version upgrade alone does not establish this property.
The plugin does not patch an installed dependency automatically.

Apply the patch to an isolated compatible source tree containing
`mate/gdn_kernels/tilelang/gdn_prefill.py`, using that tree's matching
TileLang/TVM-FFI dependencies:

```bash
git -C /path/to/isolated/mate-tree apply --check /absolute/path/to/mate-gdn-h-ready.patch
git -C /path/to/isolated/mate-tree apply /absolute/path/to/mate-gdn-h-ready.patch
PYTHONPATH=/path/to/isolated/mate-tree \
  python tests/musa/repro_mate_gdn_prefill.py --output /tmp/mate-gdn.json
```

The reproducer checks an independent FP32 recurrence, refreshed/poisoned
buffers and repeated execution. Retain the exact patched source identity with
full-model results. A passing operator reproducer alone does not prove C4
full-model repeatability.
