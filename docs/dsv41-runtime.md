# DSV4.1 long-prefill memory and FlagCX P/D runtime

Long prefill previously padded the entire score matrix before reducing each
candidate block. Even an aligned width copied the full matrix. The new path
reduces complete blocks directly and pads only the final partial block.
The prefill loop also releases each score tensor after its consumers have
been enqueued, before allocating the next chunk. KV capacity is unchanged.

FlagCX carries P/D KV and auxiliary state writes. Native Mooncake retains
bootstrap, metadata, completion/ACK and failure state; Mooncake Store remains
an independent cache layer. The FlagCX library and its Python wrapper require
no changes. This adapter reuses their existing P2P API and checks the qualified
library and wrapper hashes before serving.

## Ownership and dependencies

| Component | Repository |
|---|---|
| Common GEMM, quantization, scoring and mHC math | FlagGems |
| SGLang candidate addressing, TopK and metadata kernels | FlagGems-sglang |
| PD protocol, score lifetime and serving configuration | SGLang-plugin-fl |
| Compiler | Unmodified FlagTree, Triton API 3.6.0 on prefill |
| P2P library | Unmodified FlagCX `648a6c489d2d54870d173926eda2d0ea0771b39d` |

The patches target the exact source hashes in the shipped manifest, rather
than all packages carrying the same SGLang version. They require the reviewed
DSV4.1 core interfaces. An unrelated vanilla release is rejected. No model
weights, serving aliases or request data are distributed in this package.

## Offline installation

Install this plugin, FlagGems and FlagGems-sglang at the pinned revisions in
`docs/dsv41-dependencies.json`. Prepare an offline tree whose parent contains
`sglang/`, before launching any workers:

```bash
python -m sglang_fl.dsv41_runtime.apply_framework_patches /path/to/site-packages prefill-memory
python -m sglang_fl.dsv41_runtime.apply_framework_patches /path/to/site-packages flagcx-pd
python -m flaggems_sglang_plugin.apply_operator_patches /path/to/site-packages stable-topk
python -m flaggems_sglang_plugin.apply_operator_patches /path/to/site-packages flagtree36-metadata
```

Each command verifies all input files, applies the patch to a temporary tree,
and verifies every output hash before writing. Source mismatches leave the
original tree intact. Apply only the required profiles; metadata compatibility
is for the qualified FlagTree 3.6 prefill environment. Patch application is not
a live worker reload. Rebuild/install the image and restart workers normally.

For the new PD adapter set `SGLANG_DSV41_FLAGCX_PD=1`, `FLAGCX_PATH` and
`SGLANG_HOST_IP`, select `--disaggregation-transfer-backend flagcx`, and load
the `sglang_dsv41_flagcx` plugin. Disable the older blanket PD replacement with
`SGLANG_FL_DISAGG_FLAGCX=0`. A new receiver advertises `flagcx://host:port`.
Only a migration sender may set
`SGLANG_DSV41_FLAGCX_ROLLING_ALLOW_LEGACY=1` to admit existing untagged Mooncake
receivers. A FlagCX error never retries on the legacy transport. Remove the
migration switch after draining the old receivers.

The qualified prefill allocator configuration is opt-in:

```bash
PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,per_process_memory_fraction:0.99
```

Set it before process creation with the native CUDA allocator. It allows
native allocator garbage collection; it does not increase the KV pool or
change decoder allocator settings.

## Validation

The original deployed memory patch passed 4,536 inputs through both selector
interfaces (9,072 exact output comparisons), including strided tensors, ties,
sentinels, NaN, aligned widths and partial tails. Large cases reduced selector
temporary peak memory by about 75%. Cold 16K/128K/477K/960K requests completed
without allocator retries or OOM. Two sampled evaluation passes completed 192
natural-stop calls. These are deployment qualification results, not a new
benchmark of the repackaged PR head.

This PR adds CPU tests for patch preconditions/output hashes and all-or-nothing
validation, plus nine protocol tests covering explicit legacy admission,
host registration, malformed descriptors and failure propagation. The
packaged import-path change and a fresh distributed image need a final GPU
integration run before this draft is marked ready.
