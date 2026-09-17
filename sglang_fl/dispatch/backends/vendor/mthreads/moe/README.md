# Qwen3.6 MUSA plugin-side integration

Status: **offline-tested candidate; GPU correctness, dispatch, memory and
performance revalidation are still required.** Historical service throughput
must not be attributed to this source layout before that validation.

## Baseline and ownership

The reference is the SGLang 0.5.11 MUSA vendor image, with core commit
`612785ffdcaf35552f1ed433a981d596ca9fe900` **plus its existing vendor changes**.
This is not an unmodified upstream SGLang checkout. The campaign added nine
core files on top; this candidate moves those additions into the plugin:

| Former core change | Plugin owner |
| --- | --- |
| `environ.py` capability flag | `moe/dispatch.py`, reads the existing opt-in environment variable |
| MUSA FA3 capture metadata | `patches/fa3_graph_metadata.py` |
| Triton runner quant-info field | Existing MThreads fused-MoE dispatch, no dataclass ABI change |
| MoE sequence, M4 scheduling, M16K scratch | `moe/fused_moe.py` |
| Fused SwiGLU kernel and launcher | `moe/kernels.py` |
| Shared expert gate-tail kernel | `moe/shared_expert_gate_tail.py` |
| Unquantized post-load capability | Loaded-weight guards in `moe/dispatch.py`, no weight mutation |
| ModelRunner pre-KV allocation | `patches/moe_workspace.py` |
| Qwen shared-expert call site | `patches/shared_expert_gate_tail.py` |

No installed SGLang file is rewritten at startup; no copied `sglang` package
is put ahead of the original on `PYTHONPATH`. Existing vendor-image core
changes remain a pinned dependency, not a new plugin-only claim.

`provenance.json` records the campaign source hashes and normalized AST hashes
for 20 relocated function bodies, including the kernel/launcher. Custom-op
names are distinct from SGLang's registrations. Scheduling, route alignment
and reduction are resolved through the original module at call time so the
existing plugin patches (especially deterministic combine) remain effective.

## Supported candidate contract

Qwen3.6-35B-A3B BF16, MTT S5000, TP2, local 256 experts, top-k 8, canonical
W13 `(256, 512, 2048)` and W2 `(256, 2048, 256)`, no EP dispatch, LoRA, EPLB,
router-input weighting, quantization, bias or fused MoE all-reduce. The
release path uses eager prefill and full decode graphs; piecewise graphs are
not admitted by the adapter. Unsupported calls keep `forward_musa` unchanged.

The routed adapter rechecks loaded weights rather than caching a capability
in a core dataclass. A launch/allocation error is propagated, never retried
against possibly modified in-place inputs.

Retained opt-in controls:

- `SGLANG_MUSA_FUSE_MOE_SWIGLU_EPILOGUE=1`: canonical-W13 fused prefill;
  the kernel sequence still disables fusion below 2048 tokens.
- `SGLANG_MUSA_M4_W13_BN64=1`: exact M4 up tile; original down/alignment config
  is not modified. The reference also enables the R2 M4 baseline resolver.
- `SGLANG_MUSA_M16K_MOE_PREALLOCATE_DOWN_WORKSPACE=1`: reserve the 512 MiB
  fixed M16K routed-down scratch before KV-memory sizing, under the exact
  model/TP2/BF16 guard. The hook runs just before `init_memory_pool`; relative
  ordering against deterministic-mode setup is different from the former
  inline call and requires checking if that mode is enabled.
- `SGLANG_MUSA_SHARED_EXPERT_GATE_TAIL_FUSED=1`: narrow text-only M4 decode
  gate-tail path, with the original router, in-place addition and TP collective.
- `SGLANG_MUSA_FA3_GRAPH_CAPTURE_CAPACITY=0`: diagnostic opt-out of the paged
  graph capacity correction. It is on by default for the supported inherited
  MUSA capture API. Existing backend overrides and changed signatures are
  left untouched.

The compiler contract is FlagTree 0.6.1+mthreads3.6 / Triton 3.6.0, not vendor
Triton 3.2. No T3.2 compatibility overlay is part of this release candidate.
The service also depends on the separately pinned MATE compatibility package;
moving the MoE code does not replace or remove that dependency.

## Required validation before release

Run the platform unit tests; then on the pinned image verify installation and
actual hook/dispatch hits, immutable baseline core hashes, loaded-weight and
route parity, eager/full-graph outputs, BF16 materialization boundaries, M4
shared-expert reduction ownership, and M16K workspace allocation before KV
sizing. Recapture source-location-sensitive compiled artifacts and benchmark
the candidate itself. The existing engineering matrix is not the formal
FlagRelease performance/accuracy acceptance matrix.

Operator handoff: FlagGems owns the fused SwiGLU and gate-tail device kernels;
FlagTree owns scheduling/fusion selection and graph/dispatch integration.
The unchanged upstream fallback remains the compatibility boundary.
