# Hy4-preview framework adapter (draft)

This draft supports pure-text requests on the SGLang 0.5.18 API family.
Enable model registration
explicitly with `SGLANG_HY4_ENABLE=1` before loading the plugin. Checkpoint model
identifiers remain `HYV4ForCausalLM`, `HYV4VLForConditionalGeneration`, and
`HYV4ForCausalLMNextN`; the user-facing model name is Hy4-preview.

The change registers the text configuration and the thin wrapper required by
the checkpoint's existing VL architecture name, indexed MLA attention with
per-head sinks, W8A8 compressed-tensors MoE routing, and the checkpoint's single
bundled NextN block. The wrapper sets `language_model_only=True`, preserves text
`input_embeds` and checkpoint weight-prefix mapping, and explicitly rejects
media. Vision configuration is inert metadata only. No vision encoder, image or
video preprocessor, media processor registration, or media inference is added. Target checkpoint depth stays
unchanged when the runtime draft shape is reduced to its one actual MTP block.
CT aliases are copied before remapping, preserving the target configuration.
The DSA sink path requires `flashmla_sparse` for prefill and decode. Context
parallelism is rejected by the INT8 and multimodal indexer adapters.

Optional paths remain off until explicitly selected:

| Setting | Scope and dependency |
| --- | --- |
| `SGLANG_HY4_MROPE_COMPAT=1` | SGLang 0.5.18 position-buffer compatibility; no media support |
| `SGLANG_HY4_DRAFT_SINK_BACKEND=1` | Model-scoped native draft factory routing with attention sinks |
| `SGLANG_HY4_DRAFT_CAUSAL_LENGTHS=1` | Per-step draft causal KV lengths |
| `HY4_INDEXER_DTYPE=int8` | Requires `flag_gems.fused.int8_sparse_indexer` quantize, cache-store and paged-logits APIs |
| `HY4_SG_SMALL_M_BF16_GATE=1` | Requires `flag_gems.fused.bf16_small_m_linear`; guarded SM90 decode/verify roles and M=1/2/4/8/16 |
| `SGLANG_HY4_INDEXER_BF16_MM=1` | Native Torch CUDA MM only, M=208/256, K=2048, N=4096, BF16, exact provider/layout and existing math-state guards |
| `SGLANG_HY4_BF16_SPARSE_MLA_ACTIVE_SPLIT_TARGET=1` or `_DRAFT=1` | Requires the public `flaggems_sglang.sparse_mla_active_split` API and matching `get_op`; SM90, Q rows 1/4, four active heads, physical 64 heads, D=576/V=512, selected width 2048 |
| `SGLANG_HY4_IHC_POST_PRE=1` | Optional mature `flaggems_sglang.ops.ihc_post_pre` API; currently not part of the sparse-MLA operator PR |

Numerical device kernels belong to operator libraries. The active-head MLA API
is proposed separately in FlagGems-sglang. The INT8 indexer and small-M linear
APIs are additional FlagGems source dependencies; their presence must be checked
in the installed operator build. Stock upstream FlagGems does not currently
provide the INT8 indexer API. An explicit INT8 request, or auto selection on a
non-SM90 device, requires that additional build and otherwise raises a clear
import error; this PR does not claim out-of-box support for those devices.
Missing MLA registration or unsupported
metadata selects the original sparse-MLA provider before launch. The MLA route
does not import a missing provider for the first time during CUDA capture.
Numerical launch exceptions propagate rather than being retried through native.

The separate iHC post/pre candidate is not registered by the current
FlagGems-sglang sparse-MLA PR. If its module is absent, the optional boundary
selects the original post then pre route. The general pre/post/head adapter uses
an optional HPC implementation where its API/device/shape domain is available;
otherwise the existing Torch model formula is retained. This framework PR
contains no copied iHC Triton kernels. This source separation changes the
available provider set relative to the private experiment and needs fresh GPU
qualification before serving use.

No job-local NCCL tuning, worker-wide TF32/BF16 reduction mutation, profiler
injection, request/step receipt, row audit, loader driver or private path is
included. The indexer-MM path reads the caller's existing math state; it never
changes it. Unsupported states retain the existing linear provider.

## Validation and limits

CPU tests cover strict per-runner role selection, default-off behavior,
prefill/draft-extend fallback, checkpoint/MTP remapping without target mutation,
CPU MM fallback without math-state mutation, and unavailable HPC fallback.
Actual unstubbed SGLang imports and all three architecture registrations, real
checkpoint metadata parsing, CPU text embedding/prefix fixtures and media
rejection pass with CUDA visibility disabled. These tests load no real model
weights. Optional accelerator import warnings do not establish device execution
qualification.
Relative imports, architecture registration, source syntax, bounded provider
shape guards and removal of experiment instrumentation are also checked.

The source family was exercised with a real 78-layer TP16/EP16 checkpoint, but
strict batch/teacher-forced numerical diagnostics did not pass. q1 and q4 also
produced different continuations for identical input and seed. An unprofiled
combined C1 trial reported mean TPOT 7.869 ms; its output differs from the earlier
11.945 ms trial from token 3, and TTFT changed from 192.227 to 505.826 ms. These
are diagnostic observations, not an equivalent-output speedup or validation of
this framework PR. The sanitized package has no new integrated GPU run. Media
inference is unsupported. Breakable prefill graphs, other devices, CP, larger active-head
MLA row counts and full-model correctness remain unqualified. Keep this PR draft.
