# MUSA native custom all-reduce

The framework owns the native communication kernels, JIT compilation,
communicator, IPC mappings and graph-buffer lifecycle. These remain in the
plugin; FlagGems-sglang contains only Triton/FlagTree implementations.

The original native sources are from SGLang MUSA commit
`cf872e6886f3f3babd9a942176f1b93b9174fb61`. See [source attribution](csrc/NOTICE)
and the [original Apache-2.0 license](csrc/LICENSE).

The explicit-input graph launcher is the default. Registered-input recapture
requires the matching graph-runner handshake and is enabled explicitly with
`SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT=1`.
