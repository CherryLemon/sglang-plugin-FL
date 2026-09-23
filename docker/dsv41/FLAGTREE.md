# FlagTree for the SGLang 0.5.18 image

FlagTree is a Triton fork and exports the Python module `triton`. The NVIDIA
wheel in this delivery comes from `flagos-ai/FlagTree`, branch
`triton_v3.7.x`, commit `dbf184230982e2f7cbe6b91fa3ca1ea069833d21`. Its Triton API version is
`3.7.1`, matching the official SGLang 0.5.18 base image. The wheel filename and
SHA256 are locked in [manifest.json](manifest.json).

The final image installs that wheel under `/opt/flagtree` and sets
`PYTHONPATH=/opt/flagtree`. This makes FlagTree's `triton` the default while
leaving the official Triton 3.7.1 distribution on disk. To inspect the selected
compiler, run:

```bash
docker run --rm --entrypoint python IMAGE -c \
  'import triton; print(triton.__version__, triton.__file__)'
```

The wheel was built with Python 3.12 in the pinned official base image. Its
NVIDIA toolkit files came from that image's Triton 3.7.1 wheel. LLVM came from
the URL selected by FlagTree's `cmake/llvm-hash.txt`:
`llvm-1f126a6d-ubuntu-x64-1.tar.gz` (SHA256
`0f9d80678e969a7966da672e2648cda7bb475c246d3a36ff6a4986d09916b0eb`).
The build used `TRITON_BUILD_PROTON=OFF`, `MAX_JOBS=16`, and disabled CMake unit
tests. Both NVIDIA and AMD codegen stayed enabled because FlagTree's Python
extension refers to generated AMD dialect headers even when only NVIDIA is
used at runtime. FlagTree's optional TLE/FlagCX device compiler hook stayed
disabled (`USE_FLAGCX=OFF`); the serving image's existing FlagCX PD plugin is
independent of that hook. No network access was allowed during the wheel build.

The final Docker build checks the wheel SHA, active module path and API version,
runs the existing CPU regression suite, then compiles 10 DSV4.1 kernels for
SM80/SM90 using FlagTree. Those checks do not establish GPU numerical accuracy,
CUDA Graph replay or throughput for the new compiler; run those separately
before replacing a live model service.
