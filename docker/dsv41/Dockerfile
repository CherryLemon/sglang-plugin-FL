# The build driver permits a local tag only after checking its immutable image ID.
ARG BASE_IMAGE=lmsysorg/sglang@sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf
FROM ${BASE_IMAGE} AS builder
COPY wheelhouse /build/wheelhouse
RUN python -m venv --system-site-packages /build/venv && \
    /build/venv/bin/pip install --no-index --no-deps --find-links=/build/wheelhouse \
    setuptools==76.1.0 setuptools-scm==9.2.2 setuptools-rust==1.12.0 \
    semantic-version==2.10.0 wheel==0.46.2
COPY gems /build/gems
COPY plugin /build/plugin
RUN SETUPTOOLS_SCM_PRETEND_VERSION=5.3.6.dev0+dsv41 \
    /build/venv/bin/pip wheel --no-deps --no-build-isolation --no-index \
    --wheel-dir=/build/wheels /build/gems /build/plugin
COPY delivery /build/delivery
RUN python /build/delivery/apply_patches.py /sgl-workspace/sglang
WORKDIR /sgl-workspace/sglang/python
RUN CARGO_NET_OFFLINE=true SGLANG_BUILD_RUST_EXTS=multimodal \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.5.18 \
    /build/venv/bin/python setup.py build_ext --inplace
RUN python /build/delivery/offline_compile.py /build/gems/src/flag_gems \
    > /build/offline-compile.json

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="SGLang FL DeepSeek V4.1 code migration" \
      org.opencontainers.image.base.name="lmsysorg/sglang:v0.5.18" \
      org.opencontainers.image.base.digest="sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf"
COPY --from=builder /build/wheels /opt/sglang-fl/wheels
COPY wheelhouse/sqlalchemy-*.whl wheelhouse/greenlet-*.whl /opt/sglang-fl/wheels/
RUN pip install --no-index --no-deps /opt/sglang-fl/wheels/*.whl
COPY delivery /opt/sglang-fl
COPY flagcx /opt/FlagCX
RUN python /opt/sglang-fl/apply_patches.py /sgl-workspace/sglang
COPY --from=builder /sgl-workspace/sglang/python/sglang/srt/rust_extensions/_multimodal*.so /sgl-workspace/sglang/python/sglang/srt/rust_extensions/
COPY --from=builder /build/offline-compile.json /opt/sglang-fl/offline-compile.json
COPY plugin/tests/dsv41 /opt/sglang-fl/tests/dsv41
ENV SGLANG_PLUGINS=sglang_fl \
    SGLANG_FL_DSV41_BACKEND=hybrid \
    SGLANG_FL_DSV41_REQUIRED=1 \
    SGLANG_FL_DSV41_QUANTIZATION=quantized \
    USE_FLAGGEMS=0 \
    SGLANG_FL_OOT_ENABLED=0 \
    SGLANG_FL_STRICT=1
RUN SGLANG_PLUGINS= SGLANG_FL_DSV41_REQUIRED=0 \
    python /opt/sglang-fl/validate_image.py && \
    SGLANG_PLUGINS= SGLANG_FL_DSV41_REQUIRED=0 \
    bash /opt/sglang-fl/validate_cpu.sh
WORKDIR /sgl-workspace/sglang
