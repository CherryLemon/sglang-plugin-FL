"""Explicit DSV4.1 dispatch; never retry a failed numerical kernel.

Hybrid is a migration profile, with a fixed vendor allowlist. FlagOS-only and
reference profiles fail model admission until the whole serving chain is covered.
Reference calls are limited to small CPU fixtures.
"""

import importlib
import logging
import os
from dataclasses import asdict, dataclass

import torch

from .quantization import QuantizationMode, block_fp8_linear

logger = logging.getLogger(__name__)
ABI_VERSION = 1

INDEX_OPS = (
    "fp4_index_logits_decode",
    "fp4_index_logits_req_to_token",
    "fp4_index_logits_candidate_blocks",
    "unpack_fp4_index_keys_to_fp8",
    "quantize_bf16_index_queries_fp8",
    "fp8_index_logits_prefill",
)
GEMS_OPS = frozenset(
    {
        "linear.block_fp8",
        "mhc.mix_and_combine",
        "mhc.post",
        "mhc.hc_mix_stats",
        "mhc.hc_mix_stats_sinkhorn",
        "mhc.hc_combine",
        *(f"indexer.{name}" for name in INDEX_OPS),
    }
)
# These composite entry points retain the source V12 implementation. Inner
# block-FP8/indexer calls still cross their own OOT boundary and are logged.
VENDOR_ONLY = frozenset(
    {
        "attention.sparse",
        "cache.low_ratio_sources",
        "moe.tp_ep",
        "rope.tail",
    }
)
ALL_OPS = GEMS_OPS | VENDOR_ONLY


@dataclass(frozen=True)
class Capabilities:
    device: str
    native_fp8_matrix: bool = False
    native_mxfp4_matrix: bool = False
    graph: bool = False
    vendor_serving: bool = False


def detect_capabilities() -> Capabilities:
    # A float8 dtype is not evidence of matrix instruction or compiler support.
    # Other vendors need their own versioned capability probe before admission.
    if torch.version.cuda is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return Capabilities(
            "cuda",
            native_fp8_matrix=(major, minor) >= (8, 9),
            native_mxfp4_matrix=major >= 10,
            graph=True,
            vendor_serving=major in (9, 10),
        )
    return Capabilities("cpu" if not torch.cuda.is_available() else "other")


def _mix_and_combine(
    layer, x, hc_fn, hc_scale, hc_base, apply_pre, norm, stats_stream=None
):
    from flag_gems.fused.dsv41.mhc import hc_combine, hc_mix_stats_sinkhorn

    x_flat = x.flatten(1)
    # apply_pre belongs to the PREVIOUS sublayer. Replacing it with the newly
    # computed pre changes the model; keep the two values separate.
    if stats_stream is None:
        pre, post, comb = hc_mix_stats_sinkhorn(
            x_flat,
            hc_fn,
            hc_scale,
            hc_base,
            layer.hc_mult,
            layer.hc_sinkhorn_iters,
            layer.rms_norm_eps,
            layer.hc_eps,
        )
    else:
        main_stream = torch.cuda.current_stream(x.device)
        stats_stream.wait_stream(main_stream)
        x.record_stream(stats_stream)
        with torch.cuda.stream(stats_stream):
            pre, post, comb = hc_mix_stats_sinkhorn(
                x_flat,
                hc_fn,
                hc_scale,
                hc_base,
                layer.hc_mult,
                layer.hc_sinkhorn_iters,
                layer.rms_norm_eps,
                layer.hc_eps,
            )
        # The caller joins the side stream before consuming the coefficients,
        # just as in the engine's native implementation.
        for coefficient in (pre, post, comb):
            coefficient.record_stream(main_stream)
    combined = (
        x[:, 0, :].contiguous()
        if apply_pre is None
        else hc_combine(x_flat, apply_pre, layer.hc_mult, x.dtype)
    )
    return norm(combined), pre, post, comb


def _mhc_post(layer, x, residual, post, comb):
    if x.shape[0] == 0:
        return torch.empty_like(residual)
    from flag_gems.fused.mhc.mhc_post import mhc_post

    return mhc_post(x, residual, post, comb)


class Dsv41Backend:
    def __init__(self, mode="hybrid", quantization="quantized", *, capabilities=None):
        if mode not in ("hybrid", "vendor", "flagos", "reference"):
            raise ValueError(f"Unknown DSV4.1 backend: {mode}")
        self.mode = mode
        self.quantization = QuantizationMode(quantization)
        self.capabilities = capabilities or detect_capabilities()
        self._selected = {}
        self._implementations = {}

    def snapshot(self):
        return {
            "abi": ABI_VERSION,
            "mode": self.mode,
            "quantization": self.quantization.value,
            "capabilities": asdict(self.capabilities),
            "selected": dict(self._selected),
            "vendor_only": sorted(VENDOR_ONLY),
        }

    def validate_model(self, config):
        if getattr(config, "model_type", None) != "deepseek_v41":
            return
        if self.mode in ("flagos", "reference"):
            raise RuntimeError(
                f"DSV4.1 {self.mode} is an operator-fixture profile; model serving "
                f"still requires vendor implementations for {sorted(VENDOR_ONLY)}"
            )
        if self.quantization is not QuantizationMode.QUANTIZED:
            raise RuntimeError(
                "BF16 is an explicit linear-operator mode; full-model BF16 is not yet covered"
            )
        if not self.capabilities.vendor_serving:
            raise RuntimeError(
                "The remaining DSV4.1 vendor chain requires CUDA SM90/SM100; other devices need sparse-attention/cache/EP ports"
            )
        logger.info("DSV4.1 backend admission: %s", self.snapshot())

    def _record(self, name, backend):
        if name not in self._selected:
            self._selected[name] = backend
            logger.info(
                "DSV4.1 op=%s backend=%s quantization=%s",
                name,
                backend,
                self.quantization.value,
            )

    def _gems(self, name):
        if (
            name == "indexer.fp8_index_logits_prefill"
            and not self.capabilities.native_fp8_matrix
        ):
            raise NotImplementedError(
                "FP8 prefill scoring requires a verified native FP8 backend; the packed FP4 decode path uses BF16"
            )
        if name not in self._implementations:
            if name == "linear.block_fp8":
                module = importlib.import_module(
                    "flag_gems.fused.dsv41.block_fp8_linear"
                )

                def linear(*args, **kwargs):
                    return module.block_fp8_linear(
                        *args,
                        **kwargs,
                        mode=self.quantization.value,
                        native_fp8=self.capabilities.native_fp8_matrix,
                    )

                fn = linear
            elif name == "mhc.mix_and_combine":
                fn = _mix_and_combine
            elif name == "mhc.post":
                fn = _mhc_post
            else:
                family, symbol = name.split(".", 1)
                module = "fp4_indexer" if family == "indexer" else "mhc"
                fn = getattr(
                    importlib.import_module(f"flag_gems.fused.dsv41.{module}"), symbol
                )
            self._implementations[name] = fn
        return self._implementations[name]

    def _reference(self, name, *args, **kwargs):
        if name != "linear.block_fp8":
            raise NotImplementedError(f"No small reference registered for {name}")

        # Bind explicitly to the engine ABI, including optional positional args.
        def linear(
            input,
            weight,
            block_size,
            weight_scale,
            input_scale=None,
            bias=None,
            act_scale_ue8m0=False,
        ):
            if input.device.type != "cpu" or weight.device.type != "cpu":
                raise ValueError("Reference profile accepts CPU fixtures only")
            if input.numel() + weight.numel() > 1_048_576:
                raise ValueError("Reference fixture exceeds the bounded work limit")
            if input_scale is not None:
                raise NotImplementedError("Reference input must be unquantized")
            return block_fp8_linear(
                input,
                weight,
                weight_scale,
                block_size,
                bias,
                mode=self.quantization,
                ue8m0=act_scale_ue8m0,
            )

        return linear(*args, **kwargs)

    def __call__(self, name, native, *args, **kwargs):
        if name not in ALL_OPS:
            raise KeyError(f"Unregistered DSV4.1 operation: {name}")
        if self.mode == "reference":
            self._record(name, "reference")
            return self._reference(name, *args, **kwargs)
        if self.mode == "vendor" or name in VENDOR_ONLY:
            if self.mode == "flagos":
                raise NotImplementedError(f"FlagOS implementation missing: {name}")
            if self.quantization is not QuantizationMode.QUANTIZED:
                raise RuntimeError("Vendor paths cannot silently consume BF16 mode")
            self._record(name, "vendor")
            return native(*args, **kwargs)
        self._record(name, "flaggems")
        return self._gems(name)(*args, **kwargs)


_backend = None


def install_backend():
    global _backend
    mode = os.environ.get("SGLANG_FL_DSV41_BACKEND", "off")
    if mode == "off":
        return None
    if _backend is not None:
        return _backend
    from sglang.srt.layers.dsv41_ops import register_backend

    backend = Dsv41Backend(
        mode, os.environ.get("SGLANG_FL_DSV41_QUANTIZATION", "quantized")
    )
    register_backend(
        backend, abi_version=ABI_VERSION, validate_model=backend.validate_model
    )
    _backend = backend
    logger.info("DSV4.1 backend registered: %s", backend.snapshot())
    return backend
