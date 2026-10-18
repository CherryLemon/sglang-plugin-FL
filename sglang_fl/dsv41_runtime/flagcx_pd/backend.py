"""Reuse the current native descriptor, state, ACK and error protocol."""

import logging
import os

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from .engine import FlagCXEngine
from .rolling import RollingPDEngine
from prometheus_client import Counter

_LEGACY_CALLS = Counter(
    "sglang:pd_legacy_mooncake_transfers",
    "Explicit migration transfers to legacy Mooncake receivers",
    ["model_name", "role", "tp_rank", "result"],
)
_LEGACY_BYTES = Counter(
    "sglang:pd_legacy_mooncake_bytes",
    "Explicit successful migration bytes to legacy Mooncake receivers",
    ["model_name", "role", "tp_rank"],
)

logger = logging.getLogger(__name__)


class FlagcxKVManager(MooncakeKVManager):
    def init_engine(self):
        flagcx = FlagCXEngine(
            os.environ["SGLANG_HOST_IP"],
            self.kv_args.gpu_id,
            self.server_args.disaggregation_ib_device,
            labels=dict(
                model_name=self.server_args.served_model_name,
                role=self.disaggregation_mode.value,
                tp_rank=str(self.kv_args.engine_rank),
            ),
        )
        labels = dict(
            model_name=self.server_args.served_model_name,
            role=self.disaggregation_mode.value,
            tp_rank=str(self.kv_args.engine_rank),
        )
        allow_legacy = os.getenv("SGLANG_DSV41_FLAGCX_ROLLING_ALLOW_LEGACY") == "1"
        legacy = None
        on_legacy = None
        if allow_legacy:
            if self.disaggregation_mode.value != "prefill":
                raise RuntimeError("Legacy receiver compatibility is sender-only")
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                MooncakeTransferEngine,
                get_mooncake_transfer_engine,
            )

            legacy = get_mooncake_transfer_engine()
            if legacy is None:
                legacy = MooncakeTransferEngine(
                    os.environ["SGLANG_HOST_IP"],
                    self.kv_args.gpu_id,
                    self.server_args.disaggregation_ib_device,
                )

            def on_legacy(code, size):
                _LEGACY_CALLS.labels(
                    **labels, result="ok" if code == 0 else "failed"
                ).inc()
                if code == 0:
                    _LEGACY_BYTES.labels(**labels).inc(size)

        self.engine = RollingPDEngine(flagcx, legacy, on_legacy)
        logger.info(
            "FLAGCX_PD_PROTOCOL_V2: explicit_receiver_tag=flagcx:// legacy_sender_admitted=%s no_failure_fallback=True",
            allow_legacy,
        )
        logger.info(
            "FLAGCX_PD_ACTIVE: gpu=%s session=%s hca=%s engine=FlagCX",
            self.kv_args.gpu_id,
            self.engine.get_session_id(),
            self.server_args.disaggregation_ib_device,
        )

    def register_buffer_to_engine(self):
        # NVIDIA auxiliary buffers are host allocations. Probing them as CUDA
        # pointers would poison the CUDA error state; use the host API directly.
        host = set(self.kv_args.aux_data_ptrs or [])
        regions = self._registerable_regions()
        for is_host in [False, True]:
            selected = [(p, n) for p, n in regions if (p in host) == is_host]
            if selected:
                ptrs, lengths = map(list, zip(*selected))
                fn = (
                    self.engine.batch_register_host
                    if is_host
                    else self.engine.batch_register
                )
                fn(ptrs, lengths)
