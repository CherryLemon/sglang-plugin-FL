"""Thin adapter around the previously validated FlagCX NVIDIA P2P API."""

import hashlib
import logging
import os
from pathlib import Path
import sys
import threading
import time

import torch
from prometheus_client import Counter

logger = logging.getLogger(__name__)
_BYTES = Counter(
    "sglang:flagcx_total_bytes_transferred",
    "Successful FlagCX PD write bytes",
    ["model_name", "role", "tp_rank"],
)
_CALLS = Counter(
    "sglang:flagcx_transfers",
    "Successful FlagCX PD batch writes",
    ["model_name", "role", "tp_rank"],
)
_FAILED = Counter(
    "sglang:flagcx_num_failed_transfers",
    "Failed FlagCX PD batch writes",
    ["model_name", "role", "tp_rank"],
)
LIB_SHA = "3bc2c0be3053e9b3413d352b884b749876f064d590c1505c57ee986e4e23e1de"
WRAPPER_SHA = "e6e4bdb2c0d0034100d861e132d5092a381cdab5764db7b2a3775b56efc1ff27"


class FlagCXEngine:
    def __init__(self, hostname, gpu_id=0, ib_device="mlx5_bond_0", labels=None):
        root = Path(os.environ["FLAGCX_PATH"])
        lib = root / "build/lib/libflagcx.so"
        wrapper = root / "plugin/interservice/flagcx_wrapper.py"
        assert hashlib.sha256(lib.read_bytes()).hexdigest() == LIB_SHA
        assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == WRAPPER_SHA
        sys.path.insert(0, str(root))
        from plugin.interservice.flagcx_wrapper import FLAGCXLibrary

        torch.cuda.set_device(gpu_id)
        self.flagcx = FLAGCXLibrary(str(lib))
        self.engine = self.flagcx.flagcxP2pEngineCreate()
        self.rpc_port = self.flagcx.flagcxP2pGetRpcPort(self.engine)
        self.flagcx.flagcxP2pStartRpcServer(self.engine)
        self.session_id = f"{hostname}:{self.rpc_port}"
        self.ib_device = ib_device
        self._connections = {}
        self._lock = threading.Lock()
        self._closed = False
        labels = labels or dict(
            model_name="p2p-validation", role="test", tp_rank=str(gpu_id)
        )
        self.bytes = _BYTES.labels(**labels)
        self.calls = _CALLS.labels(**labels)
        self.failed = _FAILED.labels(**labels)

    def batch_register(self, ptrs, lengths):
        assert len(ptrs) == len(lengths)
        for ptr, length in zip(ptrs, lengths):
            self.flagcx.flagcxP2pRegister(self.engine, ptr, length)
        return 0

    def batch_register_host(self, ptrs, lengths):
        assert len(ptrs) == len(lengths)
        for ptr, length in zip(ptrs, lengths):
            self.flagcx.flagcxP2pRegisterHost(self.engine, ptr, length)
        return 0

    def batch_deregister(self, ptrs):
        # The pinned engine owns registrations until its process/engine exits.
        return 0

    def _get_conn(self, session_id):
        conn = self._connections.get(session_id)
        if conn is None:
            with self._lock:
                conn = self._connections.get(session_id)
                if conn is None:
                    conn = self.flagcx.flagcxP2pGetConn(self.engine, session_id)
                    self._connections[session_id] = conn
        return conn

    def batch_transfer_sync(self, session_id, buffers, peer_addresses, lengths):
        try:
            if not (len(buffers) == len(peer_addresses) == len(lengths)):
                raise ValueError("FlagCX descriptor list length mismatch")
            if not lengths:
                return 0
            self.flagcx.flagcxP2pBatchWriteSync(
                self._get_conn(session_id), buffers, peer_addresses, lengths
            )
            self.bytes.inc(sum(lengths))
            self.calls.inc()
            return 0
        except Exception:
            self.failed.inc()
            logger.exception("FlagCX PD batch write failed; no backend fallback")
            return -1

    def transfer_sync(self, session_id, buffer, peer_address, length):
        return self.batch_transfer_sync(session_id, [buffer], [peer_address], [length])

    def get_session_id(self):
        return self.session_id

    def get_ib_device(self):
        return self.ib_device

    def close(self):
        if not self._closed:
            self.flagcx.flagcxP2pEngineDestroy(self.engine)
            self._closed = True
