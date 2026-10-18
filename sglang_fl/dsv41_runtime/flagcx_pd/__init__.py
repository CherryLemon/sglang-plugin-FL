"""Source-pinned FlagCX P/D engine with the native KV/state control protocol."""

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_installed = False


def classes(class_type):
    if os.getenv("SGLANG_DSV41_FLAGCX_PD") != "1":
        raise RuntimeError("FlagCX P/D transport requires explicit opt-in")
    from sglang.srt.disaggregation.base import KVArgs
    from sglang.srt.disaggregation.utils import KVClassType

    if class_type == KVClassType.KVARGS:
        return KVArgs
    if class_type == KVClassType.MANAGER:
        from .backend import FlagcxKVManager

        return FlagcxKVManager
    from sglang.srt.disaggregation.mooncake.conn import (
        MooncakeKVBootstrapServer,
        MooncakeKVReceiver,
        MooncakeKVSender,
    )

    return {
        KVClassType.SENDER: MooncakeKVSender,
        KVClassType.RECEIVER: MooncakeKVReceiver,
        KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
    }[class_type]


def load_plugin():
    global _installed
    if os.getenv("SGLANG_DSV41_FLAGCX_PD") != "1" or _installed:
        return
    import sglang

    root = Path(sglang.__file__).parent / "srt"
    pins = json.loads((Path(__file__).parent / "source-pins.json").read_text())
    for name, digest in pins.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("unreviewed FlagCX KV protocol source: " + name)
    from sglang.srt import server_args
    from sglang.srt.disaggregation import utils

    if "flagcx" not in server_args.DISAGG_TRANSFER_BACKEND_CHOICES:
        server_args.add_disagg_transfer_backend_choices(["flagcx"])
    assert utils.TransferBackend("flagcx") is utils.TransferBackend.FLAGCX
    _installed = True
    logger.info(
        "FLAGCX_PD_REGISTERED: native KV/state protocol, FlagCX P2P engine; source pins verified"
    )
