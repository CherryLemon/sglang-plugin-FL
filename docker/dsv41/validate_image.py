"""CPU-only image checks. Does not import FlagGems' device runtime or load weights."""

import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path

root = Path(__file__).resolve().parent
lock = json.loads((root / "manifest.json").read_text())
versions = {}
for package, expected in lock["packages"].items():
    actual = metadata.version(package)
    if actual != expected:
        raise RuntimeError(f"Dependency changed: {package} {actual} != {expected}")
    versions[package] = actual
for group in ("sglang.srt.plugins", "sglang.srt.platforms"):
    entry = [x for x in metadata.entry_points(group=group) if x.name == "sglang_fl"]
    assert len(entry) == 1 and callable(entry[0].load()), group
for module in (
    "sglang.srt.configs.deepseek_v41",
    "sglang.srt.server_args",
    "sglang.srt.models.deepseek_v4",
    "sglang.srt.models.deepseek_v4_dspark",
    "sglang.srt.layers.attention.deepseek_v4_backend",
    "sglang.srt.managers.scheduler",
    "sglang.srt.speculative.dspark_components.dspark_worker_v2",
    "sglang_fl.dsv41.backend",
):
    importlib.import_module(module)
from sglang.srt.layers.dsv41_ops import ABI_VERSION
from sglang.srt.rust_extensions import _multimodal

assert ABI_VERSION == 1 and callable(_multimodal.dsv41.resize_patchify)
for component in ("flaggems", "plugin"):
    distribution = metadata.distribution(
        "flag_gems" if component == "flaggems" else "sglang_fl"
    )
    for name, expected in lock[component]["package_files"].items():
        actual = hashlib.sha256(
            Path(distribution.locate_file(name)).read_bytes()
        ).hexdigest()
        assert actual == expected, f"{component} file changed: {name}"
print(
    json.dumps(
        {
            "versions": versions,
            "oot_abi": ABI_VERSION,
            "rust_dsv41": True,
            "gpu_tests_run": False,
        },
        indent=2,
    )
)
