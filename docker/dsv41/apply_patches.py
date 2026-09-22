"""Apply the reviewed patch series to the exact engine in the official image."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

delivery = Path(__file__).resolve().parent
lock = json.loads((delivery / "manifest.json").read_text())
repo = Path(sys.argv[1])
head = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
if head != lock["sglang"]["base"]:
    raise SystemExit(f"Wrong engine base: {head}")
for entry in lock["engine_patches"]:
    patch = delivery / entry["path"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"Patch hash mismatch: {patch.name}")
    subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)
for name, expected in lock["sglang"]["files"].items():
    actual = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Patched source hash mismatch: {name}")
print(
    f"Applied {len(lock['engine_patches'])} patches; verified {len(lock['sglang']['files'])} files"
)
