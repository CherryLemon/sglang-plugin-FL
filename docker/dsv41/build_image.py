"""Prepare an offline build context and build the pinned NVIDIA migration image.

The patched FlagGems checkout is explicit. All downloads happen before docker
build, and no credentials or proxy configuration enter the context or image.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def snapshot(repo, target):
    # git archive excludes local caches, credentials and untracked files.
    with tarfile.open(fileobj=io.BytesIO(git(repo, "archive", "HEAD"))) as archive:
        archive.extractall(target, filter="data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flaggems", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="sglang-fl-dsv41:0.5.18")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    delivery = Path(__file__).resolve().parent
    plugin = delivery.parent.parent
    lock = json.loads((delivery / "manifest.json").read_text())
    image_id = subprocess.check_output(
        [
            "docker",
            "image",
            "inspect",
            lock["base_image"]["tag"],
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    if image_id != lock["base_image"]["image_id"]:
        raise SystemExit("Local base tag does not match the locked official image ID")
    for repo, component in [(args.flaggems, "flaggems"), (plugin, "plugin")]:
        package_path = "src/flag_gems" if component == "flaggems" else "sglang_fl"
        tree = git(repo, "rev-parse", f"HEAD:{package_path}").decode().strip()
        if tree != lock[component]["package_tree"]:
            raise SystemExit(f"Package tree drift: {component}")
        for name, expected in lock[component]["package_files"].items():
            path = repo / ("src" if component == "flaggems" else "") / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise SystemExit(
                    f"Source drift: {component}/{name}; regenerate and review the lock"
                )
        if git(
            repo,
            "diff",
            "HEAD",
            "--",
            "src" if component == "flaggems" else "sglang_fl",
        ).strip():
            raise SystemExit(
                f"Commit {component} implementation changes before building"
            )
    # Refuse to overwrite another build's artifacts.
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot(args.flaggems, args.output / "gems")
    snapshot(plugin, args.output / "plugin")
    # Build receipts describe the resulting image and stay outside it. The
    # compiler report is regenerated in the builder stage for this source tree.
    shutil.copytree(
        delivery,
        args.output / "delivery",
        ignore=shutil.ignore_patterns(
            "__pycache__", "build-result.json", "offline-compile.json"
        ),
    )
    wheels = args.output / "wheelhouse"
    wheels.mkdir()
    for name, expected in lock["build_wheels"].items():
        path = args.wheelhouse / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Wheel hash mismatch: {name}")
        shutil.copyfile(path, wheels / name)
    shutil.copyfile(delivery / "Dockerfile", args.output / "Dockerfile")
    if args.prepare_only:
        return
    env = dict(os.environ, DOCKER_BUILDKIT="0")
    subprocess.run(
        [
            "docker",
            "build",
            "--network=none",
            "--pull=false",
            "--build-arg",
            f"BASE_IMAGE={lock['base_image']['tag']}",
            "-t",
            args.tag,
            str(args.output),
        ],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
