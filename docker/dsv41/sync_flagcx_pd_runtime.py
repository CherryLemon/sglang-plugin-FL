"""Sync reviewed source into a prepared OCI runtime container for GPU bring-up."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--flaggems", type=Path, required=True)
    args = parser.parse_args()
    engine = ["nerdctl", "--namespace=sglang-fl-0518"]
    packages = (
        (args.sglang / "python/sglang", "/sgl-workspace/sglang/python/sglang"),
        (args.plugin / "sglang_fl", "/usr/local/lib/python3.12/dist-packages/sglang_fl"),
        (args.flaggems / "src/flag_gems", "/usr/local/lib/python3.12/dist-packages/flag_gems"),
    )
    for source, target in packages:
        assert source.is_dir(), source
        subprocess.run([*engine, "cp", f"{source}/.", f"{args.container}:{target}"], check=True)

    samples = {
        "/sgl-workspace/sglang/python/sglang/srt/arg_groups/overrides.py":
            args.sglang / "python/sglang/srt/arg_groups/overrides.py",
        "/sgl-workspace/sglang/python/sglang/srt/arg_groups/deepseek_v4_hook.py":
            args.sglang / "python/sglang/srt/arg_groups/deepseek_v4_hook.py",
        "/sgl-workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py":
            args.sglang / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
        "/sgl-workspace/sglang/python/sglang/srt/model_executor/pool_configurator.py":
            args.sglang / "python/sglang/srt/model_executor/pool_configurator.py",
        "/usr/local/lib/python3.12/dist-packages/sglang_fl/disaggregation/conn.py":
            args.plugin / "sglang_fl/disaggregation/conn.py",
        "/usr/local/lib/python3.12/dist-packages/sglang_fl/disaggregation/transfer_engine.py":
            args.plugin / "sglang_fl/disaggregation/transfer_engine.py",
        "/usr/local/lib/python3.12/dist-packages/flag_gems/fused/dsv41/mhc.py":
            args.flaggems / "src/flag_gems/fused/dsv41/mhc.py",
    }
    code = (
        "import hashlib,json,pathlib,sys; "
        "print(json.dumps({name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() "
        "for name in sys.argv[1:]}))"
    )
    actual = json.loads(subprocess.check_output(
        [*engine, "exec", args.container, "python", "-c", code, *samples], text=True
    ))
    expected = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in samples.items()}
    assert actual == expected, (actual, expected)
    print(json.dumps({"container": args.container, "source_sha256": actual}))


if __name__ == "__main__":
    main()
