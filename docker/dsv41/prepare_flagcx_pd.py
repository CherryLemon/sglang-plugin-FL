"""Prepare one isolated 8-GPU FlagCX PD container on a validated host."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--flagcx", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    namespace = "sglang-fl-0518"
    name = f"sglang-fl-dsv41-flagcx-{args.role}"
    engine = ["nerdctl", f"--namespace={namespace}"]
    image_id = subprocess.check_output(
        [*engine, "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    if subprocess.run([*engine, "container", "inspect", name], capture_output=True).returncode == 0:
        raise SystemExit(f"{name} already exists; inspect it before replacing")

    model = Path("/public-nvme/models/DeepSeek-V4.1-Flash")
    assert (model / "config.json").is_file()
    assert (args.flagcx / "build/lib/libflagcx.so").is_file()
    driver = Path(f"/data/sglang-fl-dsv41-flagcx-{args.role}/driver")
    cache = Path(f"/data/sglang-fl-dsv41-flagcx-{args.role}/cache")
    driver.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    for name_so in (
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        "libnvidia-nvvm.so.4",
        "libnvidia-ptxjitcompiler.so.1",
    ):
        dest = driver / name_so
        if not dest.is_file():
            source = Path("/usr/lib/x86_64-linux-gnu") / name_so
            assert source.is_file(), source
            shutil.copyfile(source, dest, follow_symlinks=True)

    cmd = [
        *engine, "run", "-d", "--name", name, "--network=host", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "--ulimit", "nofile=1048576",
        "--mount", f"type=bind,src={model},dst=/models/DeepSeek-V4.1-Flash,readonly",
        "--mount", f"type=bind,src={driver},dst=/driver,readonly",
        "--mount", f"type=bind,src={cache},dst=/work/cache",
        "--mount", f"type=bind,src={args.results},dst=/work/results",
        "--mount", f"type=bind,src={args.flagcx},dst=/opt/FlagCX,readonly",
        "--mount", "type=bind,src=/public-nvme/yjwu/devenv/sglang-plugin-FL/docker/dsv41,dst=/work/scripts,readonly",
        "-e", "LD_LIBRARY_PATH=/driver:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "-e", "MODEL_PATH=/models/DeepSeek-V4.1-Flash",
        "-e", "FLAGCX_PATH=/opt/FlagCX",
        "-e", "SGLANG_FL_DIST_BACKEND=nccl",
        "-e", "TRITON_CACHE_DIR=/work/cache/triton",
        "-e", "TORCH_EXTENSIONS_DIR=/work/cache/torch_extensions",
        "-e", "CUDA_CACHE_PATH=/work/cache/cuda",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1",
    ]
    for pattern in ("/dev/nvidia*", "/dev/nvidia-caps/*", "/dev/infiniband/*"):
        for device in sorted(Path("/").glob(pattern.lstrip("/"))):
            if device.is_char_device():
                cmd.extend(("--device", str(device)))
    cmd += ["--entrypoint", "sleep", args.image, "infinity"]
    container_id = subprocess.check_output(cmd, text=True).strip()
    print(json.dumps({"container": name, "id": container_id, "image_id": image_id}))


if __name__ == "__main__":
    main()
