# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify the exact selector outputs, then measure peak temporary memory."""

from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import hashlib
import json
import time


def load_functions(path):
    import torch
    import torch.nn.functional as F

    tree = ast.parse(Path(path).read_text())
    names = {
        "_candidate_block_amax",
        "select_candidate_blocks",
        "select_candidate_block_ids",
    }
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    scope = {
        "torch": torch,
        "F": F,
        "envs": SimpleNamespace(
            SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK=SimpleNamespace(
                get=lambda: False
            )
        ),
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--large", action="store_true")
    a = p.parse_args()
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    old = load_functions(a.reference)
    new = load_functions(a.candidate)
    torch.manual_seed(739101)
    cases = 0
    sizes = [
        1,
        3,
        4,
        7,
        8,
        9,
        15,
        16,
        17,
        127,
        128,
        129,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
    ]
    for dtype in [torch.float32, torch.bfloat16]:
        for rows in [1, 3, 8]:
            for width in sizes:
                for block in [4, 8, 16]:
                    for pattern in ["random", "ties", "sentinels", "strided"]:
                        x = torch.randn(
                            rows,
                            width * (2 if pattern == "strided" else 1),
                            dtype=dtype,
                            device=a.device,
                        )
                        if pattern == "strided":
                            x = x[:, ::2]
                        if pattern == "ties":
                            x.zero_()
                            x[:, ::3] = -0.0
                        if pattern == "sentinels":
                            x[:, ::3] = -torch.inf
                            x[:, ::7] = torch.inf
                            x[:, ::11] = torch.nan
                        lens = torch.tensor(
                            [(i * 73) % (width + 1) for i in range(rows)],
                            device=a.device,
                            dtype=torch.int32,
                        )[:, None]
                        for count in [1, 3, 16]:
                            for name in [
                                "select_candidate_blocks",
                                "select_candidate_block_ids",
                            ]:
                                ref = old[name](x, lens, count, block)
                                actual = new[name](x, lens, count, block)
                                if not torch.equal(actual, ref):
                                    raise AssertionError(
                                        (
                                            dtype,
                                            rows,
                                            width,
                                            block,
                                            pattern,
                                            count,
                                            name,
                                        )
                                    )
                            cases += 1
    stress = []
    if a.large:
        assert a.device.startswith("cuda")
        for rows, width in [
            (2048, 71680),
            (2048, 120832),
            (512, 488448),
            (373, 490684),
            (2048, 131071),
        ]:
            torch.cuda.empty_cache()
            x = torch.randn(rows, width, device=a.device, dtype=torch.float32)
            lens = torch.full((rows, 1), width, dtype=torch.int32, device=a.device)
            results = []
            for label, scope in [("reference", old), ("candidate", new)]:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                start = time.perf_counter()
                result = scope["select_candidate_block_ids"](x, lens, 2048, 8)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                results.append(
                    (
                        result.clone(),
                        {
                            "label": label,
                            "elapsed_s": elapsed,
                            "extra_peak_bytes": torch.cuda.max_memory_allocated()
                            - base,
                        },
                    )
                )
                del result
            assert torch.equal(results[0][0], results[1][0])
            for _ in range(10):
                assert torch.equal(
                    new["select_candidate_block_ids"](x, lens, 2048, 8), results[1][0]
                )
            stress.append(
                {"rows": rows, "width": width, "results": [x[1] for x in results]}
            )
            del x, results
    result = {
        "status": "PASS",
        "device": a.device,
        "selector_cases": cases,
        "bitwise_equal": True,
        "large": stress,
        "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
        "torch": torch.__version__,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
