"""Compile representative Triton kernels to cubins without a GPU or driver.

This is a compiler/API check, not a numerical, graph replay or performance test.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

root = Path(sys.argv[1])
results = []


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_case(fn, types, constants, *, arch=90, warps=4):
    signature = {
        name: "constexpr" if name in constants else types.get(name, "i32")
        for name in fn.arg_names
    }
    source = ASTSource(fn, signature=signature, constexprs=constants)
    kernel = triton.compile(
        source,
        target=GPUTarget("cuda", arch, 32),
        options={"num_warps": warps, "num_stages": 2},
    )
    results.append(
        {
            "kernel": fn.__name__,
            "arch": arch,
            "constants": constants,
            "ptx_sha256": hashlib.sha256(kernel.asm["ptx"].encode()).hexdigest(),
        }
    )


linear = load("dsv41_compile_linear", "fused/dsv41/block_fp8_linear.py")
for native, arch in [(True, 90), (False, 80)]:
    compile_case(
        linear._quantize,
        {"X": "*bf16", "Q": "*fp8e4nv" if native else "*bf16", "S": "*fp32"},
        {"K": 128, "GROUP": 32, "POW2": True, "EMULATE": not native},
        arch=arch,
        warps=1,
    )
for quantized in (True, False):
    compile_case(
        linear._bf16_matmul,
        {"A": "*bf16", "W": "*u8", "AS": "*fp32", "WS": "*fp32", "O": "*bf16"},
        {
            "K": 128,
            "GROUP_N": 32,
            "GROUP_K": 32,
            "QUANTIZED": quantized,
            "BM": 16,
            "BN": 32,
        },
        arch=80,
    )

# The generic GEMM's only package-level dependency is the device identity.
sys.modules["flag_gems"] = SimpleNamespace(device="cuda")
gemm = load("dsv41_compile_gemm", "ops/w8a8_block_fp8_matmul.py")
compile_case(
    gemm.w8a8_block_fp8_matmul_kernel,
    {"A": "*fp8e4nv", "B": "*fp8e4nv", "C": "*bf16", "As": "*fp32", "Bs": "*fp32"},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
)
indexer = load("dsv41_compile_indexer", "fused/dsv41/fp4_indexer.py")
pointer_types = {
    "q_ptr": "*bf16",
    "w_ptr": "*bf16",
    "slots_ptr": "*i64",
    "req_to_token_ptr": "*i32",
    "req_ptr": "*i64",
    "candidate_blocks_ptr": "*i32",
    "lens_ptr": "*i64",
    "table_ptr": "*u8",
    "out_ptr": "*fp32",
    "candidate_scores_ptr": "*fp32",
    "candidate_lens_ptr": "*i32",
}
for compact in (False, True):
    compile_case(
        indexer._fp4_index_logits_kernel,
        pointer_types,
        {
            "H": 32,
            "HALF_D": 64,
            "BLOCK_L": 64,
            "SKIP_INVALID": True,
            "RATIO": 2,
            "USE_REQ_TO_TOKEN": not compact,
            "USE_CANDIDATE_BLOCKS": compact,
            "CANDIDATE_BLOCK_SIZE": 8,
            "WRITE_CANDIDATES": not compact,
            "WRITE_LOGITS": True,
        },
    )
compile_case(
    indexer._fp4_index_logits_grouped_kernel,
    {
        "Q": "*bf16",
        "W": "*bf16",
        "RT": "*i32",
        "REQ": "*i64",
        "LENS": "*i64",
        "TABLE": "*u8",
        "OUT": "*fp32",
        "CS": "*fp32",
        "CL": "*i32",
    },
    {
        "GROUP": 6,
        "PGROUP": 8,
        "RATIO": 2,
        "BL": 64,
        "CB": 8,
        "WRITE_LOGITS": False,
        "WRITE_CANDIDATES": True,
    },
)
mhc = load("dsv41_compile_mhc", "fused/dsv41/mhc.py")
compile_case(
    mhc._hc_mix_stats_partial_kernel,
    {
        "x_ptr": "*bf16",
        "w_ptr": "*fp32",
        "part_mix_ptr": "*fp32",
        "part_sq_ptr": "*fp32",
    },
    {
        "MIX": 24,
        "MIX_PAD": 32,
        "NUM_SLICES": 80,
        "BLOCK_M": 8,
        "BLOCK_K": 64,
        "DOT_PRECISION": "tf32x3",
    },
)
compile_case(
    mhc._hc_combine_kernel,
    {"x_ptr": "*bf16", "pre_ptr": "*fp32", "y_ptr": "*bf16"},
    {"HC": 4, "BLOCK_H": 1024},
)
print(
    json.dumps(
        {"triton": triton.__version__, "gpu_executed": False, "cases": results},
        indent=2,
    )
)
