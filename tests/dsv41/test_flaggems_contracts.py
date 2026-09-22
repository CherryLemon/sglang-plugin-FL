"""Exercise FlagGems host contracts without initializing a GPU runtime."""

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def _source():
    if os.environ.get("DSV41_FLAGGEMS_SOURCE"):
        return Path(os.environ["DSV41_FLAGGEMS_SOURCE"]) / "src/flag_gems"
    return Path(importlib.metadata.distribution("flag_gems").locate_file("flag_gems"))


@pytest.fixture
def gemm(monkeypatch):
    # Only the device identity is needed by this module's host config selector.
    # The production package deliberately cannot initialize on a CPU host.
    monkeypatch.setitem(sys.modules, "flag_gems", SimpleNamespace(device="kunlunxin"))
    spec = importlib.util.spec_from_file_location(
        "dsv41_gemm_contract", _source() / "ops/w8a8_block_fp8_matmul.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_cuda_and_cached_tiles_do_not_cross_group32(gemm, monkeypatch):
    assert gemm._get_default_w8a8_block_fp8_config(32, 32)["BLOCK_SIZE_K"] == 32
    config = dict(
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=4,
        num_warps=4,
        num_stages=3,
    )
    monkeypatch.setattr(gemm, "get_w8a8_block_fp8_configs", lambda *a: {8: config})
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(kwargs)

            return launch

    monkeypatch.setattr(gemm, "w8a8_block_fp8_matmul_kernel", Kernel())
    gemm.w8a8_block_fp8_matmul(
        torch.empty(3, 128),
        torch.empty(35, 128),
        torch.ones(3, 4),
        torch.ones(2, 4),
        [32, 32],
    )
    assert calls[0]["BLOCK_SIZE_K"] == 32
    assert config["BLOCK_SIZE_K"] == 128


def test_empty_m_n_k_and_encoded_scales(gemm):
    for m, n, k in [(0, 35, 128), (3, 0, 128), (3, 35, 0)]:
        y = gemm.w8a8_block_fp8_matmul(
            torch.empty(m, k),
            torch.empty(n, k),
            torch.empty(m, (k + 31) // 32),
            torch.empty((n + 31) // 32, (k + 31) // 32),
            [32, 32],
        )
        assert y.shape == (m, n)
        if k == 0:
            assert torch.equal(y, torch.zeros_like(y))
    with pytest.raises(TypeError, match="decode UE8M0"):
        gemm.w8a8_block_fp8_matmul(
            torch.empty(1, 32),
            torch.empty(32, 32),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.uint8),
            [32, 32],
        )


def test_fp4_empty_candidate_contracts():
    spec = importlib.util.spec_from_file_location(
        "dsv41_fp4_contract", _source() / "fused/dsv41/fp4_indexer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = torch.empty(2, 32, 128, dtype=torch.bfloat16)
    scores, lengths = module.fp4_index_logits_req_to_token(
        q,
        torch.ones(2, 32),
        torch.zeros(2, 64, dtype=torch.int32),
        torch.tensor([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        torch.empty(1, 68 * 64, dtype=torch.uint8),
        64,
        1,
        0,
        candidate_block_size=8,
        write_logits=False,
    )
    assert scores.shape == (2, 0)
    assert torch.equal(lengths, torch.zeros(2, dtype=torch.int32))
