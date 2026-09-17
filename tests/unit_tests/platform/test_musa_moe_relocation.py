# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline provenance/linkage checks; no GPU performance claim is made here."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_fl.dispatch.backends.vendor.mthreads import moe

ROOT = Path(moe.__file__).parent
PROVENANCE = json.loads((ROOT / "provenance.json").read_text())


@pytest.mark.parametrize("filename", list(PROVENANCE["files"]))
def test_relocated_function_bodies_match_captured_source(filename):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    expected = PROVENANCE["files"][filename]["function_ast_sha256_excluding_decorators"]
    assert expected
    for name, digest in expected.items():
        node = nodes[name]
        node.decorator_list = []
        normalized = ast.dump(node, include_attributes=False).replace(
            ", type_params=[]", ""
        )
        actual = hashlib.sha256(normalized.encode()).hexdigest()
        assert actual == digest, f"{filename}:{name} diverged from measured source"


def test_custom_ops_have_distinct_names_and_preserve_mutation_contract():
    tree = ast.parse((ROOT / "fused_moe.py").read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in ("inplace_fused_experts", "outplace_fused_experts"):
        decorator = functions[name].decorator_list[0]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
        assert kwargs["op_name"] == "musa_qwen36_" + name
        if name.startswith("inplace"):
            assert kwargs["mutates_args"] == ["hidden_states"]
        else:
            assert kwargs["out_shape"] == "hidden_states"


@pytest.mark.parametrize(
    "name",
    [
        "moe_sum_reduce",
        "try_get_optimal_moe_config",
        "moe_align_block_size",
        "moe_sum_reduce_torch_compile",
    ],
)
def test_existing_patches_are_resolved_at_call_time(name):
    # Execute only the tiny host forwarding function under test, not the
    # Triton module and its hardware imports. Production does no AST rewriting.
    tree = ast.parse((ROOT / "fused_moe.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    core = SimpleNamespace()
    setattr(core, name, lambda *a, **k: ("before", a, k))
    namespace = {"_core": core}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "<forwarder-test>", "exec"
        ),
        namespace,
    )
    forwarded = namespace[name]
    assert forwarded(1, route=2) == ("before", (1,), {"route": 2})
    setattr(core, name, lambda *a, **k: ("after", a, k))
    assert forwarded(3, route=4) == ("after", (3,), {"route": 4})
