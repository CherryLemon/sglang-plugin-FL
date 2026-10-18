# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Model hooks register once and leave the actual transport/configuration intact."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[3]


def resolving_view(args):
    return args


def validate_features(args):
    cfg = resolving_view(args)
    if cfg.disaggregation_transfer_backend != "mooncake":
        raise ValueError("unsupported protocol")
    if cfg.invalid_feature:
        raise ValueError("invalid feature")
    args.resolved = True


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelHookTests(unittest.TestCase):
    def setUp(self):
        self.module = load(
            "isolated_dsv41_hooks.patch", "sglang_fl/dsv41_runtime/patch.py"
        )

    def test_flagcx_admitted_without_changing_resolver_or_transport(self):
        args = types.SimpleNamespace(
            disaggregation_transfer_backend="flagcx", invalid_feature=False
        )
        self.module._validate_features(validate_features, args)
        self.assertTrue(args.resolved)
        self.assertEqual(args.disaggregation_transfer_backend, "flagcx")
        self.assertIs(validate_features.__globals__["resolving_view"], resolving_view)

    def test_other_feature_validation_remains_active(self):
        args = types.SimpleNamespace(
            disaggregation_transfer_backend="flagcx", invalid_feature=True
        )
        with self.assertRaisesRegex(ValueError, "invalid feature"):
            self.module._validate_features(validate_features, args)
        self.assertEqual(args.disaggregation_transfer_backend, "flagcx")

    def test_other_transports_use_original_validation(self):
        for backend in ("mooncake", "nixl"):
            args = types.SimpleNamespace(
                disaggregation_transfer_backend=backend, invalid_feature=False
            )
            calls = []

            def original(value):
                calls.append(value)
                return "original"

            self.assertEqual(self.module._validate_features(original, args), "original")
            self.assertEqual(calls, [args])

    def test_entry_point_installs_hooks_without_environment_configuration(self):
        entry = load("isolated_dsv41_entry", "sglang_fl/dsv41_runtime/__init__.py")
        hooks = types.ModuleType("isolated_dsv41_entry.patch")
        hooks.apply_dsv41_patches = unittest.mock.Mock()
        with patch.dict(sys.modules, {hooks.__name__: hooks}):
            entry.load_plugin()
        hooks.apply_dsv41_patches.assert_called_once_with()

    def test_registration_uses_repository_hook_registry_once(self):
        registry = types.SimpleNamespace(register=unittest.mock.Mock())
        kinds = types.SimpleNamespace(AROUND="around", REPLACE="replace")
        native = types.ModuleType("sglang_fl.disaggregation.patch")
        native.apply_disaggregation_patch = unittest.mock.Mock()
        prefill = types.ModuleType("isolated_dsv41_hooks.prefill")
        for name in ("candidate_block_ids", "candidate_blocks", "extend_indexer"):
            setattr(prefill, name, lambda *args: None)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.plugins.hook_registry": types.SimpleNamespace(
                    HookRegistry=registry, HookType=kinds
                ),
                "sglang_fl.disaggregation.patch": native,
                "isolated_dsv41_hooks.prefill": prefill,
            },
        ):
            self.module.apply_dsv41_patches()
            self.module.apply_dsv41_patches()
        self.assertEqual(registry.register.call_count, 4)
        self.assertEqual(registry.register.call_args_list[0].args[2], kinds.REPLACE)
        native.apply_disaggregation_patch.assert_called_once_with()
        targets = [call.args[0] for call in registry.register.call_args_list]
        self.assertTrue(all(target.startswith("sglang.srt.") for target in targets))


if __name__ == "__main__":
    unittest.main()
