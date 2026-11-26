# SPDX-License-Identifier: Apache-2.0
"""Build reuse and invalidation without compilers, Torch, NPU or model processes."""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import native_moe_build as cache
import run_native_moe_compare as runner


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.parent = self.root / "persistent"
        self.modules = {}
        for name in cache.INSTALLED_MODULES:
            path = self.root / (name + ".so")
            path.write_bytes(name.encode())
            self.modules[name] = path
        self.spec = patch.object(
            cache.importlib.util, "find_spec", side_effect=lambda name: SimpleNamespace(origin=str(self.modules[name]))
        )
        self.spec.start()
        self.addCleanup(self.spec.stop)

    def test_source_and_environment_changes_invalidate_but_build_outputs_do_not(self):
        repo = self.root / "repo"
        cann = self.root / "cann"
        cann.mkdir()
        for component in cache.BUILD_COMPONENTS:
            path = repo / "csrc" / component / "code.h"
            path.parent.mkdir(parents=True)
            path.write_text("original")
        (repo / "script").mkdir()
        for name in ("native_moe_build.py", "run_native_moe_compare.py", "run_host_sync_stage2.py"):
            (repo / "script" / name).write_text("original")
        with patch.object(cache, "build_environment", return_value={"soc": "pr"}) as environment:
            first = cache.build_fingerprint(repo, cann, "pr", {})
            generated = repo / "csrc/host_sync/build/object.h"
            generated.parent.mkdir()
            generated.write_text("generated header")
            (repo / "csrc/host_sync/ignored.so").write_bytes(b"binary")
            self.assertEqual(first, cache.build_fingerprint(repo, cann, "pr", {}))
            (repo / "csrc/host_sync/code.h").write_text("changed")
            second = cache.build_fingerprint(repo, cann, "pr", {})
            self.assertNotEqual(first, second)
            environment.return_value = {"soc": "dt"}
            self.assertNotEqual(second, cache.build_fingerprint(repo, cann, "dt", {}))

    def test_environment_checks_abi_cann_tools_without_keying_on_card_number(self):
        cann = self.root / "cann"
        cann.mkdir()
        version = cann / "version.info"
        version.write_text("9.1")
        first = cache.build_environment(cann, "ascend950pr_957c", {"ASCEND_RT_VISIBLE_DEVICES": "6"})
        other_card = cache.build_environment(cann, "ascend950pr_957c", {"ASCEND_RT_VISIBLE_DEVICES": "7"})
        self.assertEqual(first, other_card)
        self.assertEqual(first["prefix"], sys.prefix)
        self.assertIn("torch-npu", first["versions"])
        version.write_text("9.1-updated")
        self.assertNotEqual(first, cache.build_environment(cann, "ascend950pr_957c", {}))

    def test_cached_binaries_require_fingerprint_and_current_installed_artifact_hashes(self):
        library = self.parent / "build_01/lib/libhost_sync_device.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"kernel")
        self.assertIsNone(cache.cached_library(self.parent, "input"))
        cache.remember_build(self.parent, "input", library)
        self.assertEqual(cache.cached_library(self.parent, "input"), library.resolve())
        self.assertIsNone(cache.cached_library(self.parent, "other inputs"))
        for path in (library, *self.modules.values()):
            original = path.read_bytes()
            path.write_bytes(original + b"replaced")
            self.assertIsNone(cache.cached_library(self.parent, "input"))
            path.write_bytes(original)
            self.assertEqual(cache.cached_library(self.parent, "input"), library.resolve())
        library.unlink()
        self.assertIsNone(cache.cached_library(self.parent, "input"))
        (self.parent / cache.CACHE_FILENAME).write_text("incomplete JSON")
        self.assertIsNone(cache.cached_library(self.parent, "input"))

    def simulate(self, *, fail_validation=False):
        args = SimpleNamespace(
            skip_build=False, library=None, build_dir=self.parent, cann_root=self.root, skip_profile=False
        )
        calls, lines, current_build = [], [], []

        def step(label, command, **kwargs):
            calls.append(label)
            if label == "configure handshake":
                current_build[:] = [Path(command[command.index("-B") + 1])]
            if label == "build handshake":
                library = current_build[0] / "lib/libhost_sync_device.so"
                library.parent.mkdir()
                library.write_bytes(b"validated kernel")
            if fail_validation and label == "NPU scale byte-layout parity":
                raise RuntimeError("failed validation")

        with (
            patch.object(runner, "build_lock", side_effect=lambda _: contextlib.nullcontext()),
            patch.object(runner, "build_fingerprint", return_value="same inputs"),
            patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="SOC=ascend950pr_957c\n")),
        ):
            for tokens in (5, 128):
                for case in ("python", "native", "native_full"):
                    args.case, args.max_tokens = case, tokens
                    runner.prepare_native_components(args, {}, step, lines.append)
        return calls, lines

    def test_six_independent_commands_build_once_but_validate_each_time(self):
        calls, lines = self.simulate()
        for label in ("build lrc_policy", "build host_sync", "configure handshake", "build handshake"):
            self.assertEqual(calls.count(label), 1, label)
        self.assertEqual(calls.count("native API check"), 6)
        self.assertEqual(calls.count("NPU scale byte-layout parity"), 6)
        self.assertEqual(sum(line.startswith("BUILD_CACHE_HIT") for line in lines), 5)
        self.assertEqual(len(list(self.parent.glob("build_*"))), 1)
        self.assertTrue((self.parent / cache.CACHE_FILENAME).is_file())

    def test_failed_validation_is_not_cached_and_next_build_uses_new_directory(self):
        with self.assertRaisesRegex(RuntimeError, "failed validation"):
            self.simulate(fail_validation=True)
        self.assertFalse((self.parent / cache.CACHE_FILENAME).exists())
        first = self.parent / "build_01/lib/libhost_sync_device.so"
        evidence = first.read_bytes()
        calls, _ = self.simulate()
        self.assertEqual(calls.count("build handshake"), 1)
        self.assertTrue((self.parent / "build_02/lib/libhost_sync_device.so").is_file())
        self.assertEqual(first.read_bytes(), evidence)

    def test_reinstalled_extension_triggers_new_build(self):
        self.simulate()
        self.modules["_vllm_ascend_lrc"].write_bytes(b"another installation")
        calls, _ = self.simulate()
        self.assertEqual(calls.count("build handshake"), 1)
        record = json.loads((self.parent / cache.CACHE_FILENAME).read_text())
        self.assertIn("build_02", record["library"])

    def test_inputs_changed_during_validation_do_not_publish_a_cache_record(self):
        library = self.parent / "build_01/lib/libhost_sync_device.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"kernel")
        args = SimpleNamespace(skip_build=False, build_dir=self.parent, cann_root=self.root)
        with (
            patch.object(runner, "build_lock", return_value=contextlib.nullcontext()),
            patch.object(runner, "build_fingerprint", side_effect=["before", "changed"]),
            patch.object(runner, "cached_library", return_value=library),
            patch.object(runner, "validate_native_components"),
            patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="SOC=ascend950pr_957c\n")),
            self.assertRaisesRegex(RuntimeError, "changed during preflight"),
        ):
            runner.prepare_native_components(args, {}, lambda *a: None, lambda s: None)
        self.assertFalse((self.parent / cache.CACHE_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
