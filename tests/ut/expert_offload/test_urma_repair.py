# SPDX-License-Identifier: Apache-2.0
"""URMA补库流程回归；不调用Docker、不修改宿主机或容器。"""

import ast
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class UrmaRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/repair_moe_urma.py"
        spec = importlib.util.spec_from_file_location("urma_repair", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_accepts_arm_and_rejects_x86_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library"
            for machine in (183, 62):
                header = bytearray(20)
                header[:6] = b"\x7fELF\x02\x01"
                header[18:20] = machine.to_bytes(2, "little")
                path.write_bytes(header)
                if machine == 183:
                    actual, digest = self.module.arm_library(path)
                    self.assertEqual(actual, path.resolve())
                    self.assertEqual(len(digest), 64)
                else:
                    with self.assertRaises(ValueError):
                        self.module.arm_library(path)

    def run_plan(self, *, apply=True, link_failure=False, host_exit=0):
        calls = []

        def execute(label, argv, **kwargs):
            calls.append((label, argv, kwargs))
            if "inspect" in argv:
                return SimpleNamespace(returncode=0, stdout='[{"State":{"Running":true}}]')
            if kwargs.get("source") == self.module.CONTAINER_PREPARE:
                return SimpleNamespace(returncode=0, stdout=json.dumps(list(self.module.LIBRARIES)))
            if link_failure and kwargs.get("source") == self.module.CONTAINER_VALIDATE:
                raise RuntimeError("unresolved libc ABI")
            return SimpleNamespace(returncode=host_exit if argv[0] == "timeout" else 0, stdout="")

        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(self.module.sys, "platform", "linux"),
            patch.object(self.module.platform, "machine", return_value="aarch64"),
            patch.object(self.module.shutil, "which", return_value="tool"),
            patch.object(self.module, "arm_library", side_effect=lambda p: (p, hashlib.sha256(b"test").hexdigest())),
            patch.object(Path, "read_bytes", return_value=b"test"),
            patch.object(self.module, "execute", side_effect=execute),
        ):
            try:
                result = self.module.main(["--container", "test", *(["--apply"] if apply else [])])
            except RuntimeError:
                result = "failed"
        return result, calls

    def test_plan_is_read_only_without_apply(self):
        result, calls = self.run_plan(apply=False)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], ["docker", "inspect", "test"])

    def test_apply_copies_only_two_libraries_and_checks_link_before_register(self):
        result, calls = self.run_plan()
        self.assertEqual(result, 0)
        self.assertFalse(any(args[:2] == ["docker", "cp"] for _, args, _ in calls))
        copies = [args for _, args, _ in calls if self.module.CONTAINER_WRITE in args]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(args[-3].startswith("/opt/moe-urma/stdin-") for args in copies))
        self.assertTrue(all(args[:4] == ["docker", "exec", "-i", "test"] for args in copies))
        sources = [kwargs.get("source") for _, _, kwargs in calls]
        self.assertLess(sources.index(self.module.CONTAINER_VALIDATE), sources.index(self.module.CONTAINER_REGISTER))
        self.assertLess(sources.index(self.module.CONTAINER_REGISTER), sources.index(self.module.CONTAINER_CHECK))

    def test_link_failure_does_not_register(self):
        result, calls = self.run_plan(link_failure=True)
        self.assertEqual(result, "failed")
        self.assertNotIn(self.module.CONTAINER_REGISTER, [kwargs.get("source") for _, _, kwargs in calls])

    def test_host_timeout_is_not_reported_as_device_ready(self):
        result, _ = self.run_plan(host_exit=124)
        self.assertEqual(result, 2)

    def test_container_payloads_compile(self):
        for name in (
            "CONTAINER_PREPARE",
            "CONTAINER_WRITE",
            "CONTAINER_VALIDATE",
            "CONTAINER_REGISTER",
            "CONTAINER_CHECK",
        ):
            compile(getattr(self.module, name), name, "exec")

    def test_stdin_receiver_checks_integrity_and_publishes_complete_file(self):
        parsed = ast.parse(self.module.CONTAINER_WRITE)
        definitions = ast.Module(
            body=[node for node in parsed.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
            type_ignores=[],
        )
        namespace = {}
        exec(compile(definitions, "receiver", "exec"), namespace)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            target = Path(folder).resolve()
            raw = b"binary\x00\xff\r\ndata"
            digest = hashlib.sha256(raw).hexdigest()
            payload = base64.b64encode(raw)
            receive = namespace["receive"]
            with self.assertRaises(AssertionError):
                receive(target, "liburma.so.0", digest, base64.b64encode(b"broken"))
            self.assertEqual(list(target.iterdir()), [])
            receive(target, "liburma.so.0", digest, payload)
            receive(target, "liburma.so.0", digest, payload)
            self.assertEqual((target / "liburma.so.0").read_bytes(), raw)
            self.assertEqual([p.name for p in target.iterdir()], ["liburma.so.0"])
            with self.assertRaises(AssertionError):
                receive(target, "liburma.so.0", hashlib.sha256(b"other").hexdigest(), base64.b64encode(b"other"))
            self.assertEqual((target / "liburma.so.0").read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
