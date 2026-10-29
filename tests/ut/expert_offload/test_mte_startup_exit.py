# SPDX-License-Identifier: Apache-2.0
"""启动退出日志取证的 CPU 测试，不加载 NPU 或模型。"""

import ast
import datetime as dt
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class StartupExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[3] / "script/inspect_mte_startup_exit.py"
        spec = importlib.util.spec_from_file_location("mte_startup_exit_test", cls.source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def service(self, root, extra=""):
        path = root / "consumer_join/mte/inference_python.log"
        path.parent.mkdir(parents=True)
        path.write_text(
            "(EngineCore pid=91939) INFO 10-29 20:35:40 [MOE-MTE] dispatch enabled\n"
            "(EngineCore pid=91939) INFO 10-29 20:35:44 Loading weights took 3.75 seconds\n"
            f"{extra}"
            "(APIServer pid=91361) Traceback (most recent call last):\n"
            "(APIServer pid=91361) RuntimeError: Engine core initialization failed; Failed core proc(s): {}\n"
            "(APIServer pid=91361) [ERROR] 2026-10-29-20:35:44 application exception\n",
            encoding="utf-8",
        )
        return path

    def test_identity_uses_full_service_year_and_separate_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.service(Path(directory))
            value = self.module.identity(path.read_text(encoding="utf-8").splitlines())
        self.assertEqual(value[:4], (["91939"], ["91361"], "91939", "91361"))
        self.assertEqual(value[-1], (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59)))

    def test_unknown_year_or_invalid_date_does_not_use_current_year(self):
        for line in (
            "(APIServer pid=2) ERROR 10-29 20:35:44 failed",
            "(APIServer pid=2) [ERROR] 2026-99-29-20:35:44 failed",
        ):
            self.assertIsNone(self.module.identity(["(EngineCore pid=1) INFO 10-29 20:35:44 ready", line])[-1])

    def test_multiple_process_sets_disable_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.service(root, "(EngineCore_DP1 pid=91940) INFO 10-29 20:35:44 ready\n")
            with patch.object(self.module, "pid_files") as files, patch.object(self.module.subprocess, "run") as run:
                report = self.module.collect(root, [root / "native"])
            files.assert_not_called()
            run.assert_not_called()
        self.assertIn("AMBIGUOUS_PROCESS_SET=True", report)
        self.assertIn("PROCESS_EXIT=NOT_ESTABLISHED", report)

    def test_native_requires_exact_time_and_does_not_treat_submit_as_fault(self):
        window = (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59))
        lines = [
            "[ERROR] 2026-10-29-11:16:17 old task failed",
            "[INFO] 2026-10-29-20:35:43 Submit task=123",
            "[ERROR] 2026-10-29-20:35:44 Runtime synchronize failed",
            "    native stack frame",
            "[ERROR] 2026-99-29-20:35:44 invalid date failure",
            "[ERROR] 2026-10-28-20:35:44 previous day failure",
            "undated failed task",
            "[INFO] 2026-10-29-20:35:45 process exit code=0",
        ]
        faults, tail, count = self.module.native_candidates(lines, window)
        self.assertEqual(faults, [2, 3])
        self.assertEqual(tail, [2, 7])
        self.assertEqual(count, 3)
        self.assertEqual(self.module.native_candidates(lines, None), ([], [], 0))

    def test_pid_boundaries_missing_root_and_empty_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "plog-91939.log",
                "plog-919390.log",
                "plog-191939.log",
                "plog-91939.gz",
                "core.91939",
                "dump_91939.bin",
            ):
                (root / name).write_text("native", encoding="utf-8")
            files, _ = self.module.pid_files(root, ["91939"])
            self.assertEqual([path.name for path in files], ["plog-91939.log"])
            self.assertEqual(self.module.pid_files(root, [])[0], [])
            self.assertIn("UNAVAILABLE", self.module.pid_files(root / "absent", ["91939"])[1][0])

    def test_api_rotation_cannot_displace_all_engine_logs(self):
        files = [Path(f"plog-91361-{i}.log") for i in range(30)]
        files.extend(Path(f"plog-91939-{i}.log") for i in range(30))
        selected = self.module.select_files(files, "91939", "91361")
        self.assertEqual(len(selected), 12)
        self.assertEqual(sum("91939" in path.name for path in selected), 9)
        self.assertEqual(sum("91361" in path.name for path in selected), 3)

    def test_no_fault_exit_zero_is_not_causal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self.service(root)
            original = service.read_bytes()
            native = root / "native"
            native.mkdir()
            (native / "plog-91939.log").write_text("[INFO] 2026-10-29-20:35:44 process exit code=0\n", encoding="utf-8")
            with patch.object(self.module, "kernel_excerpt", return_value=["KERNEL_UNAVAILABLE"]):
                report = self.module.collect(root, [native])
            self.assertEqual(service.read_bytes(), original)
            self.assertEqual(self.module.service_path(root / "consumer_join"), service.resolve())
        self.assertIn("NATIVE_NORMAL_TAIL", report)
        self.assertNotIn("NATIVE_FAULT_CANDIDATE", report)
        self.assertIn("CAUSE=NOT_ESTABLISHED", report)

    def test_tail_offset_reports_unread_bytes_and_discards_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plog.log"
            path.write_bytes(b"prefix-long-line\nnext\nlast\n")
            with patch.object(self.module, "TAIL_BYTES", 13):
                size, offset, lines = self.module.read_tail(path)
        self.assertEqual(size, 27)
        self.assertEqual(offset, 17)
        self.assertEqual(lines, ["next", "last"])

    def test_unprefixed_native_failure_is_preserved(self):
        lines = ["(EngineCore pid=1) INFO 10-29 20:35:40 ready"]
        lines.extend(["unprefixed filler"] * 70)
        lines.extend(["terminate called after throwing an instance", "native stack", "(APIServer pid=2) Traceback"])
        report = "\n".join(self.module.service_excerpt(lines, "1", "2"))
        self.assertIn("terminate called", report)
        self.assertIn("native stack", report)
        lines.extend(["(APIServer pid=2) stack frame"] * 50)
        lines.append("(APIServer pid=2) RuntimeError: Failed core proc(s): {'core': -11}")
        report = "\n".join(self.module.service_excerpt(lines, "1", "2"))
        self.assertIn("{'core': -11}", report)
        for line in ("exitcode=-11", "exit code 137", "Received signal 11", "killed by signal 9"):
            self.assertIsNotNone(self.module.FAULT.search(line))
        for line in ("process exit code=0", "Submit task=137"):
            self.assertIsNone(self.module.FAULT.search(line))

    def test_kernel_excludes_old_error_and_does_not_claim_pid_identity(self):
        window = (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59))
        result = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                "2026-10-29T11:16:17+0800 python[1903358]: segfault old\n"
                "2026-10-29T20:35:44+0800 Killed process 99999 (python)\n"
            ),
        )
        with patch.object(self.module.subprocess, "run", return_value=result) as run:
            report = "\n".join(self.module.kernel_excerpt(window, ["91939"]))
        self.assertNotIn("1903358", report)
        self.assertIn("pid_text_match=False", report)
        self.assertIn("namespace", report)
        self.assertEqual(run.call_args.args[0], ["dmesg", "--time-format", "iso", "--color=never"])

    def test_missing_service_and_permission_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "找不到"):
                self.module.service_path(root)
            self.service(root)
            native = root / "native"
            native.mkdir()
            (native / "plog-91939.log").write_text("", encoding="utf-8")
            with (
                patch.object(self.module, "read_tail", side_effect=PermissionError("denied")),
                patch.object(self.module, "kernel_excerpt", return_value=[]),
            ):
                report = self.module.collect(root, [native])
        self.assertIn("NATIVE_UNREADABLE", report)
        with patch.object(self.module.subprocess, "run", side_effect=PermissionError("dmesg denied")):
            self.assertIn(
                "KERNEL_UNAVAILABLE", self.module.kernel_excerpt((dt.datetime.min, dt.datetime.max), ["1"])[0]
            )

    def test_output_is_exclusive_and_does_not_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(self.module.uuid, "uuid4", return_value=SimpleNamespace(hex="fixed")):
                summary = self.module.write_summary("first", root)
                with self.assertRaises(FileExistsError):
                    self.module.write_summary("second", root)
            self.assertEqual(summary.read_text(encoding="utf-8"), "first")

    def test_service_symlink_outside_run_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.log"
            outside.write_text("private", encoding="utf-8")
            run = root / "run"
            link = run / "mte/inference_python.log"
            link.parent.mkdir(parents=True)
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest("环境未提供符号链接权限")
            with self.assertRaisesRegex(ValueError, "越出"):
                self.module.service_path(run)

    def test_only_standard_library_imports_and_no_unbounded_commands(self):
        tree = ast.parse(self.source.read_text(encoding="utf-8"))
        allowed = {"argparse", "datetime", "os", "re", "subprocess", "uuid", "pathlib"}
        imports = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        imports.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        self.assertLessEqual(imports, allowed)
        self.assertNotIn("torch", imports)


if __name__ == "__main__":
    unittest.main()
