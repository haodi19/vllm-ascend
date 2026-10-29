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

    def test_native_requires_exact_time_and_preserves_original_line_numbers(self):
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plog.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = self.module.scan_native(path, window)
        self.assertEqual([record["line"] for record in result["records"]], [3])
        self.assertEqual(result["records"][0]["context"], [(4, "    native stack frame")])
        self.assertEqual(result["normal_tail"][-1], (8, lines[7]))
        self.assertEqual(result["window_lines"], 3)
        self.assertGreaterEqual(result["undated_faults"], 1)
        self.assertNotIn("previous day failure", str(result["records"]))

    def test_runtime_timeout_parameters_are_not_failures(self):
        lines = (
            "[DEBUG] RUNTIME(91939,):2026-10-29-20:35:40.511.143 [api_error.cc:1091] 91939 "
            "StreamSynchronize:stream synchronize entry, stream_id=61, timeout=-1ms.",
            "[INFO] RUNTIME(91939,):2026-10-29-20:35:44.321.547 [david_event_task_v200_base.cc:138] "
            "92009 ConstructDavidSqeForEventRecordTask: event_record: device_id=7, stream_id=61, "
            "task_id=1415, task_sn=11708, subType=event record use single notify, timeout=0s.",
            "[DEBUG] RUNTIME(91939,):2026-10-29-20:35:44.487.322 StreamSynchronize entry, timeout=60000ms.",
            "[INFO] RUNTIME(91939,):2026-10-29-20:35:29.104.128 [api_error.cc:1104] 91939 "
            "StreamSynchronize: stream synchronize exit, stream_id=61, result=0x0",
            "[DEBUG] 2026-10-29-20:35:44 set timeout value 500 ms",
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertFalse(self.module.is_fault(line))
                self.assertIsNone(self.module.fault_kind(line))

    def test_actual_failure_with_timeout_parameter_is_preserved(self):
        for line in (
            "[ERROR] RUNTIME(91939,):2026-10-29-20:35:44.490.000 Device exception; timeout=0s.",
            "[ERROR] RUNTIME(91939,):2026-10-29-20:35:44.490.000 StreamSynchronize failed; timeout=-1ms.",
            "[WARNING] 2026-10-29-20:35:44 task timed out, timeout=60000ms",
            "2026-10-29-20:35:44 stream timeout occurred",
            "2026-10-29-20:35:44 wait timeout",
            "2026-10-29-20:35:44 task timeout",
            "2026-10-29-20:35:44 sync timeout",
            "Fatal Python error: Segmentation fault",
            "process exitcode=-11",
        ):
            with self.subTest(line=line):
                self.assertTrue(self.module.is_fault(line))
                self.assertIsNotNone(self.module.fault_kind(line))

    def test_process_signals_require_exit_evidence_not_event_parameters(self):
        for line in (
            "native worker received SIGTERM",
            "native worker received SIGBUS",
            "native worker received SIGILL",
            "native worker received SIGFPE",
            "Received signal 15",
            "Received signal=11",
            "killed by signal=11",
            "process exited with signal 15",
        ):
            with self.subTest(line=line):
                self.assertEqual(self.module.fault_kind(line), "fatal")
        for line in ("event signal=11", "event signal 11"):
            with self.subTest(line=line):
                self.assertFalse(self.module.is_fault(line))

    def test_timeout_noise_cannot_hide_real_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.service(root)
            native = root / "native"
            native.mkdir()
            normal = (
                "[DEBUG] RUNTIME(91939,):2026-10-29-20:35:44.100.000 "
                "StreamSynchronize:stream synchronize entry, stream_id=61, timeout=-1ms.\n"
            )
            actual = (
                "[ERROR] RUNTIME(91939,):2026-10-29-20:35:44.200.000 "
                "Runtime synchronize failed: FAILURE_BETWEEN_NORMAL_PARAMETERS\n"
            )
            (native / "plog-91939.log").write_text(normal * 1500 + actual + normal * 1500, encoding="utf-8")
            with patch.object(self.module, "kernel_excerpt", return_value=[]):
                report = self.module.collect(root, [native])
        self.assertIn("FAILURE_BETWEEN_NORMAL_PARAMETERS", report)
        fault_lines = [line for line in report.splitlines() if "NATIVE_FAULT_CANDIDATE" in line]
        self.assertFalse(any("stream synchronize entry" in line for line in fault_lines))

    def test_errors_in_other_files_cannot_exhaust_fatal_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.service(root)
            native = root / "native"
            native.mkdir()
            (native / "plog-91939-a.log").write_text(
                "2026-10-29-20:35:44 Fatal Python error: Segmentation fault FATAL_EXIT_MARKER\n",
                encoding="utf-8",
            )
            for index in range(5):
                (native / f"plog-91939-z{index}.log").write_text(
                    "[ERROR] 2026-10-29-20:35:43 a preceding unrelated query failed\n" * 30,
                    encoding="utf-8",
                )
            with patch.object(self.module, "kernel_excerpt", return_value=[]):
                report = self.module.collect(root, [native])
        self.assertIn("FATAL_EXIT_MARKER", report)
        self.assertIn("PROCESS_EXIT=NOT_ESTABLISHED", report)

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

    def test_full_scan_preserves_error_in_file_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plog.log"
            content = (
                "[ERROR] 2026-10-29-20:35:44 prefix failure 原始首行\n"
                + "[INFO] 2026-10-29-20:35:44 successful submission\n" * 512
            ).encode("utf-8")
            path.write_bytes(content)
            window = (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59))
            result = self.module.scan_native(path, window)
        self.assertEqual(result["size"], len(content))
        self.assertEqual(result["bytes_read"], len(content))
        self.assertEqual(result["lines"], 513)
        self.assertEqual(result["records"][0]["line"], 1)
        self.assertIn("prefix failure", result["records"][0]["text"])

    def test_short_native_dates_use_service_window_year(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plog.log"
            path.write_text(
                "WARNING 10-28 20:35:44 earlier day failed\nERROR 10-29 20:35:44 same day failed\n",
                encoding="utf-8",
            )
            window = (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59))
            result = self.module.scan_native(path, window)
        self.assertEqual(result["window_lines"], 1)
        self.assertEqual([record["line"] for record in result["records"]], [2])
        self.assertEqual(result["undated_faults"], 0)

    def test_undated_fatal_is_visible_without_window_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plog.log"
            path.write_text(
                "Fatal Python error: Segmentation fault\n[INFO] 2026-10-29-20:35:44 task submitted\n",
                encoding="utf-8",
            )
            window = (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59))
            result = self.module.scan_native(path, window)
        self.assertEqual(result["records"], [])
        self.assertEqual(result["undated_faults"], 1)
        self.assertEqual(result["undated_records"], [(1, "Fatal Python error: Segmentation fault")])

    def test_undated_cleanup_errors_do_not_hide_uncorrelated_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.service(root)
            native = root / "native"
            native.mkdir()
            (native / "plog-91939.log").write_text(
                "Fatal Python error: Segmentation fault UNDATED_FATAL_MARKER\n"
                + "cleanup failed\n" * 5
                + "[INFO] 2026-10-29-20:35:44 task submitted\n",
                encoding="utf-8",
            )
            with patch.object(self.module, "kernel_excerpt", return_value=[]):
                report = self.module.collect(root, [native])
        undated = [line for line in report.splitlines() if "NATIVE_UNDATED_UNCORRELATED" in line]
        self.assertTrue(any("UNDATED_FATAL_MARKER" in line for line in undated))
        attributed = [line for line in report.splitlines() if "NATIVE_FAULT_CANDIDATE" in line]
        self.assertFalse(any("UNDATED_FATAL_MARKER" in line for line in attributed))
        self.assertIn("PROCESS_EXIT=NOT_ESTABLISHED", report)

    def test_native_count_covers_errors_omitted_from_bounded_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plog.log"
            path.write_text(
                "".join(f"[ERROR] 2026-10-29-20:35:44 query failed marker={index}\n" for index in range(200)),
                encoding="utf-8",
            )
            window = (dt.datetime(2026, 10, 29, 20, 35, 29), dt.datetime(2026, 10, 29, 20, 35, 59))
            result = self.module.scan_native(path, window)
        self.assertEqual(result["counts"]["error"], 200)
        self.assertLess(len(result["records"]), 200)
        self.assertEqual(result["records"][0]["line"], 1)
        self.assertEqual(result["records"][-1]["line"], 200)

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
            self.assertTrue(self.module.is_fault(line))
        for line in ("process exit code=0", "Submit task=137"):
            self.assertFalse(self.module.is_fault(line))

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
                patch.object(self.module, "scan_native", side_effect=PermissionError("denied")),
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
        allowed = {"argparse", "collections", "datetime", "os", "re", "subprocess", "uuid", "pathlib"}
        imports = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        imports.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        self.assertLessEqual(imports, allowed)
        self.assertNotIn("torch", imports)


if __name__ == "__main__":
    unittest.main()
