# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for observing actual child exit states without extra waits."""

import contextlib
import importlib.util
import io
import json
import multiprocessing
import multiprocessing.process
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def exit_child(code):
    os._exit(code)


class StartupBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[3] / "script/mte_startup_bootstrap.py"
        spec = importlib.util.spec_from_file_location("mte_startup_bootstrap_test", cls.source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_import_does_not_install_observers(self):
        before = multiprocessing.process.BaseProcess.start
        _, points = self.module.popen_observation_points()
        methods = [getattr(owner, name) for owner, name in points]
        spec = importlib.util.spec_from_file_location("mte_startup_bootstrap_import_check", self.source)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        self.assertIs(multiprocessing.process.BaseProcess.start, before)
        self.assertEqual([getattr(owner, name) for owner, name in points], methods)

    def console_entry(self, root, declaration):
        path = root / "vllm"
        path.write_text(f"#!{declaration}\n# console entry content is not executed\n", encoding="utf-8")
        return path

    def test_interpreter_validation_accepts_matching_absolute_python(self):
        with tempfile.TemporaryDirectory() as directory:
            entrypoint = self.console_entry(Path(directory), shlex.quote(Path(sys.executable).as_posix()))
            with patch.object(self.module.shutil, "which", return_value=str(entrypoint)):
                metadata = self.module.validate_vllm_interpreter()
        self.assertEqual(metadata["python_executable"], str(Path(sys.executable).resolve()))
        self.assertEqual(metadata["python_version"], sys.version.split()[0])
        self.assertEqual(metadata["vllm_entrypoint"], str(entrypoint))

    def test_interpreter_validation_accepts_simple_env_and_env_split(self):
        for declaration in ("/usr/bin/env python3", "/usr/bin/env -S python3"):
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory() as directory:
                entrypoint = self.console_entry(Path(directory), declaration)
                paths = {"vllm": str(entrypoint), "python3": sys.executable}
                with patch.object(self.module.shutil, "which", side_effect=paths.get):
                    metadata = self.module.validate_vllm_interpreter()
                self.assertEqual(metadata["python_executable"], str(Path(sys.executable).resolve()))

    def test_interpreter_validation_accepts_equivalent_venv_symlink_when_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            executable = base / "python3"
            executable.write_text("not executed", encoding="utf-8")
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text("home = base\n", encoding="utf-8")
            link = venv / "bin/python3"
            try:
                link.symlink_to(executable)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            entrypoint = self.console_entry(root, shlex.quote(link.as_posix()))
            with (
                patch.object(self.module.shutil, "which", return_value=str(entrypoint)),
                patch.object(self.module.sys, "executable", str(link)),
                patch.object(self.module.sys, "prefix", str(venv)),
                patch.object(self.module.sys, "base_prefix", str(base)),
            ):
                metadata = self.module.validate_vllm_interpreter()
            self.assertEqual(metadata["python_executable"], str(executable.resolve()))
            self.assertEqual(metadata["declared_interpreter"], str(link))

    def test_interpreter_validation_rejects_mismatch_missing_and_unknown_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other_python = root / "python3"
            other_python.write_text("not executed", encoding="utf-8")
            entrypoint = self.console_entry(root, shlex.quote(other_python.as_posix()))
            with (
                patch.object(self.module.shutil, "which", return_value=str(entrypoint)),
                self.assertRaisesRegex(RuntimeError, "interpreter mismatch"),
            ):
                self.module.validate_vllm_interpreter()
            with (
                patch.object(self.module.shutil, "which", return_value=None),
                self.assertRaisesRegex(RuntimeError, "no vllm entrypoint"),
            ):
                self.module.validate_vllm_interpreter()
            for declaration in ("/bin/sh", "/usr/bin/env -S python3 -u", "python3", ""):
                entrypoint = self.console_entry(root, declaration)
                with (
                    self.subTest(declaration=declaration),
                    patch.object(self.module.shutil, "which", return_value=str(entrypoint)),
                    self.assertRaises(RuntimeError),
                ):
                    self.module.validate_vllm_interpreter()
            entrypoint.write_text("not a console script\n", encoding="utf-8")
            with (
                patch.object(self.module.shutil, "which", return_value=str(entrypoint)),
                self.assertRaisesRegex(RuntimeError, "recognized Python"),
            ):
                self.module.validate_vllm_interpreter()

    def test_same_executable_cannot_hide_different_virtual_environment_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declared_venv = root / "declared"
            (declared_venv / "bin").mkdir(parents=True)
            (declared_venv / "pyvenv.cfg").write_text("home = base\n", encoding="utf-8")
            executable = declared_venv / "bin/python3"
            executable.write_text("not executed", encoding="utf-8")
            current_venv = root / "current"
            current_venv.mkdir()
            entrypoint = self.console_entry(root, shlex.quote(executable.as_posix()))
            with (
                patch.object(self.module.shutil, "which", return_value=str(entrypoint)),
                patch.object(self.module.sys, "executable", str(executable)),
                patch.object(self.module.sys, "prefix", str(current_venv)),
                patch.object(self.module.sys, "base_prefix", str(root)),
                self.assertRaisesRegex(RuntimeError, "virtual-environment prefix"),
            ):
                self.module.validate_vllm_interpreter()

    def test_observation_preserves_call_arguments_result_and_exception(self):
        records = []
        observer = self.module.ProcessObserver(records.append)
        popen = SimpleNamespace(pid=123, method="test")
        original = Mock(side_effect=[None, 7, 7])
        wrapped = observer.wrap_observation(original)
        self.assertIsNone(wrapped(popen, 42, option="test"))
        self.assertEqual(wrapped(popen, 42, option="test"), 7)
        self.assertEqual(wrapped(popen, 42, option="test"), 7)
        self.assertEqual(original.call_count, 3)
        for call in original.call_args_list:
            self.assertEqual(call.args, (popen, 42))
            self.assertEqual(call.kwargs, {"option": "test"})
        self.assertEqual([record["returncode"] for record in records], [7])
        failure = RuntimeError("original failure")
        throwing = Mock(side_effect=failure)
        with self.assertRaises(RuntimeError) as caught:
            observer.wrap_observation(throwing)(popen)
        self.assertIs(caught.exception, failure)
        throwing.assert_called_once_with(popen)

    def test_existing_cached_exitcode_is_recorded_without_extra_poll(self):
        records = []
        observer = self.module.ProcessObserver(records.append)
        popen = SimpleNamespace(pid=321, method="fork", returncode=-signal.SIGTERM)
        original = Mock(side_effect=lambda process: process.returncode)
        wrapped = observer.wrap_observation(original)
        self.assertEqual(wrapped(popen), -signal.SIGTERM)
        original.assert_called_once_with(popen)
        self.assertEqual(records[0]["signal"], "SIGTERM")
        self.assertEqual(records[0]["signal_number"], signal.SIGTERM)

    def test_diagnostic_emission_failure_does_not_replace_child_result(self):
        observer = self.module.ProcessObserver(Mock(side_effect=RuntimeError("output unavailable")))
        popen = SimpleNamespace(pid=321, method="test")
        original = Mock(return_value=7)
        self.assertEqual(observer.wrap_observation(original)(popen), 7)
        original.assert_called_once_with(popen)
        process = SimpleNamespace(_popen=popen, name="EngineCore")
        signal_request = Mock(return_value=None)
        self.assertIsNone(observer.wrap_signal_request(signal_request, "kill", "SIGKILL")(process))
        signal_request.assert_called_once_with(process)

    def test_fork_inherited_observer_does_not_report_from_other_pid(self):
        records = []
        observer = self.module.ProcessObserver(records.append)
        with patch.object(self.module.os, "getpid", return_value=observer.owner_pid + 1):
            observer.observed_exit(SimpleNamespace(pid=999), 9)
            observer.record("unexpected_child_record")
        self.assertEqual(records, [])

    def test_signal_request_preserves_original_call_and_is_not_delivery_claim(self):
        records = []
        observer = self.module.ProcessObserver(records.append)
        process = SimpleNamespace(_popen=SimpleNamespace(pid=456), name="EngineCore")
        sentinel = object()
        original = Mock(return_value=sentinel)
        wrapped = observer.wrap_signal_request(original, "terminate", "SIGTERM")
        self.assertIs(wrapped(process, "arg", option=True), sentinel)
        original.assert_called_once_with(process, "arg", option=True)
        self.assertEqual(records[0]["event"], "signal_request")
        self.assertEqual(records[0]["child_pid"], 456)
        self.assertNotIn("returncode", records[0])

    def test_real_nonzero_child_exit_for_every_available_start_method(self):
        for method in multiprocessing.get_all_start_methods():
            with self.subTest(method=method):
                records = []
                observer = self.module.ProcessObserver(records.append).install()
                process = multiprocessing.get_context(method).Process(
                    target=exit_child, args=(7,), name="EngineCore-test"
                )
                try:
                    process.start()
                    child_pid = process.pid
                    process.join(15)
                    self.assertFalse(process.is_alive())
                    self.assertEqual(process.exitcode, 7)
                    self.assertEqual(process.exitcode, 7)
                    starts = [record for record in records if record["event"] == "start"]
                    exits = [record for record in records if record["event"] == "exit"]
                    self.assertEqual(len(starts), 1)
                    self.assertEqual(len(exits), 1)
                    self.assertEqual(starts[0]["child_pid"], child_pid)
                    self.assertEqual(exits[0]["child_pid"], child_pid)
                    self.assertEqual(exits[0]["name"], "EngineCore-test")
                    self.assertEqual(exits[0]["start_method"], method)
                    self.assertEqual(exits[0]["observer_pid"], os.getpid())
                    self.assertEqual(exits[0]["returncode"], 7)
                    self.assertIsNone(exits[0]["signal"])
                    self.assertFalse(any(record["event"] == "observer_gap" for record in records))
                finally:
                    if process.pid is not None:
                        if process.is_alive():
                            process.kill()
                            process.join(5)
                        process.close()
                    observer.close()

    @unittest.skipUnless(os.name == "posix", "POSIX signal exit status requires POSIX")
    def test_real_sigkill_preserves_request_before_negative_exit(self):
        records = []
        observer = self.module.ProcessObserver(records.append).install()
        process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(30,), name="EngineCore-kill")
        try:
            process.start()
            process.kill()
            process.join(15)
            self.assertEqual(process.exitcode, -signal.SIGKILL)
            actions = [record["event"] for record in records]
            self.assertLess(actions.index("signal_request"), actions.index("exit"))
            exits = [record for record in records if record["event"] == "exit"]
            self.assertEqual(len(exits), 1)
            self.assertEqual(exits[0]["signal"], "SIGKILL")
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
            process.close()
            observer.close()

    def test_interpreter_atexit_cleanup_retains_child_exit_observation(self):
        code = """
import importlib.util
import multiprocessing
import os
import sys
spec = importlib.util.spec_from_file_location('bootstrap', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
observer = module.ProcessObserver().install()
child = multiprocessing.get_context('spawn').Process(target=os._exit, args=(7,), name='EngineCore-atexit')
child.start()
# Deliberately no join, poll, exitcode read or observer.close: stdlib atexit owns cleanup.
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(self.source)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [
            json.loads(line.split(self.module.PREFIX, 1)[1])
            for line in result.stderr.splitlines()
            if line.startswith(self.module.PREFIX)
        ]
        exits = [record for record in records if record["event"] == "exit"]
        self.assertEqual(len(exits), 1, result.stderr)
        self.assertEqual(exits[0]["returncode"], 7)
        self.assertEqual(exits[0]["name"], "EngineCore-atexit")

    def test_install_restores_hooks_and_rejects_double_install(self):
        base = multiprocessing.process.BaseProcess
        originals = (base.start, base.terminate, base.kill)
        _, points = self.module.popen_observation_points()
        original_points = [getattr(owner, name) for owner, name in points]
        observer = self.module.ProcessObserver(lambda record: None).install()
        try:
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                observer.install()
        finally:
            observer.close()
        self.assertEqual((base.start, base.terminate, base.kill), originals)
        self.assertEqual([getattr(owner, name) for owner, name in points], original_points)

    def test_explicit_diagnostic_and_serve_are_required_before_install(self):
        for argv in (["serve", "model"], ["--diagnostic"], ["--diagnostic", "bench"]):
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(self.module, "ProcessObserver") as observer,
                patch.object(self.module.faulthandler, "enable") as enable,
                patch.object(self.module, "validate_vllm_interpreter") as validate,
            ):
                with self.assertRaises(SystemExit):
                    self.module.main(argv)
                observer.assert_not_called()
                enable.assert_not_called()
                validate.assert_not_called()

    def test_main_interpreter_mismatch_fails_before_observation_or_vllm(self):
        with (
            patch.object(self.module, "validate_vllm_interpreter", side_effect=RuntimeError("mismatch")),
            patch.object(self.module.faulthandler, "enable") as enable,
            patch.object(self.module, "ProcessObserver") as observer,
            patch.object(self.module, "run_vllm") as run,
            self.assertRaisesRegex(RuntimeError, "mismatch"),
        ):
            self.module.main(["--diagnostic", "serve", "model"])
        enable.assert_not_called()
        observer.assert_not_called()
        run.assert_not_called()

    def test_main_preserves_vllm_arguments_and_keeps_observer_for_atexit(self):
        observer = Mock()
        observer.install.return_value = observer
        arguments = ["serve", "model", "--additional-config", '{"x": true}']
        with (
            patch.object(self.module, "ProcessObserver", return_value=observer),
            patch.object(self.module.faulthandler, "enable") as enable,
            patch.object(self.module, "run_vllm", return_value=17) as run,
            patch.object(self.module, "validate_vllm_interpreter", return_value={"python_executable": "verified"}),
        ):
            self.assertEqual(self.module.main(["--diagnostic", *arguments]), 17)
        run.assert_called_once_with(arguments)
        enable.assert_called_once_with(file=2, all_threads=True)
        observer.close.assert_not_called()
        observer.install.assert_called_once_with(interpreter={"python_executable": "verified"})

    def test_main_preserves_original_exception(self):
        observer = Mock()
        observer.install.return_value = observer
        failure = RuntimeError("engine startup failed")
        with (
            patch.object(self.module, "ProcessObserver", return_value=observer),
            patch.object(self.module.faulthandler, "enable"),
            patch.object(self.module, "run_vllm", side_effect=failure),
            patch.object(self.module, "validate_vllm_interpreter", return_value={}),
            self.assertRaises(RuntimeError) as caught,
        ):
            self.module.main(["--diagnostic", "serve", "model"])
        self.assertIs(caught.exception, failure)
        observer.record.assert_any_call("entry_exception", exception_type="RuntimeError")


if __name__ == "__main__":
    unittest.main()
