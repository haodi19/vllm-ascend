"""离线解析流程测试，不依赖 NPU 或修改真实 profile。"""

import contextlib
import io
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


class RecoveryTests(unittest.TestCase):
    def test_timeout_kills_owned_process_group(self):
        script = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(script), *sys.path]):
            module = runpy.run_path(str(script / "recover_moe_profile.py"))
        child = MagicMock()
        child.__enter__.return_value = child
        child.pid = 123
        child.wait.side_effect = [module["subprocess"].TimeoutExpired("parser", 1), 0]
        kill = Mock()
        with (
            patch.object(module["subprocess"], "Popen", return_value=child) as launch,
            patch.dict(
                module["run_parser"].__globals__,
                os=types.SimpleNamespace(name="posix", killpg=kill),
                signal=types.SimpleNamespace(SIGKILL=9),
            ),self.assertRaises(module["subprocess"].TimeoutExpired)
        ):
            module["run_parser"](["parser"], io.StringIO(), 1)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(123, 9)

    def test_diagnostic_excerpt_includes_exception(self):
        script = Path(__file__).resolve().parents[3] / "script"
        module = runpy.run_path(str(script / "diagnose_profile_export.py"))
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "parser.log"
            log.write_text("start\nTraceback:\nfile.py line 1\nValueError: synthetic\nend", encoding="utf-8")
            before = log.read_bytes()
            self.assertIn("ValueError: synthetic", module["excerpt"](log))
            self.assertEqual(log.read_bytes(), before)

    def test_reparse_copy_preserves_original_and_checks_labels(self):
        script = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(script), *sys.path]):
            module = runpy.run_path(str(script / "recover_moe_profile.py"))
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source", root / "recovered"
                worker = source / "mte/python/test_ascend_pt"
                raw = worker / "FRAMEWORK"
                raw.mkdir(parents=True)
                (raw / "torch.op_range").write_bytes(b"raw data")
                old = worker / "ASCEND_PROFILER_OUTPUT"
                old.mkdir()
                (old / "trace_view.json").write_text("broken")

                def reparse(command, stream, timeout, valid=valid):
                    target = Path(command[-1]) / "test_ascend_pt"
                    self.assertEqual((target / "FRAMEWORK/torch.op_range").read_bytes(), b"raw data")
                    exported = target / "ASCEND_PROFILER_OUTPUT"
                    self.assertFalse(exported.exists())
                    exported.mkdir()
                    payload = '[{"name":"moe_offload/update_weights"}]' if valid else '[{"name":"kernel"}]'
                    (exported / "trace_view.json").write_text(payload)

                with (
                    patch.object(sys, "argv", ["recover", str(source), "--output-dir", str(output)]),
                    patch.dict(module["main"].__globals__, run_parser=reparse),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(module["main"](), 0 if valid else 1)
                self.assertEqual((old / "trace_view.json").read_text(), "broken")


if __name__ == "__main__":
    unittest.main()
