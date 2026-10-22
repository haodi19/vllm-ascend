"""离线解析流程测试，不依赖 NPU 或修改真实 profile。"""

import contextlib
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RecoveryTests(unittest.TestCase):
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

                def reparse(command, valid=valid, **kwargs):
                    target = Path(command[-1]) / "test_ascend_pt"
                    self.assertEqual((target / "FRAMEWORK/torch.op_range").read_bytes(), b"raw data")
                    exported = target / "ASCEND_PROFILER_OUTPUT"
                    self.assertFalse(exported.exists())
                    exported.mkdir()
                    payload = '[{"name":"moe_offload/update_weights"}]' if valid else '[{"name":"kernel"}]'
                    (exported / "trace_view.json").write_text(payload)

                with (
                    patch.object(sys, "argv", ["recover", str(source), "--output-dir", str(output)]),
                    patch.object(module["subprocess"], "run", side_effect=reparse),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(module["main"](), 0 if valid else 1)
                self.assertEqual((old / "trace_view.json").read_text(), "broken")


if __name__ == "__main__":
    unittest.main()
