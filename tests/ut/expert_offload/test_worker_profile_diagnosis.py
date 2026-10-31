# SPDX-License-Identifier: Apache-2.0
"""Read-only diagnostics must distinguish raw-data presence from usable exports."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class WorkerProfileDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(script), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "worker_diagnosis_test", script / "diagnose_worker_profile.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_raw_inventory_does_not_confuse_framework_with_device_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "FRAMEWORK").mkdir()
            (root / "FRAMEWORK/torch.op_range").write_bytes(b"framework")
            (root / "PROF_1").mkdir()
            inventory = self.module.raw_inventory(root)
            self.assertGreater(inventory["framework_bytes"], 0)
            self.assertEqual(inventory["cann"][0]["data"], [])
            data = root / "PROF_1/device_7/data"
            data.mkdir(parents=True)
            (data / "sample").write_bytes(b"raw")
            self.assertEqual(self.module.raw_inventory(root)["cann"][0]["data"][0]["bytes"], 3)

    def test_error_excerpt_retains_underlying_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parser.log"
            path.write_text("Traceback\n  parse()\nOSError: [Errno 28] No space left on device\n", encoding="utf-8")
            self.assertIn("No space left on device", self.module.excerpt(path))

    def test_default_visits_failed_cases_and_one_success_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case in self.module.CASES:
                worker = root / "profile" / case / "cpp/rank_ascend_pt"
                worker.mkdir(parents=True)
                (root / case).mkdir()
                status = "complete_with_warnings" if case in ("baseline", "allocated") else "invalid"
                (root / case / "profile_report.json").write_text(json.dumps(dict(profile_status=status)))
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch.object(self.module, "audit_worker") as audit, contextlib.redirect_stdout(io.StringIO()):
                self.module.main([str(root)])
            self.assertEqual(
                [call.args[0].parts[-3] for call in audit.call_args_list],
                ["baseline", "atomic", "spin", "baseline_repeat"],
            )
            self.assertEqual([call.kwargs["check_trace"] for call in audit.call_args_list], [False, True, True, True])
            self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_truncated_trace_is_not_repaired_or_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary)
            trace = worker / "ASCEND_PROFILER_OUTPUT/trace_view.json"
            trace.parent.mkdir()
            trace.write_bytes(b'[{"ph":"X"')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.module.audit_worker(worker)
            self.assertIn("TRACE_INVALID", output.getvalue())
            self.assertEqual(trace.read_bytes(), b'[{"ph":"X"')


if __name__ == "__main__":
    unittest.main()
