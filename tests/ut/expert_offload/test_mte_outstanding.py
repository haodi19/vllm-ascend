# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class OutstandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location("outstanding", scripts / "inspect_mte_outstanding.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_snapshot_excludes_other_wait_types(self):
        lines = [
            "PrintStreamTimeoutSnapshotInfo:stream_id=23, task_id=64, taskType=66 (KERNEL_AIVEC).",
            "PrintStreamTimeoutSnapshotInfo:stream_id=49, task_id=1304, taskType=111 (EVENT_WAIT).",
        ]
        self.assertEqual(self.module.snapshot_targets(lines), {(23, 64)})

    def test_graph_snapshot_includes_mixed_kernel_and_waits(self):
        lines = [
            "PrintStreamTimeoutSnapshotInfo:stream_id=21, task_id=2, taskType=66 (KERNEL_AIVEC).",
            "PrintStreamTimeoutSnapshotInfo:stream_id=23, task_id=82, taskType=0 (KERNEL_AICORE), kernel_name=HcPre.",
            "PrintStreamTimeoutSnapshotInfo:stream_id=22, task_id=11, taskType=111 (EVENT_WAIT).",
            "PrintStreamTimeoutSnapshotInfo: get stream wait timeout snapshot suc",
        ]
        self.assertEqual(self.module.snapshot_targets(lines, all_types=True), {(21, 2), (23, 82), (22, 11)})

    def test_last_submission_survives_later_timeout_lookup(self):
        lines = ["AllocTaskInfo: stream_id=23 task_id=64 task_sn=1"] + ["noise"] * 50
        lines += ["StreamLaunchKernelV2: kernel_name=target", "DavidSendTask: stream_id=23 task_id=64 task_sn=2"] + [
            "noise"
        ] * 50
        lines += ["GetTaskInfo: stream_id=23 task_id=64", "PrintStreamTimeoutSnapshotInfo: stream_id=23 task_id=64"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_text("\n".join(lines), encoding="utf-8")
            result = self.module.submission_context([path], {(23, 64), (27, 711)})
        text = "\n".join(row[2] for row in result[(23, 64)])
        self.assertIn("kernel_name=target", text)
        self.assertIn("task_sn=2", text)
        self.assertNotIn("task_sn=1", text)
        self.assertNotIn((27, 711), result)


if __name__ == "__main__":
    unittest.main()
