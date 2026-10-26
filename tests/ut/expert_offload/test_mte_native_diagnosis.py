"""原生栈诊断的输出筛选与设备时间分解测试。"""

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
NATIVE = runpy.run_path(str(ROOT / "script/diagnose_mte_tensor_stall.py"))
TIMELINE = runpy.run_path(str(ROOT / "script/diagnose_moe_transfer_gaps.py"))


def event(name, tid, start, end):
    return dict(name=name, tid=tid, start=start, end=end, dur=end - start)


class NativeDiagnosisTests(unittest.TestCase):
    def test_worker_first_and_idle_threads_omitted(self):
        text = (
            "Thread 1 (LWP 100):\n#0 futex_wait\n"
            "Thread 2 (LWP 101):\n#0 aclrtSynchronizeStream\n"
            "Thread 3 (LWP 102):\n#0 custom_worker\n"
        )
        result = NATIVE["selected_stacks"](text, 102)
        self.assertTrue(result.startswith("Thread 3"))
        self.assertNotIn("Thread 1", result)
        self.assertIn("Thread 2", result)

    def test_attach_only_descendants(self):
        def status(path):
            return {"/proc/30/status": "PPid:\t20", "/proc/20/status": "PPid:\t10"}[path.as_posix()]

        with patch.object(Path, "read_text", status):
            self.assertTrue(NATIVE["descendant"](30, 10))
            self.assertFalse(NATIVE["descendant"](1, 10))
        with patch.object(Path, "read_text", side_effect=OSError):
            self.assertFalse(NATIVE["descendant"](99, 10))

    def test_device_only_work_union_excludes_waits_and_memcpy(self):
        events = [
            event("host_sync_publish", "p", 0, 2),
            event("host_sync_wait", "p", 2, 100),
            event("host_sync_publish", "m", 150, 152),
            event("GMM", "m", 110, 140),
            event("Quant", "m", 130, 145),
            event("GMM", "m", 50, 70),
            event("DAVID_EVENT_WAIT", "m", 100, 150),
            event("MEMCPY_ASYNC", "m", 100, 110),
        ]
        row = TIMELINE["device_pipeline"](events)[0]
        self.assertEqual(row["wait_to_publish_us"], 50)
        self.assertEqual(row["work_after_wait_us"], 35)
        self.assertEqual(row["other_after_wait_us"], 15)
        self.assertEqual(row["work_during_wait_us"], 20)

    def test_ambiguous_waits_not_assigned(self):
        events = [
            event("host_sync_publish", "p", 0, 2),
            event("host_sync_wait", "p", 2, 100),
            event("host_sync_wait", "p", 100, 120),
            event("host_sync_publish", "m", 150, 152),
        ]
        self.assertFalse(TIMELINE["device_pipeline"](events))


if __name__ == "__main__":
    unittest.main()
