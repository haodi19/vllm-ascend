"""空间不足、原始采集缺失和轮询超时的回归检查。"""

import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[3] / "script"


class ProfileStorageTests(unittest.TestCase):
    def test_probe_cleanup_orders_sync_before_memory_release(self):
        module = runpy.run_path(str(SCRIPT / "probe_mte_quant_copy.py"))
        calls = Mock()
        module["cleanup_session"](calls.compute, calls.transfer, calls.graph, calls.session)
        self.assertEqual(
            [str(call) for call in calls.mock_calls],
            [
                "call.compute.synchronize()",
                "call.transfer.synchronize()",
                "call.graph.reset()",
                "call.session.end_device_access(stream_synchronized=True)",
                "call.session.stop()",
                "call.session.close()",
            ],
        )
        calls.reset_mock()
        calls.compute.synchronize.side_effect = RuntimeError("device busy")
        with self.assertRaises(RuntimeError):
            module["cleanup_session"](calls.compute, calls.transfer, calls.graph, calls.session)
        calls.session.end_device_access.assert_not_called()

    def test_missing_target_checks_existing_parent_and_rejects_full_disk(self):
        module = runpy.run_path(str(SCRIPT / "profile_storage.py"))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(module["shutil"], "disk_usage", return_value=SimpleNamespace(free=0)) as usage,
        ):
            with self.assertRaisesRegex(RuntimeError, "磁盘空间不足"):
                module["require_space"](Path(directory) / "not_created/child")
            usage.assert_called_once_with(Path(directory).resolve())
            self.assertFalse((Path(directory) / "not_created").exists())

    def test_enough_space(self):
        module = runpy.run_path(str(SCRIPT / "profile_storage.py"))
        with patch.object(module["shutil"], "disk_usage", return_value=SimpleNamespace(free=25 << 30)):
            module["require_space"](SCRIPT)

    def test_missing_device_inputs_rejected_before_reparse(self):
        module = runpy.run_path(str(SCRIPT / "profile_storage.py"))
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory)
            with self.assertRaisesRegex(ValueError, "需重新采集"):
                module["require_raw_data"](worker)
            device = worker / "PROF_1/device_0"
            (device / "data").mkdir(parents=True)
            (device / "start_info.0").write_bytes(b"start")
            with self.assertRaises(ValueError):
                module["require_raw_data"](worker)
            (device / "data/task").write_bytes(b"task")
            module["require_raw_data"](worker)

    def test_copy_after_wait_timeout_is_not_success(self):
        module = runpy.run_path(str(SCRIPT / "probe_mte_quant_copy.py"))
        module["check_state"]([2, 1, 2, 55], 2)
        for state in ([2, 3, 0, 1000000], [2, 2, 0, 44], [1, 1, 1, 44]):
            with self.assertRaises(RuntimeError):
                module["check_state"](state, 2)


if __name__ == "__main__":
    unittest.main()
