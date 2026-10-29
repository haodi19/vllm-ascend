# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TaskSerialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location("task_serial", scripts / "inspect_mte_task_serial.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_last_submission_uses_latest_valid_query(self):
        label, record = self.module.last_submission(
            'old LAST_TASK {"code": 0, "task_id": 74899}\nscale LAST_TASK {"code": 0, "task_id": 74901}'
        )
        self.assertEqual(label, "scale")
        self.assertEqual(record["task_id"], 74901)

    def test_query_failure_is_not_silently_replaced_with_earlier_task(self):
        for text in (
            "",
            'scale LAST_TASK {"code": 0, "task_id": 4294967295}',
            'scale LAST_TASK {"code": 1, "task_id": null}',
        ):
            with self.assertRaises(ValueError):
                self.module.last_submission(text)

    def test_reused_task_position_does_not_match_another_serial(self):
        lines = ["stream_id=27 task_id=711 task_sn=71843"] + ["noise"] * 120
        lines += ["kernel launch", "stream_id=27 task_id=711 task_sn=74901", "send task"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime.log"
            path.write_text("\n".join(lines), encoding="utf-8")
            result = self.module.serial_context(path, 74901)
            self.assertNotIn((1, lines[0]), result)
            self.assertIn((123, lines[122]), result)
            self.assertIn((124, "send task"), result)
            self.assertEqual(self.module.serial_context(path, 7490), [])


if __name__ == "__main__":
    unittest.main()
