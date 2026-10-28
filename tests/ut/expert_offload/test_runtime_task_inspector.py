# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path


class RuntimeTaskInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/inspect_mte_runtime_task.py"
        spec = importlib.util.spec_from_file_location("inspector", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_identity_uses_runtime_pid_not_parent_prefix(self):
        text = '[EngineCore pid=99] [MOE-MTE ACL_DETAIL] "[PID: 56880] stream_id=27, task_id=773"'
        self.assertEqual(self.module.failure_identity(text), (56880, 27, 773))

    def test_missing_detail_is_explicit_error(self):
        with self.assertRaises(ValueError):
            self.module.failure_identity("HTTP 500")

    def test_exact_pass_excludes_fallback_and_accepts_camelcase(self):
        self.assertEqual(self.module.select_lines(["KERNEL_AIVEC"], 27, 773, fallback=False), [])
        line = "launch streamId=27 taskId=773 kernel=example"
        self.assertEqual(self.module.select_lines([line], 27, 773, fallback=False), [(1, line)])

    def test_exact_numbers_do_not_match_longer_task_id(self):
        lines = ["stream_id=27 task_id=7730"] + ["noise"] * 5 + ["stream_id=27 task_id=773"]
        selected = self.module.select_lines(lines, 27, 773)
        self.assertIn((7, lines[-1]), selected)
        self.assertNotIn((1, lines[0]), selected)


if __name__ == "__main__":
    unittest.main()
