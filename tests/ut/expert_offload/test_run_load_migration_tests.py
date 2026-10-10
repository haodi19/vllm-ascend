"""Console runner orchestration tests; no NPU, compiler or child execution."""

import contextlib
import io
import os
import runpy
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]


class TestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = runpy.run_path(str(REPO / "script/run_load_migration_tests.py"))

    def test_sequence_and_fixed_comparison(self):
        steps = self.runner["test_steps"]("python", False)
        self.assertEqual(len(steps), 10)
        self.assertIn("pip", steps[0][1])
        self.assertIn("test_load_submit_cpp.py", steps[1][1][-1])
        self.assertEqual(
            [cmd[cmd.index("--load-backend") + 1] for _, cmd in steps[4:8]], ["python", "cpp", "cpp", "python"]
        )
        for _, cmd in steps[2:]:
            self.assertEqual(cmd[cmd.index("--backend") + 1], "cpp")
            self.assertEqual(cmd[cmd.index("--h2d-backend") + 1], "torch")
        self.assertEqual(self.runner["test_steps"]("python", True), steps[1:])

    def test_child_visibility_and_completion(self):
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "7"}),
            patch("subprocess.run", return_value=types.SimpleNamespace(returncode=0)) as run,
            contextlib.redirect_stdout(output),
        ):
            result = self.runner["main"](["--npu", "3", "--skip-build"])
            self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "7")
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 9)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "3")
            self.assertEqual(call.kwargs["cwd"], REPO)
            self.assertNotIn("stdout", call.kwargs)
            self.assertNotIn("shell", call.kwargs)
        self.assertIn("ALL STEPS PASSED", output.getvalue())

    def test_failure_stops_remaining_steps(self):
        output = io.StringIO()
        with (
            patch(
                "subprocess.run", side_effect=[types.SimpleNamespace(returncode=0), types.SimpleNamespace(returncode=2)]
            ) as run,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(self.runner["main"]([]), 2)
        self.assertEqual(run.call_count, 2)
        self.assertIn("FAILED step 2", output.getvalue())
        self.assertNotIn("ALL STEPS PASSED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
