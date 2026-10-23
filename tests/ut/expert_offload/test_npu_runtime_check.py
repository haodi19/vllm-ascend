"""基础检查脚本的进程退出与超时测试，不需要 NPU。"""

import contextlib
import io
import os
import runpy
import sys
import unittest
from pathlib import Path


class RuntimeCheckTests(unittest.TestCase):
    def test_exit_and_timeout(self):
        module = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/check_npu_runtime.py"))
        run = module["run_probe"]
        self.assertEqual(run([sys.executable, "-c", "pass"], os.environ.copy(), 10), 0)
        self.assertEqual(run([sys.executable, "-c", "raise SystemExit(3)"], os.environ.copy(), 10), 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run([sys.executable, "-c", "import time; time.sleep(30)"], os.environ.copy(), 1), 124)


if __name__ == "__main__":
    unittest.main()
