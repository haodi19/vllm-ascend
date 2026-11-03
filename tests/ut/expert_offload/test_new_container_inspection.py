# SPDX-License-Identifier: Apache-2.0
"""新容器只读检查：使用临时目录，不依赖Linux、Torch或NPU。"""

import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class NewContainerInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/inspect_new_moe_container.py"
        spec = importlib.util.spec_from_file_location("new_container_inspection", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_mounts_hide_host_sources_and_match_library_mount(self):
        text = (
            "1 0 8:1 /secret/host/root /usr/lib64 rw - ext4 /secret/device rw\n"
            "2 0 8:1 /host /home rw - ext4 /secret/device rw\n"
            "3 0 8:1 / /unrelated rw - ext4 /secret/device rw\n"
        )
        self.assertEqual(self.module.mount_paths(text), [("/usr/lib64", "ext4"), ("/home", "ext4")])

    def test_scan_finds_existing_projects_and_does_not_change_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "project"
            package = repo / "vllm_ascend"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("raise RuntimeError('must not import')", encoding="utf-8")
            (repo / "expert_rank_gsm8k.json").write_text("{}", encoding="utf-8")
            model = root / "DeepSeek-V4-Flash"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            hidden = root / "bench_results"
            hidden.mkdir()
            (hidden / "expert_rank_gsm8k.json").write_text("{}", encoding="utf-8")
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            found, _, _ = self.module.scan_paths([root], 5, 50)
            self.assertIn(("ASCEND_REPO", str(repo)), found)
            self.assertIn(("MODEL_CONFIG", str(model / "config.json")), found)
            self.assertNotIn(("HOT_EXPERTS", str(hidden / "expert_rank_gsm8k.json")), found)
            self.assertIn(("ASCEND_REPO", str(repo)), self.module.scan_paths([repo], 0, 10)[0])
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_search_limits_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "child").mkdir()
            _, count, limits = self.module.scan_paths([root], 5, 1)
            self.assertEqual(count, 1)
            self.assertIn("directory/time limit", limits)

    def test_missing_package_and_unreadable_text(self):
        with patch.object(
            self.module.importlib.metadata,
            "distribution",
            side_effect=self.module.importlib.metadata.PackageNotFoundError,
        ):
            self.assertEqual(self.module.package_info("torch"), "torch: MISSING")
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("FileNotFoundError", self.module.read_text(Path(directory) / "absent"))

    def test_command_timeout_and_url_redaction(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch.object(self.module.subprocess, "run", side_effect=subprocess.TimeoutExpired(["npu-smi", "info"], 1)),
        ):
            self.module.command("test", ["npu-smi", "info"], timeout=1)
            self.module.emit("https://user:password@example.test/path")
        self.assertIn("TIMEOUT", output.getvalue())
        self.assertNotIn("password", output.getvalue())


if __name__ == "__main__":
    unittest.main()
