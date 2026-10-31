# SPDX-License-Identifier: Apache-2.0
"""Capacity guards are independent of local test-machine free space."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ProfileStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/run_verified_cpp_tests.py"
        spec = importlib.util.spec_from_file_location("profile_storage_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_missing_target_uses_existing_parent_without_creating_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with (
                patch.object(
                    self.module.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * 1024**3)
                ) as usage,
                patch.object(
                    self.module.os, "statvfs", return_value=SimpleNamespace(f_files=100, f_favail=90), create=True
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.module.ensure_profile_storage(root / "profile/case/cpp", stage="test")
            usage.assert_called_once_with(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_low_bytes_or_inodes_rejected(self):
        for free, inodes in ((0.14, 90), (30, 0)):
            with (
                self.subTest(free=free, inodes=inodes),
                patch.object(self.module.shutil, "disk_usage", return_value=SimpleNamespace(free=free * 1024**3)),
                patch.object(
                    self.module.os, "statvfs", return_value=SimpleNamespace(f_files=100, f_favail=inodes), create=True
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "Insufficient profile storage"),
            ):
                self.module.ensure_profile_storage(Path.cwd(), stage="test")


if __name__ == "__main__":
    unittest.main()
