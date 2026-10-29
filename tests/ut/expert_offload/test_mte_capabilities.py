# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/inspect_mte_diagnostic_capabilities.py"
        spec = importlib.util.spec_from_file_location("capabilities", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_declarations_ignore_comments_and_keep_multiline_parameters(self):
        text = """// aclrtFakeDump(x);
        /* rtFakeDebug(x); */
        aclError aclrtExampleDump(void* graph,
                                 const char* path);
        aclError aclrtSetDevice(int device);
        """
        self.assertEqual(self.module.declarations(text), ["aclrtExampleDump(void* graph, const char* path);"])

    def test_fixture_probe_reads_headers_without_running_device_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            include = root / "include/acl"
            include.mkdir(parents=True)
            (include / "acl_rt.h").write_text("aclError aclrtExampleDump(void* graph);", encoding="utf-8")
            output = io.StringIO()
            with (
                patch.object(self.module.shutil, "which", return_value=None),
                patch.object(self.module.subprocess, "run") as run,
                contextlib.redirect_stdout(output),
            ):
                self.module.inspect(root)
            run.assert_not_called()
            self.assertIn("API", output.getvalue())
            self.assertIn("aclrtExampleDump", output.getvalue())
            self.assertEqual(list(root.rglob("*.h")), [include / "acl_rt.h"])


if __name__ == "__main__":
    unittest.main()
