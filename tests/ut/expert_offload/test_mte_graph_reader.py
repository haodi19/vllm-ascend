# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class GraphReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location("graph_reader", scripts / "inspect_mte_graph_dump.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def node(self, model, stream, task, name):
        return {
            "name": name,
            "args": {"Model Id": model, "Stream Id": stream, "Task Id": task, "Task Type": "KERNEL_AIVEC"},
        }

    def test_event_candidates_do_not_cross_models_or_hide_reuse(self):
        wait = self.node(63, 21, 0, "EVENT_WAIT_8")
        first = self.node(63, 23, 1, "EVENT_RECORD_8")
        second = self.node(63, 23, 8, "EVENT_RECORD_8")
        other = self.node(64, 23, 1, "EVENT_RECORD_8")
        self.assertEqual(self.module.event_candidates([wait, first, second, other], wait), [first, second])

    def test_reader_identifies_wait_and_missing_external_load_task(self):
        wait = self.node(63, 21, 2, "host_sync_wait_2")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.module.report_graph([wait, {"ph": "M"}], {(21, 2), (27, 711)})
        self.assertIn("host_sync_wait_2", out.getvalue())
        self.assertIn("NOT_IN_THIS_GRAPH", out.getvalue())


if __name__ == "__main__":
    unittest.main()
