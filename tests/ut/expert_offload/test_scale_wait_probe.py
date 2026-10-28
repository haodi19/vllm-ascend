# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call


class ScaleWaitProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/probe_scale_wait_dependency.py"
        spec = importlib.util.spec_from_file_location("scale_wait_probe", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_fork_join_orders_ready_before_launch_and_done_after_launch(self):
        operations = Mock()
        npu = SimpleNamespace(Event=Mock(side_effect=["ready", "done"]), stream=lambda _: nullcontext())
        events = self.module.capture_exchange(npu, operations.compute, operations.handoff, operations.launch, True)
        self.assertEqual(events, ("ready", "done"))
        self.assertEqual(
            operations.mock_calls,
            [
                call.compute.record_event("ready"),
                call.handoff.wait_event("ready"),
                call.launch(),
                call.handoff.record_event("done"),
                call.compute.wait_event("done"),
            ],
        )

    def test_independent_baseline_has_no_added_events(self):
        npu, compute, handoff, launch = Mock(), Mock(), Mock(), Mock()
        self.assertEqual(self.module.capture_exchange(npu, compute, handoff, launch, False), ())
        launch.assert_called_once_with()
        self.assertEqual(npu.mock_calls + compute.mock_calls + handoff.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
