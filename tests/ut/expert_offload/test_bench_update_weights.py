"""Run directly with python -B; no vLLM installation or NPU required."""

import contextlib
import io
import logging
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


class TestUpdateWeightsBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = runpy.run_path(str(REPO / "bench_update_weights.py"))

    def test_subtraction_uses_same_call_and_keeps_submission(self):
        # Total=100, observe=10, choose=20, submit=10, sync=30.
        ticks = iter((0, 1, 11, 12, 32, 33, 43, 44, 74, 100))
        timer = self.bench["Timings"](lambda: next(ticks))
        functions = [
            timer.wrap(name, lambda: None) for name in ("observe", "choose_victims", "load_submit", "synchronize")
        ]

        def call(_):
            for fn in functions:
                fn()

        result = timer.run(call, None)
        self.assertEqual(result["total"], 100)
        self.assertEqual(result["outer"], 40)
        self.assertEqual(result["load_submit"], 10)

    def test_timer_resets_missing_observe_for_prefetch(self):
        ticks = iter(range(20))
        timer = self.bench["Timings"](lambda: next(ticks))
        timer.run(timer.wrap("observe", lambda _: None), None)
        result = timer.run(lambda _: None, None)
        self.assertEqual(result["observe"], 0)
        self.assertEqual(result["outer"], result["total"])

    def test_planning_scenarios_and_input_validation(self):
        for flags in (
            [],
            ["--scenario", "hit"],
            ["--scenario", "miss"],
            ["--tokens", "4"],
            ["--prefetch"],
            ["--no-router-scores"],
        ):
            with self.subTest(flags=flags):
                output = io.StringIO()
                with (
                    patch.object(sys, "argv", ["bench", "--warmup", "1", "--iterations", "3", *flags]),
                    contextlib.redirect_stdout(output),
                ):
                    self.bench["main"]()
                self.assertIn("Validation passed", output.getvalue())
        with (
            patch.object(sys, "argv", ["bench", "--tokens", "5"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.bench["main"]()

    def test_original_load_helpers_copy_correct_expert_to_slot(self):
        base = REPO / "vllm_ascend/expert_offload"
        transport = self.bench["load_transport"](base / "h2d_transfer.py")
        cls, _ = self.bench["load_manager"](
            base / "expert_offload_manager.py", None, transport, logging.getLogger("test")
        )
        manager = cls()
        manager.w13_expert_size_bytes = manager.w2_expert_size_bytes = 16
        host = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
        manager._expert_src_storage = lambda layer_idx, eid, name: (host.untyped_storage()[eid * 16 : (eid + 1) * 16])
        manager._build_quant_attr_h2d_tasks = lambda *args: []
        manager._get_h2d_transport = transport.TorchCopyH2DTransport
        layer = types.SimpleNamespace(
            w13_weight=torch.zeros((2, 16), dtype=torch.uint8), w2_weight=torch.zeros((2, 16), dtype=torch.uint8)
        )
        manager._load_expert_weights_into_slots(layer, 0, [(0, 3), (1, 2)])
        for name in ("w13_weight", "w2_weight"):
            self.assertTrue(torch.equal(getattr(layer, name)[0], host[3]))
            self.assertTrue(torch.equal(getattr(layer, name)[1], host[2]))


if __name__ == "__main__":
    unittest.main()
