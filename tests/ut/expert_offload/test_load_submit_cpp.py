"""Standalone CPU parity tests; optional extension is required, never skipped."""

import logging
import runpy
import sys
import types
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


class TestLiteralLoadMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = runpy.run_path(str(REPO / "bench_update_weights.py"))
        cls.transport = cls.bench["load_transport"](REPO / "vllm_ascend/expert_offload/h2d_transfer.py")

    def harness(self, native, packed=False, scales=True, failure=None):
        cls, _ = self.bench["load_manager"](
            REPO / "vllm_ascend/expert_offload/expert_offload_manager.py",
            None,
            self.transport,
            logging.getLogger("migration-test"),
        )
        manager = cls()
        manager.w13_expert_size_bytes = 16
        manager.w2_expert_size_bytes = 8
        sources = {
            "w13": torch.arange(128, dtype=torch.uint8).reshape(8, 16),
            "w2": torch.arange(64, dtype=torch.uint8).reshape(8, 8) + 100,
        }
        layer = types.SimpleNamespace()
        suffix = "_packed" if packed else ""
        weights = [torch.zeros((4, 16), dtype=torch.uint8), torch.zeros((4, 8), dtype=torch.uint8)]
        for name, tensor in zip(("w13_weight", "w2_weight"), weights):
            setattr(layer, name + suffix, tensor)
        if scales:
            layer.w13_weight_scale = torch.zeros((4, 2), dtype=torch.float16)
            layer.w13_weight_scale_fp32 = torch.zeros((4, 2), dtype=torch.float32)
        events = []

        def source(layer_idx, eid, name):
            events.append(("source", layer_idx, eid, name))
            if failure == "source":
                raise RuntimeError("source failure")
            size = sources[name].shape[1]
            return sources[name].untyped_storage()[eid * size : (eid + 1) * size]

        def quant(layer, layer_idx, eid, slot):
            events.append(("quant", layer_idx, eid, slot))
            if not scales:
                return []
            return [
                self.transport.H2DCopyTask(
                    torch.tensor([eid + 0.5, eid + 1.5], dtype=torch.float16),
                    layer.w13_weight_scale[slot],
                    4,
                    "scale",
                    non_blocking=False,
                )
            ]

        manager._expert_src_storage = source
        manager._build_quant_attr_h2d_tasks = quant
        transport = self.transport.TorchCopyH2DTransport()
        manager._get_h2d_transport = lambda: transport
        if native:
            self.bench["install_load_migration"](manager, transport, self.transport.H2DCopyTask)
        original_copy = transport.copy_batch

        def copy(tasks):
            events.append(("batch", [(t.name, t.nbytes, t.non_blocking) for t in tasks]))
            if failure == "copy":
                raise RuntimeError("copy failure")
            return original_copy(tasks)

        transport.copy_batch = copy
        refresh = manager._refresh_expert_fp32_scale

        def refresh_logged(layer, slot):
            events.append(("refresh", slot))
            return refresh(layer, slot)

        manager._refresh_expert_fp32_scale = refresh_logged
        return manager, layer, weights, events

    def test_bytes_scales_task_metadata_and_call_order(self):
        for packed in (False, True):
            for scales in (False, True):
                for loads in ([], [(2, 7)], [(1, 5), (3, 2)], [(1, 4), (1, 6)]):
                    with self.subTest(packed=packed, scales=scales, loads=loads):
                        pairs = [self.harness(native, packed, scales) for native in (False, True)]
                        for manager, layer, _, _ in pairs:
                            # Original list(loads) must materialize this generator once.
                            manager._load_expert_weights_into_slots(layer, 3, (pair for pair in loads))
                        self.assertEqual(pairs[0][3], pairs[1][3])
                        for python_tensor, cpp_tensor in zip(pairs[0][2], pairs[1][2]):
                            self.assertTrue(torch.equal(python_tensor, cpp_tensor))
                        if scales:
                            self.assertTrue(
                                torch.equal(pairs[0][1].w13_weight_scale_fp32, pairs[1][1].w13_weight_scale_fp32)
                            )
                            for slot, _ in loads:
                                self.assertTrue(
                                    torch.equal(
                                        pairs[1][1].w13_weight_scale_fp32[slot],
                                        pairs[1][1].w13_weight_scale[slot].float(),
                                    )
                                )

    def test_failures_preserve_partial_work_and_do_not_refresh(self):
        for failure in ("source", "copy"):
            pairs = [self.harness(native, failure=failure) for native in (False, True)]
            for manager, layer, _, events in pairs:
                with self.assertRaisesRegex(RuntimeError, failure + " failure"):
                    manager._load_expert_weights_into_slots(layer, 0, [(0, 2)])
                self.assertFalse(any(event[0] == "refresh" for event in events))
            self.assertEqual(pairs[0][3], pairs[1][3])

    def test_malformed_pair_and_generator_error_match(self):
        for bad in ((), (1,), (1, 2, 3), None):
            for native in (False, True):
                manager, layer, _, events = self.harness(native)
                with self.assertRaises((ValueError, TypeError)):
                    manager._load_expert_weights_into_slots(layer, 0, [bad])
                self.assertEqual(events, [])
        for native in (False, True):
            manager, layer, _, events = self.harness(native)

            def broken():
                yield (0, 1)
                raise RuntimeError("generator failed")

            with self.assertRaisesRegex(RuntimeError, "generator failed"):
                manager._load_expert_weights_into_slots(layer, 0, broken())
            self.assertEqual(events, [])

    def test_copy_flags_and_stop_at_failing_task(self):
        from _vllm_ascend_lrc import LoadSubmitMigration

        native = LoadSubmitMigration(None, None, torch.float32)
        traces = []
        for copy_fn in (self.transport.TorchCopyH2DTransport().copy_batch, native.copy_batch):
            calls = []

            def copy(source, non_blocking, events=calls):
                events.append((source, non_blocking))
                if source == "bad":
                    raise RuntimeError("copy failed")

            tasks = [
                types.SimpleNamespace(source=source, destination=types.SimpleNamespace(copy_=copy), non_blocking=flag)
                for source, flag in (("good", False), ("bad", True), ("unreached", False))
            ]
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                copy_fn(tasks)
            traces.append(calls)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], [("good", False), ("bad", True)])


if __name__ == "__main__":
    unittest.main()
