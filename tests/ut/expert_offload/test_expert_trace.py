# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of routing provenance, partial failure and request windows."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE = load_module("expert_trace_test", "vllm_ascend/expert_offload/expert_trace.py")
REPORT = load_module("expert_trace_report_test", "script/moe_expert_trace_report.py")


class ExpertTraceTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.transport = SimpleNamespace(copy_batch=lambda tasks: None)
        self.manager = SimpleNamespace(
            moe_layers=[object(), object()],
            num_total_experts=4,
            topk=2,
            num_device_experts_for_layer=lambda layer: 2,
            num_device_layers=2,
            prefetch_topk=1,
            offload_config=SimpleNamespace(config={"cache_policy_backend": "cpp"}),
            _cpu_tensor_storage_bytes=lambda values: 100,
            _host_memory_snapshot=lambda: (1, 2, 3),
            w13_weights_cpu=[],
            w2_weights_cpu=[],
            scale_cpu_buffers=[],
            offset_cpu_buffers=[],
            scale_bias_cpu_buffers=[],
            _synchronize_h2d=lambda: None,
            _get_h2d_transport=lambda: self.transport,
        )
        m = self.manager

        def load(layer, layer_idx, loads):
            self.transport.copy_batch(
                [SimpleNamespace(name=f"w13[L{layer_idx},E{eid}->S{slot}]", nbytes=64) for slot, eid in loads]
            )

        def update(args):
            # A predicted set can exceed the prefetch transfer cap.
            mapping = args[1]
            needed = set(args[0].reshape(-1))
            missing = sorted(e for e in needed if mapping[e] < 0)
            selected = missing[:1] if args[5] else missing
            loads = []
            for eid in selected:
                victim = next(e for e, slot in enumerate(mapping) if slot >= 0 and e not in needed)
                slot = mapping[victim]
                mapping[victim] = -1
                mapping[eid] = slot
                loads.append((slot, eid))
            m._load_expert_weights_into_slots(None, args[3], loads)
            m._synchronize_h2d()

        def prefill(layer, mapping):
            self.transport.copy_batch([SimpleNamespace(name=f"prefill:E{eid}", nbytes=64) for eid in range(4)])
            m._synchronize_h2d()

        m._load_expert_weights_into_slots = load
        m._update_weights = update
        m._prefill_load_layer = prefill
        self.trace = TRACE.install_expert_trace(m)

    def args(self, routes, mapping, prefetch=False, layer=1):
        return (np.array(routes), mapping, None, layer, None, prefetch)

    def test_capture_bound_method_and_request_reset(self):
        captured = self.manager._update_weights
        captured(self.args([[0, 1]], np.array([0, 1, -1, -1])))
        self.assertEqual(self.trace.records, [])
        for _ in range(2):
            self.trace.start()
            captured(self.args([[0, 1]], np.array([0, 1, -1, -1])))
            report = self.trace.stop()
            self.assertTrue(report["complete"])
            self.assertEqual(report["records"][0]["layer_visit"], 1)

    def test_hits_missing_prefetch_cap_evictions_and_pair(self):
        self.trace.start()
        mapping = np.array([0, 1, -1, -1])
        self.manager._update_weights(self.args([[2, 3], [2, 3]], mapping, prefetch=True))
        self.manager._update_weights(self.args([[1, 2]], mapping))
        report = self.trace.stop()
        pre, reactive = report["records"]
        self.assertEqual(pre["missing_ids"], [2, 3])
        self.assertEqual(pre["loaded_ids"], [2])
        self.assertEqual(pre["missing_not_loaded"], [3])
        self.assertEqual(pre["evicted_ids"], [0])
        self.assertEqual(pre["logical_h2d_bytes"], 64)
        self.assertEqual(pre["route_rows"], [[2, 3], [2, 3]])
        self.assertEqual(reactive["hit_ids"], [1, 2])
        self.assertEqual(reactive["preceding_prefetch_event"], pre["event_id"])
        self.assertEqual(reactive["prefetched_ids_used"], [2])
        self.assertEqual(reactive["logical_h2d_bytes"], 0)

    def test_failed_copy_not_reported_completed(self):
        self.trace.start()

        def broken_wait():
            raise RuntimeError("stream timeout")

        self.manager._synchronize_h2d = self.trace.wrap_wait(broken_wait)
        with self.assertRaisesRegex(RuntimeError, "stream timeout"):
            self.manager._update_weights(self.args([[1, 2]], np.array([0, 1, -1, -1])))
        report = self.trace.stop()
        self.assertFalse(report["complete"])
        record = report["records"][0]
        self.assertTrue(record["submit_returned"])
        self.assertFalse(record["h2d_wait_returned"])
        self.assertEqual(record["status"], "failed")

    def test_missing_wait_is_incomplete(self):
        self.trace.start()
        self.manager._synchronize_h2d = lambda: None
        self.manager._update_weights(self.args([[0, 1]], np.array([0, 1, -1, -1])))
        self.assertFalse(self.trace.stop()["complete"])

    def test_prefill_separate_from_decode(self):
        self.trace.start()
        self.manager._prefill_load_layer(1, None)
        report = self.trace.stop()
        self.assertEqual(report["records"][0]["kind"], "prefill_pool")
        self.assertEqual(report["records"][0]["logical_h2d_bytes"], 256)
        self.assertEqual(report["records"][0]["full_pool_experts"], 4)

    def test_report_keeps_ids_counts_and_checks_round(self):
        self.trace.start()
        for layer in range(2):
            self.manager._update_weights(self.args([[1, 2]], np.array([0, 1, -1, -1]), layer=layer))
        report = self.trace.stop()
        report["request"] = {"index": 1}
        self.assertTrue(REPORT.layer_rounds(report)[0]["complete_layer_sweep"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.txt"
            REPORT.write_trace(report, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn('"loaded_experts": 2', text)
            self.assertIn('"hit_ids": [1]', text)
            self.assertIn('"h2d_bytes": 128', text)
        report["records"].reverse()
        self.assertFalse(REPORT.layer_rounds(report)[0]["complete_layer_sweep"])

    def test_overlapping_windows_rejected(self):
        self.trace.start()
        with self.assertRaises(ValueError):
            self.trace.start()
        self.trace.stop()

    def test_empty_or_truncated_trace_not_complete(self):
        from unittest.mock import patch

        self.trace.start()
        self.assertFalse(self.trace.stop()["complete"])
        self.trace.start()
        with patch.object(TRACE, "MAX_RECORDS", 1):
            for _ in range(2):
                self.manager._update_weights(self.args([[0, 1]], np.array([0, 1, -1, -1])))
        report = self.trace.stop()
        self.assertFalse(report["complete"])
        self.assertEqual(report["dropped"], 1)

    def test_control_endpoint_round_trip(self):
        import json

        timing = load_module("trace_timing_http_test", "vllm_ascend/expert_offload/callback_timing.py")
        collector = timing.CallbackTimings("cpp")
        collector.serve(0, self.trace)
        self.addCleanup(collector.close)
        port = collector.server.server_address[1]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def post(path):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"{}")
            with opener.open(req, timeout=2) as response:
                return json.load(response)

        self.assertTrue(post("/expert_trace/start")["started"])
        self.manager._update_weights(self.args([[0, 1]], np.array([0, 1, -1, -1])))
        self.assertTrue(post("/expert_trace/stop")["complete"])

    def test_diagnostic_entrypoint_combines_existing_outputs(self):
        from unittest.mock import patch

        sys.path.insert(0, str(ROOT / "script"))
        self.addCleanup(sys.path.remove, str(ROOT / "script"))
        import run_moe_expert_trace as entry

        def fake_run(args, root):
            self.assertTrue(args.expert_trace)
            self.assertTrue(args.skip_profile)
            self.assertTrue(args.no_load_diagnostics)
            self.assertTrue(args.diagnose_prefetch_wait_at_join)
            self.assertEqual((args.worker_scan, args.worker_probe), ("allocated", "atomic"))
            directory = root / "mte"
            directory.mkdir()
            (directory / "expert_trace_001.txt").write_text("COMPLETE true\nEVENT test\n", encoding="utf-8")
            (directory / "inference_cpp.log").write_text("Actual usage: weights=33 GiB\n", encoding="utf-8")
            print("Effective serve command: example")
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with (
                patch.object(sys, "platform", "linux"),
                patch.object(sys, "argv", ["run_moe_expert_trace.py", "--npu", "0", "--output-dir", str(root)]),
                patch.object(entry, "run", side_effect=fake_run),
            ):
                self.assertEqual(entry.main(), 0)
            text = (root / "expert_trace.txt").read_text(encoding="utf-8")
            self.assertIn("EVENT test", text)
            self.assertIn("weights=33 GiB", text)
            self.assertIn("Effective serve command: example", text)

    def test_runner_preset_import_and_parser(self):
        sys.path.insert(0, str(ROOT / "script"))
        self.addCleanup(sys.path.remove, str(ROOT / "script"))
        from unittest.mock import patch

        import run_moe_host_sync as runner

        with patch.object(sys, "platform", "linux"):
            args = runner.parse_args(["--npu", "0", "--backend", "mte", "--skip-profile", "--expert-trace"])
            self.assertTrue(args.expert_trace)
            with self.assertRaises(SystemExit):
                runner.parse_args(["--npu", "0", "--expert-trace"])


if __name__ == "__main__":
    unittest.main()
