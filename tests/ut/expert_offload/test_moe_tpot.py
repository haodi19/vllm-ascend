# SPDX-License-Identifier: Apache-2.0
"""流式TPOT与同请求Host间隔反事实估算的CPU测试。"""

import contextlib
import hashlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TpotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(root / "script"), *sys.path]):
            cls.module = importlib.import_module("moe_tpot")

    def response(self, *, tokens=5, done=True, finish=True, text_chunks=("abc", "de")):
        records = [dict(choices=[dict(index=0, text="")])]
        records.extend(dict(choices=[dict(index=0, text=text)]) for text in text_chunks)
        if finish:
            records.append(dict(choices=[dict(index=0, text="", finish_reason="length")]))
        if tokens is not None:
            records.append(dict(choices=[], usage=dict(completion_tokens=tokens)))
        text = "".join("data: " + json.dumps(record) + "\n\n" for record in records)
        if done:
            text += "data: [DONE]\n\n"
        return io.BytesIO(text.encode())

    def parse(self, **kwargs):
        ticks = iter(range(1, 100))
        return self.module.decode_sse(self.response(**kwargs), 0, clock=lambda: next(ticks) * 1000000)

    def sample(self):
        # Local client first content=10ms, last content=30ms; five output tokens.
        return dict(
            request_start_epoch_us=0,
            request_end_epoch_us=40000,
            first_content_epoch_us=10000,
            last_content_epoch_us=30000,
            ttft_us=10000,
            output_latency_us=30000,
            decode_us=20000,
            http_wall_us=40000,
            tpot_us=5000,
            completion_tokens=5,
            content_chunks=3,
            wall_clock_drift_us=0,
        )

    def test_empty_prefix_and_usage_tail_not_counted_as_generated_tokens(self):
        result = self.parse()
        self.assertEqual(result["ttft_us"], 4000)
        self.assertEqual(result["output_latency_us"], 6000)
        self.assertEqual(result["decode_us"], 2000)
        self.assertEqual(result["tpot_us"], 500)
        self.assertEqual(result["content_chunks"], 2)  # MTP chunks are not tokens.
        self.assertEqual(result["digest"], hashlib.sha256(b"abcde").hexdigest())

    def test_single_output_token_is_undefined_and_one_chunk_is_visible(self):
        result = self.parse(tokens=1, text_chunks=("a",))
        self.assertIsNone(result["tpot_us"])
        result["http_wall_us"] = 20000
        summary = self.module.summarize_streams([result])
        self.assertEqual(summary["undefined_single_token"], 1)
        self.assertIsNone(summary["tpot_ms"]["raw"])
        self.assertEqual(summary["one_content_chunk"], 1)
        self.assertEqual(self.parse(text_chunks=("abcde",))["tpot_us"], 0)

    def test_missing_trailer_usage_finish_or_done_is_rejected(self):
        for options in (dict(tokens=None), dict(done=False), dict(finish=False), dict(text_chunks=())):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.parse(**options)

    def test_stream_error_timeout_and_size_limit(self):
        with self.assertRaises(ValueError):
            self.module.decode_sse(io.BytesIO(b'data: {"error":"failed"}\n\n'), 0, clock=lambda: 0)
        with self.assertRaises(TimeoutError):
            self.module.decode_sse(self.response(), 0, clock=lambda: 2000000000, timeout=1)
        with patch.object(self.module, "MAX_SSE_LINE_BYTES", 8), self.assertRaises(ValueError):
            self.module.decode_sse(self.response(), 0, clock=lambda: 0)

    def test_stream_request_uses_usage_and_separates_http_tail(self):
        opener = Mock()
        opener.open.return_value = self.response()
        with (
            patch.object(self.module, "decode_sse", return_value=self.parse()),
            patch.object(self.module.time, "time_ns", side_effect=[1000000000, 1020000000]),
            patch.object(self.module.time, "perf_counter_ns", side_effect=[0, 20000000]),
        ):
            result = self.module.streaming_completion(opener, 8155, dict(prompt="x"), timeout=300)
        body = json.loads(opener.open.call_args.args[0].data)
        self.assertTrue(body["stream"] and body["stream_options"]["include_usage"])
        self.assertEqual(result["http_wall_us"], 20000)
        self.assertEqual(result["first_content_epoch_us"], 1004000)
        self.assertEqual(result["wall_clock_drift_us"], 0)

    def test_miracle_clips_to_decode_and_unions_overlapping_callbacks(self):
        callbacks = [(-2000, -1000), (1000, 2000), (15000, 19000), (16000, 17000), (25000, 26000), (35000, 36000)]
        result = self.module.miracle_sample(self.sample(), callbacks)
        # Gaps intersecting [10ms,30ms]: [10,15] + [19,25] + [26,30].
        self.assertEqual(result["removed_host_gap_ms"], 15)
        self.assertEqual(result["miracle_tpot_ms"], 1.25)
        self.assertEqual(result["measured_tpot_ms"], 5)
        self.assertEqual(result["contributing_gaps"], 3)
        self.assertEqual(result["host_callbacks"], 5)

    def test_miracle_does_not_use_gap_threshold_or_remove_callback_body(self):
        result = self.module.miracle_sample(self.sample(), [(9000, 11000), (11001, 12000), (28000, 31000)])
        self.assertEqual(result["removed_host_gap_ms"], 16.001)  # Both 1us and 16ms gaps count.
        self.assertAlmostEqual(result["miracle_tpot_ms"], 0.99975)

    def test_miracle_rejects_unaligned_clock_and_missing_decode_callbacks(self):
        sample = self.sample()
        for callbacks in (
            [(100000, 101000), (120000, 121000)],
            [(1000, 2000), (3000, 4000)],
            [(-1, 1000), (2000, 3000)],
        ):
            with self.subTest(callbacks=callbacks), self.assertRaises(ValueError):
                self.module.miracle_sample(sample, callbacks)
        sample["wall_clock_drift_us"] = self.module.MAX_CLOCK_DRIFT_US + 1
        with self.assertRaisesRegex(ValueError, "clock drift"):
            self.module.miracle_sample(sample, [(12000, 13000), (15000, 16000)])

    def test_trace_only_reads_host_callbacks_and_checks_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            rows = [dict(ph="X", name="moe_offload/host_callback|kind=prefetch", pid=1, ts=n, dur=1) for n in (10, 20)]
            rows.append(dict(ph="X", name="moe_offload/update_weights|kind=prefetch", pid=1, ts=10, dur=1))
            path.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(self.module.read_host_intervals(path), [(10, 11), (20, 21)])
            path.write_text(json.dumps(rows)[:-1], encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.read_host_intervals(path)

    def test_unusable_profile_keeps_actual_tpot_and_does_not_fake_miracle(self):
        report = dict(stream_samples=[self.sample()], request_window=dict(stream_samples=[self.sample()]))
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            self.module.print_tpot_comparison(dict(mte_combined=report), Path(directory), ["mte_combined"])
        self.assertIn('"scope": "no_profile"', output.getvalue())
        self.assertIn('"scope": "with_profile"', output.getvalue())
        self.assertIn("MIRACLE_UNAVAILABLE", output.getvalue())

    def test_complete_profile_prints_miracle_from_same_request_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trace.json"
            rows = [
                dict(ph="X", name="moe_offload/host_callback|kind=reactive", pid=1, ts=a, dur=b - a)
                for a, b in [(9000, 11000), (11001, 12000), (28000, 31000)]
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")
            different_no_profile = dict(self.sample(), tpot_us=999)
            report = dict(
                stream_samples=[self.sample()],
                request_window=dict(stream_samples=[different_no_profile]),
                profile_trace_data=[dict(file=str(path), usable=True, json_complete=True)],
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.module.print_tpot_comparison(dict(mte_combined=report), root, ["mte_combined"])
            saved = json.loads((root / "tpot_miracle.json").read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["measured_tpot_ms"], 5)
            self.assertAlmostEqual(saved[0]["miracle_tpot_ms"], 0.99975)
            self.assertIn("MIRACLE_ESTIMATE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
