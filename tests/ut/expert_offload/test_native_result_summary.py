# SPDX-License-Identifier: Apache-2.0
"""Saved results remain useful after failed/repeated separately launched jobs."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import reanalyze_native_prefetch_overlap as overlap
import summarize_native_moe_results as summary
from inspect_moe_profile import _stats
from moe_outlier_stats import outlier_summary
from native_moe_artifacts import saved_timing_records
from reanalyze_native_moe_profiles import saved_case
from test_native_prefetch_overlap import capture


def timing(root, case, tokens=5):
    directory = root / case
    directory.mkdir(parents=True, exist_ok=True)
    sample = dict(
        digest="same",
        usage=dict(completion_tokens=tokens),
        completion_tokens=tokens,
        content_chunks=3,
        decode_us=2000,
        ttft_us=3000,
        output_latency_us=5000,
        tpot_us=2000 / (tokens - 1),
        http_wall_us=5100,
    )
    row = dict(case=case, physical_npu=6, requests=2, output_match=True, stream_samples=[sample, sample])
    (directory / "timing_result.json").write_text(json.dumps(row), encoding="utf-8")
    return row


def profile(root, case):
    trace = capture(root, case)
    timing(root, case)
    gap = dict(direction="prefetch->reactive", outliers_us=outlier_summary([12, 15]), **_stats([12, 15]))
    report = dict(
        mapping=json.loads(trace.with_name("clock_alignment.json").read_text()),
        aligned_profile=dict(status="exported"),
        scan=dict(host_gaps=[gap], loading_track=dict(name="Stream 54", loaded_batches=2)),
        bubble_stats=[dict(scope="full_capture", kept=dict(n=3, mean=123.45))],
    )
    (root / case / "profile_report.json").write_text(json.dumps(report), encoding="utf-8")


class NativeResultSummaryTests(unittest.TestCase):
    def test_profile_only_device_comes_from_local_capture_not_copied_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile(root, "native")
            (root / "summary.txt").write_text("PROFILE_ONLY: physical_npu=7\n")
            self.assertEqual(saved_case(root, "native")[1], 7)
            summary.rebuild(root)
            self.assertEqual(saved_case(root, "native")[1], 7)

    def test_failed_profile_does_not_discard_complete_timing_or_append_failure_stack(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = timing(root, "python")
            (root / "native").mkdir()
            (root / "native/native_partial.json").write_text('{"samples":[]}')
            raw = "Traceback: old failure\nRESULT " + json.dumps(expected) + "\nFAILED profile\n" * 3
            (root / "summary.txt").write_text(raw)
            result = summary.rebuild(root)
            self.assertNotIn("Traceback", result)
            self.assertNotIn("RESULT ", result)
            self.assertNotIn("FAILED profile", result)
            self.assertIn("| python | 原始 | 2 | [5] | 2/2.000", result)
            self.assertIn("| native | 缺失/未完成 |", result)
            self.assertNotIn("| native | 原始", result)
            self.assertIn(raw, (root / "run_details.log").read_text())
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(summary.rebuild(root), result)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_offline_summary_retains_host_bubbles_and_both_restats_without_parsing_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case in summary.CASES:
                profile(root, case)
            overlap.reanalyze([root], root, lambda message: None)
            source = {p: p.read_bytes() for p in root.rglob("*.json")}
            with (
                patch("subprocess.Popen", side_effect=AssertionError("no subprocess")),
                patch("analyze_prefetch_compute_overlap.analyze", side_effect=AssertionError("no large trace")),
            ):
                result = summary.rebuild(root)
            for value in (
                "Host间隔",
                "Stream 54",
                "123.450",
                "host_callback",
                "update_weights",
                "计算晚完成",
                "方案1",
                "方案2",
                "保守",
                "直接",
            ):
                self.assertIn(value, result)
            for case in summary.CASES:
                self.assertIn(f"| {case} |", result)
                self.assertIn("方案2", (root / case / "summary.txt").read_text(encoding="utf-8"))
            self.assertEqual(source, {p: p.read_bytes() for p in source})

    def test_legacy_record_still_loads_after_summary_is_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = timing(root, "python")
            (root / "python/timing_result.json").unlink()
            (root / "summary.txt").write_text("RUNTIME_DEVICE: physical_npu=6\nRESULT " + json.dumps(row))
            summary.rebuild(root)
            self.assertEqual(saved_timing_records(root), dict(python=row))
            self.assertIn("| python | 原始", summary.rebuild(root))

    def test_bad_structured_result_never_falls_back_to_successful_old_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = timing(root, "native")
            timing(root, "python")
            (root / "summary.txt").write_text("RESULT " + json.dumps(row))
            (root / "native/timing_result.json").write_text("invalid JSON")
            result = summary.rebuild(root)
            self.assertIn("| python | 原始", result)
            self.assertNotIn("| native | 原始", result)
            self.assertIn("| native | 缺失/未完成", result)

    def test_failed_order_is_explicit_when_conditional_estimate_is_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile(root, "native")
            overlap.reanalyze([root], root, lambda message: None)
            path = root / "prefetch_overlap.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["native"]["coordinate_estimate"] = dict(data["native"])
            path.write_text(json.dumps(data))
            result = summary.rebuild(root)
            self.assertIn("条件坐标估算", result)
            self.assertIn("顺序检查未通过", result)

    def test_two_token_suites_are_printed_once_and_missing_case_stays_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for tokens in (5, 128):
                timing(root / f"token{tokens}", "python", tokens)
            (root / "status.txt").write_text("max_tokens=5; exit=1; case=native\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(summary.main([str(root)]), 0)
            text = out.getvalue()
            self.assertEqual(text.count("===== token5 ====="), 1)
            self.assertEqual(text.count("===== token128 ====="), 1)
            self.assertIn("BATCH_EXIT=1", text)
            self.assertIn("SUMMARY_FILE=", text)


if __name__ == "__main__":
    unittest.main()
