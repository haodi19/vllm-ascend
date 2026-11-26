# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import inspect_native_capture_consistency as audit
from test_native_prefetch_overlap import capture


def host(start, end, call):
    return dict(start=start, end=end, context=dict(call=str(call), layer=str(call)))


def pair(start, end):
    return dict(start=start, end=start + 1), dict(start=start + 1, end=end)


class NativeCaptureConsistencyTests(unittest.TestCase):
    def test_prefix_and_suffix_are_reported_without_changing_pairing(self):
        pairs = [pair(-30, -20), pair(8, 22), pair(28, 42), pair(50, 60)]
        hosts = [host(10, 20, 1), host(30, 40, 2)]
        original = copy.deepcopy((pairs, hosts))
        result = audit.inspect_window(pairs, hosts, 2)
        self.assertEqual(result["candidate_count"], 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["offset"], 1)
        self.assertEqual(candidate["unmatched_prefix"], 1)
        self.assertEqual(candidate["unmatched_suffix"], 1)
        self.assertEqual(candidate["prefix_clearly_before_host_window"], 1)
        self.assertEqual(candidate["suffix_clearly_after_host_window"], 1)
        self.assertEqual(candidate["unmatched_not_clearly_outside"], 0)
        self.assertEqual((pairs, hosts), original)

    def test_interior_extra_pair_cannot_be_removed_to_force_a_match(self):
        result = audit.inspect_window([pair(8, 22), pair(24, 26), pair(28, 42)], [host(10, 20, 1), host(30, 40, 2)], 2)
        self.assertEqual(result["candidate_count"], 0)

    def test_clock_residual_keeps_strict_failure_and_nearby_extra_task(self):
        result = audit.inspect_window([pair(8, 18.5), pair(21, 24)], [host(10, 20, 1)], 2)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["strict_order_failures"], 1)
        self.assertEqual(candidate["max_order_residual_us"], 1.5)
        self.assertEqual(candidate["suffix_clearly_after_host_window"], 0)
        self.assertEqual(candidate["unmatched_not_clearly_outside"], 1)

    def test_large_residual_is_not_explained_by_changing_clock_offset(self):
        result = audit.inspect_window([pair(8, 15)], [host(10, 20, 1)], 2)
        self.assertEqual(result["candidate_count"], 0)

    def test_ambiguous_candidates_are_retained(self):
        result = audit.inspect_window([pair(0, 100), pair(1, 101)], [host(10, 20, 1)], 2)
        self.assertEqual(result["candidate_count"], 2)

    def test_fewer_pairs_and_search_limit_are_explicit(self):
        hosts = [host(10, 20, 1), host(30, 40, 2)]
        self.assertEqual(
            audit.inspect_window([pair(8, 22)], hosts, 2)["status"], "fewer_device_pairs_than_host_markers"
        )
        result = audit.inspect_window([pair(0, 100)] * (audit.MAX_WINDOW_SHIFTS + 1), hosts[:1], 2)
        self.assertEqual(result["status"], "too_many_candidate_windows")

    def test_command_diff_keeps_workload_changes_but_labels_service_options(self):
        command = [
            "python",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "/weights/model",
            "--port",
            "8156",
            "--additional-config",
            json.dumps(dict(expert_offload_config=dict(host_update_backend="python", cache_size=24))),
            "--speculative-config",
            '{"num_speculative_tokens": 2}',
            "--enforce-eager",
        ]
        left = audit.command_settings(command)
        changed = list(command)
        changed[6] = "8157"
        changed[8] = json.dumps(dict(expert_offload_config=dict(host_update_backend="native", cache_size=32)))
        changed[10] = '{"num_speculative_tokens": 3}'
        differences = {row["key"]: row for row in audit.command_differences(left, audit.command_settings(changed))}
        self.assertEqual(differences["--port"]["category"], "service_or_profiler")
        self.assertEqual(
            differences["--additional-config.expert_offload_config.host_update_backend"]["category"],
            "expected_case_selector",
        )
        self.assertEqual(
            differences["--additional-config.expert_offload_config.cache_size"]["category"], "workload_or_other"
        )
        self.assertEqual(differences["--speculative-config.num_speculative_tokens"]["category"], "workload_or_other")

    def test_duplicate_command_option_is_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.command_settings(["python", "--seed=1", "--seed", "2"])

    def test_complete_saved_capture_audit_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture(root, "native")
            (root / "native_command.json").write_text(json.dumps(["python", "serve", "model"]), encoding="utf-8")
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                complete = audit.inspect_run(root, ["native"])
            self.assertTrue(complete)
            self.assertIn("WINDOW_HYPOTHESIS", output.getvalue())
            self.assertIn('"matches_timing": true', output.getvalue())
            self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})

    def test_saved_counter_mismatch_and_missing_trace_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = capture(root, "native")
            counters_path = root / "native_profile_counters.json"
            counters = json.loads(counters_path.read_text(encoding="utf-8"))
            counters["count"] += 1
            counters_path.write_text(json.dumps(counters), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "counters disagree"):
                audit.inspect_capture(root, "native")
            path.unlink()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                audit.inspect_capture(root, "native")

    def test_profile_selection_does_not_hide_python_command_differences(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case in ("python", "native"):
                (root / f"{case}_timing_result.json").write_text(
                    json.dumps(dict(case=case, stream_samples=[dict(digest=case, usage=dict(completion_tokens=128))])),
                    encoding="utf-8",
                )
                command = ["python", "serve", "model", "--seed", "1" if case == "python" else "2"]
                (root / f"{case}_command.json").write_text(json.dumps(command), encoding="utf-8")
            output = io.StringIO()
            with patch.object(audit, "inspect_capture") as scan, contextlib.redirect_stdout(output):
                self.assertTrue(audit.inspect_run(root, ["python", "native"], profile_cases=["native"]))
            scan.assert_called_once_with(root, "native")
            self.assertIn('"key": "--seed"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
