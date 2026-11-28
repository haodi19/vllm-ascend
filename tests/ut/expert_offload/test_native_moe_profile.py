# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for gated native profiling and full loading-stream gaps."""

import importlib
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
profile = importlib.import_module("native_moe_profile")


def event(name, start, duration, pid=2, tid=56):
    return dict(name=name, ts=start, dur=duration, pid=pid, tid=tid, ph="X")


def fixture():
    result = [
        dict(name="process_name", pid=2, ph="M", args=dict(name="Ascend Hardware")),
        dict(name="thread_name", pid=2, tid=56, ph="M", args=dict(name="Stream 56")),
        event("host_sync_publish_2", 0, 1, tid=25),
        event("host_sync_wait_2", 1, 3, tid=25),
    ]
    for call, (kind, layer, start) in enumerate((("prefetch", 1, 10), ("reactive", 1, 150), ("prefetch", 2, 300)), 1):
        context = f"backend=cpp|layer={layer}|kind={kind}|rows=1|call={call}"
        result.extend(
            [
                event("moe_offload/host_callback|" + context, start, 100, 1, 7),
                event("moe_offload/update_weights|" + context, start + 1, 98, 1, 7),
                event("MEMCPY_ASYNC", start + 10, 10),
                # A sub-us internal gap is excluded; Transpose occupies the rest.
                event("aclnnInplaceCopy_TransposeAiCore_Transpose", start + 20.05, 39.95),
            ]
        )
    # A long gap is excluded, not mixed into the normal bubbles.
    result.append(event("MEMCPY_ASYNC", 6000, 10))
    return dict(traceEvents=result)


class NativeProfileTests(unittest.TestCase):
    def test_timing_gate_checks_both_python_and_native_status(self):
        before = dict(
            pid=123,
            active=False,
            mstx_active=False,
            mstx_enabled=True,
            mstx_marker_attempts=0,
            inflight=0,
            native=True,
            native_count=0,
            native_marker_errors=0,
        )
        for case in ("python", "native", "native_full"):
            profile.check_timing_status(before, case)
            profile.check_timing_status(dict(before), case, before)
            for changed in (
                dict(pid=124),
                dict(active=True),
                dict(inflight=1),
                dict(mstx_active=True),
                dict(mstx_marker_attempts=1),
            ):
                with self.subTest(case=case, changed=changed), self.assertRaises(RuntimeError):
                    profile.check_timing_status(dict(before, **changed), case, before)
        for changed in (
            dict(native=False),
            dict(native_count=None),
            dict(native_count=1),
            dict(native_marker_errors=1),
        ):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                profile.check_timing_status(dict(before, **changed), "native", before)

    def test_live_profile_never_relaunches_rewarms_or_stops_timed_service(self):
        for fail_request in (False, True):
            with self.subTest(fail_request=fail_request), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = SimpleNamespace(
                    port=8156,
                    timing_port=18156,
                    startup_timeout=60,
                    max_tokens=5,
                    warmup_requests=5,
                    request_timeout=60,
                    profile_timeout=60,
                    profile_requests=2,
                    min_profile_free_gib=5,
                    bubble_stream_name="auto",
                )
                command = ["python", "--additional-config", '{"expert_offload_config":{}}']
                child = Mock(pid=123)
                child.poll.return_value = None
                status = dict(
                    pid=456, active=False, mstx_active=False, mstx_enabled=True, mstx_marker_attempts=0, inflight=0
                )
                sample = dict(digest="same", usage=dict(completion_tokens=5))
                controls = []

                def request(_opener, _port, path, *args, controls=controls, status=status, **kwargs):
                    controls.append(path)
                    if path == "/v1/models":
                        return dict(data=[dict(id="model")])
                    if path == "/status":
                        return status
                    if path in ("/clock/before", "/clock/after"):
                        return dict(samples=[])
                    if path == "/stop":
                        return dict(mstx_errors=0, errors=0, dropped=0, count=6)
                    return {}

                def completion(*args, controls=controls, fail_request=fail_request, sample=sample, **kwargs):
                    self.assertEqual(controls[-1], "/start")
                    self.assertIn("/start_profile", controls)
                    if fail_request:
                        raise TimeoutError("profile request failed")
                    return sample

                stop, ports = Mock(), Mock()
                with (
                    patch.object(profile.subprocess, "Popen") as launch,
                    patch.object(profile, "request", side_effect=request),
                    patch.object(profile, "streaming_completion", side_effect=completion) as requests,
                    patch("shutil.disk_usage", side_effect=AssertionError("storage precheck must not run")),
                    patch.object(profile, "check_host_memory") as memory,
                    patch.object(profile, "after_service_memory") as reclaim,
                    patch.object(profile, "log_memory"),
                    patch.object(profile, "wait_profile"),
                    patch.object(profile, "discover_roots", return_value=[root / "profile/python/capture"]),
                    patch.object(profile, "align_capture", return_value=dict(status="calibration_validated")),
                    patch.object(profile, "publish_aligned_capture", return_value=dict(status="exported")),
                    patch.object(profile, "analyze_profile", return_value=(dict(host_gaps=[]), "")),
                    patch.object(profile, "report_profiles") as summary,
                ):
                    if fail_request:
                        with self.assertRaisesRegex(TimeoutError, "profile request failed"):
                            profile.profile_cases(
                                args,
                                {"python": command},
                                root,
                                {},
                                lambda _: None,
                                ports,
                                stop,
                                [("same", sample["usage"])],
                                live_service=child,
                                timing_status=status,
                            )
                    else:
                        result = profile.profile_cases(
                            args,
                            {"python": command},
                            root,
                            {},
                            lambda _: None,
                            ports,
                            stop,
                            [("same", sample["usage"])],
                            live_service=child,
                            timing_status=status,
                        )
                        self.assertEqual(set(result), {"python"})
                        self.assertEqual(requests.call_count, args.profile_requests)
                        self.assertTrue((root / "python/profile_report.json").is_file())
                    launch.assert_not_called()
                    stop.assert_not_called()
                    ports.assert_not_called()
                    memory.assert_not_called()
                    reclaim.assert_not_called()
                    summary.assert_not_called()  # outer runner reports after closing the service
                self.assertEqual(
                    controls[-2:], ["/clock/after", "/stop_profile"] if not fail_request else ["/stop", "/stop_profile"]
                )

    def test_full_submission_must_be_exercised_not_just_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native_full.log"
            for text in ("", "[MOE-NATIVE] FULL_SUBMIT ready;"):
                log.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "not exercised"):
                    profile.verify_full_submit(log)
            log.write_text(
                "[MOE-NATIVE] FULL_SUBMIT ready;\n[MOE-NATIVE] FULL_SUBMIT executed=prefill;", encoding="utf-8"
            )
            profile.verify_full_submit(log)

    def test_native_scale_allocation_contract_reported_and_enforced(self):
        record = dict(
            policy="shared_final_layout_v1",
            layers=43,
            duplicate_bytes=0,
            reused_bytes=8657043456,
            replaced_bytes=0,
            derived_bytes=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native.log"
            log.write_text("(EngineCore) [MOE-NATIVE] SCALE_MEMORY " + json.dumps(record), encoding="utf-8")
            reports = []
            self.assertEqual(profile.report_native_scale_memory(log, reports.append), [record])
            self.assertIn('"duplicate_bytes": 0', reports[0])
            for bad in (
                dict(record, duplicate_bytes=123),
                dict(record, policy="old"),
                dict(record, layers=0),
                dict(record, reused_bytes=-1),
            ):
                log.write_text("[MOE-NATIVE] SCALE_MEMORY " + json.dumps(bad), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    profile.report_native_scale_memory(log, reports.append)
            log.write_text("[MOE-NATIVE] prepared layers=43 scale_buffer_bytes=8657043456", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evidence missing"):
                profile.report_native_scale_memory(log, reports.append)

    def test_profile_failure_summary_contains_engine_oom_before_api_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native_profile.log"
            log.write_text(
                "old startup message\n" * 5000
                + "pack_scale: torch.empty_strided\ntorch.OutOfMemoryError: aclrtMallocHostWithCfg 207001\n"
                + "API stack frame\n" * 100
                + "RuntimeError: Engine core initialization failed\n",
                encoding="utf-8",
            )
            lines = []
            profile.report_profile_failure("native", log, RuntimeError("service exited"), lines.append)
        self.assertIn("pack_scale: torch.empty_strided", lines)
        self.assertIn("torch.OutOfMemoryError: aclrtMallocHostWithCfg 207001", lines)
        self.assertIn("RuntimeError: Engine core initialization failed", lines)
        # Adjacent startup lines may appear as context, but not the large prefix.
        self.assertLessEqual(lines.count("old startup message"), profile.FAILURE_LOG_CONTEXT)
        self.assertLessEqual(
            len(lines),
            3
            + profile.FAILURE_LOG_TAIL_LINES
            + profile.FAILURE_LOG_MAX_MATCHES * (2 * profile.FAILURE_LOG_CONTEXT + 1),
        )

    def test_profile_request_failure_stops_owned_service(self):
        args = SimpleNamespace(
            port=8156,
            timing_port=18156,
            startup_timeout=60,
            max_tokens=5,
            warmup_requests=1,
            request_timeout=60,
            min_profile_free_gib=5,
        )
        command = ["python", "--additional-config", '{"expert_offload_config":{}}']
        child = Mock()
        child.poll.return_value = None
        stop = Mock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("shutil.disk_usage", side_effect=AssertionError("storage precheck must not run")) as storage,
            patch.object(profile, "check_host_memory") as memory_before,
            patch.object(profile, "log_memory"),
            patch.object(profile, "after_service_memory") as memory_after,
            patch.object(profile.subprocess, "Popen", return_value=child),
            patch.object(profile, "request", side_effect=[dict(data=[dict(id="model")]), dict(mstx_enabled=True)]),
            patch.object(profile, "streaming_completion", side_effect=TimeoutError("request timeout")),
            patch.object(profile, "align_capture") as align,
            self.assertRaisesRegex(TimeoutError, "request timeout"),
        ):
            profile.profile_cases(args, {"python": command}, Path(directory), {}, lambda _: None, Mock(), stop, [])
        stop.assert_called_once_with(child)
        self.assertEqual(memory_before.call_args.args[0], "before profile/python")
        self.assertEqual(memory_after.call_args.args[0], "after profile/python")
        self.assertEqual(memory_after.call_args.args[2], child.pid)
        self.assertTrue(memory_after.call_args.kwargs["drop_cache"])
        align.assert_not_called()
        storage.assert_not_called()

    def test_native_control_uses_native_counters_and_leaves_callbacks_unwrapped(self):
        timing = runpy.run_path(str(ROOT / "vllm_ascend/expert_offload/callback_timing.py"))
        module = ModuleType("vllm_ascend.expert_offload.callback_timing")
        module.CallbackTimings = timing["CallbackTimings"]
        with patch.dict(sys.modules, {module.__name__: module}):
            native = runpy.run_path(str(ROOT / "vllm_ascend/expert_offload/native_profile.py"))
        counters = Mock()
        counters.stop.return_value = dict(count=17, marker_errors=2)
        collector = native["NativeTimings"](counters)
        collector.start()
        self.assertTrue(collector.active)
        observed = collector.stop()
        self.assertEqual(observed["count"], 17)
        self.assertEqual(observed["mstx_errors"], 2)
        self.assertFalse(collector.active)
        self.assertEqual(collector.samples, [])
        self.assertEqual(collector.load_backend, "native")
        self.assertEqual(collector.transport, "acl")
        counters.start.assert_called_once()
        counters.stop.assert_called_once()
        counters.status.return_value = dict(active=False, count=17, marker_errors=2, inflight=0, marker_attempts=85)
        status = collector.status()
        self.assertFalse(status["active"])
        self.assertEqual(status["native_count"], 17)
        self.assertEqual(status["native_marker_errors"], 2)
        self.assertEqual(status["mstx_marker_attempts"], 85)
        counters.status.return_value.update(active=True, inflight=1)
        status = collector.status()
        self.assertTrue(status["active"])
        self.assertTrue(status["mstx_active"])
        self.assertEqual(status["inflight"], 1)

    def test_profile_config_leaves_timing_command_and_workload_unchanged(self):
        config = dict(
            expert_offload_config=dict(
                host_update_backend="native", callback_timing_port=0, callback_mstx_enabled=False, num_device_experts=24
            )
        )
        original = ["python", "--additional-config", json.dumps(config), "--port", "8156"]
        modified = profile.profile_command(original, Path("profile/native"), 18156)
        self.assertEqual(json.loads(original[2]), config)
        observed = json.loads(modified[2])["expert_offload_config"]
        self.assertEqual(observed["host_update_backend"], "native")
        self.assertEqual(observed["num_device_experts"], 24)
        self.assertTrue(observed["callback_mstx_enabled"])
        self.assertEqual(observed["callback_timing_port"], 18156)
        self.assertNotIn("--profiler-config", original)
        self.assertIn("--profiler-config", modified)
        repeated = profile.profile_command(modified, Path("profile/new"), 18157)
        self.assertEqual(repeated.count("--profiler-config"), 1)
        self.assertEqual(json.loads(repeated[-1])["torch_profiler_dir"], str(Path("profile/new")))

    def test_host_gaps_and_transpose_occupancy_are_both_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(fixture()), encoding="utf-8")
            lines = []
            scan, details = profile.analyze_profile(path, "native", 3, "Stream 56", lines.append)
        gaps = {row["direction"]: row["mean_us"] for row in scan["host_gaps"]}
        self.assertEqual(gaps, {"prefetch->reactive": 40, "reactive->prefetch": 50})
        bubbles = [
            json.loads(line.split(" ", 1)[1]) for line in details.splitlines() if line.startswith("TARGET_GAPS_US ")
        ]
        full = next(row for row in bubbles if row["scope"] == "full_capture")
        self.assertEqual(full["excluded_short"], 3)
        self.assertEqual(full["excluded_long"], 1)
        # Real idle gaps 160-70=90 and 310-210=100; not copy-only 140/150.
        self.assertEqual(full["kept"]["n"], 2)
        self.assertEqual(full["kept"]["mean"], 95)
        self.assertTrue(any(line.startswith("HOST_CALLBACK_GAP native ") for line in lines))
        self.assertTrue(any("TARGET_DIRECTION_GAPS_US" in line for line in lines))
        table = []
        profile.report_gap_table({"native": dict(scan=scan, bubble_stats=bubbles)}, table.append)
        self.assertTrue(any("full_capture | 2 | 95" in line for line in table))

    def test_incomplete_host_coverage_and_wrong_stream_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(fixture()), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "callbacks=3/4"):
                profile.analyze_profile(path, "native", 4, "Stream 56", lambda _: None)
            lines = []
            with self.assertRaises(ValueError):
                profile.analyze_profile(path, "native", 3, "Stream 27", lines.append)
            tracks = [json.loads(line.split(" ", 2)[2]) for line in lines if line.startswith("native COPY_TRACK ")]
            self.assertEqual([row["name"] for row in tracks], ["Stream 56"])
            self.assertTrue(any(line.startswith("HOST_CALLBACK_GAP native ") for line in lines))
            self.assertFalse(any("TARGET_GAPS_US " in line for line in lines))

    def test_zero_eligible_gaps_are_missing_not_zero_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(fixture()), encoding="utf-8")
            # All adjacent gaps are excluded by the upper bound, but the
            # selected stream still has three matched loading batches.
            with patch.object(profile, "MAX_BUBBLE_US", 50):
                scan, details = profile.analyze_profile(path, "native", 3, "Stream 56", lambda _: None)
        bubbles = [
            json.loads(line.split(" ", 1)[1])
            for line in details.splitlines()
            if line.startswith(("TARGET_GAPS_US ", "TARGET_DIRECTION_GAPS_US "))
        ]
        self.assertEqual(scan["loading_track"]["loaded_batches"], 3)
        self.assertTrue(all(row["kept"] is None for row in bubbles))
        lines = []
        profile.report_gap_table({"native": dict(scan=scan, bubble_stats=bubbles)}, lines.append)
        for row in bubbles:
            scope = row.get("scope", row.get("direction"))
            self.assertIn(f"| native | Loading bubbles {scope} | 0 | — | — | — |", lines)

    def test_unrelated_existing_stream_is_not_reported_as_loading_bubble_result(self):
        data = fixture()
        data["traceEvents"].extend(
            [
                dict(name="thread_name", pid=2, tid=54, ph="M", args=dict(name="Stream 54")),
                # Both copies cross Host boundaries; no unique loading batch.
                event("MEMCPY_ASYNC", 0, 200, tid=54),
                event("MEMCPY_ASYNC", 270, 400, tid=54),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            scan, details = profile.analyze_profile(path, "native", 3, "Stream 54", lambda _: None)
        bubbles = [
            json.loads(line.split(" ", 1)[1])
            for line in details.splitlines()
            if line.startswith(("TARGET_GAPS_US ", "TARGET_DIRECTION_GAPS_US "))
        ]
        self.assertEqual(scan["loading_track"]["loaded_batches"], 0)
        self.assertEqual(bubbles[0]["kept"]["mean"], 70)
        self.assertIsNone(bubbles[-1]["kept"])
        lines = []
        profile.report_gap_table({"native": dict(scan=scan, bubble_stats=bubbles)}, lines.append)
        self.assertTrue(any(line.startswith("BUBBLES_UNVERIFIED native: Stream 54") for line in lines))
        self.assertTrue(any("| Host prefetch->reactive |" in line for line in lines))
        self.assertFalse(any("| Loading bubbles " in line for line in lines))


if __name__ == "__main__":
    unittest.main()
