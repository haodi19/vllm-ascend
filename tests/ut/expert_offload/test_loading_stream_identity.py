# SPDX-License-Identifier: Apache-2.0
"""Actual loading-stream identity: no NPU required for these regression cases."""

import contextlib
import importlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from test_native_moe_profile import fixture

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
gaps = importlib.import_module("verify_memcpy_gaps")

IDENTITY = runpy.run_path(str(ROOT / "vllm_ascend/expert_offload/profile_loading_stream.py"))
TIMING = runpy.run_path(str(ROOT / "vllm_ascend/expert_offload/callback_timing.py"))


def marker(stream_id=54, pid=1):
    return dict(
        ph="X",
        name=f"moe_offload/loading_stream|schema=1|stream_id={stream_id}|device=0",
        pid=pid,
        tid=8,
        ts=0,
        dur=1,
    )


class LoadingStreamTests(unittest.TestCase):
    def test_query_uses_acl_id_not_handle_and_does_not_initialize_acl(self):
        query = Mock()

        def get_id(handle, output):
            self.assertEqual(handle.value, 123456)
            output._obj.value = 54
            return 0

        query.side_effect = get_id
        library = SimpleNamespace(aclrtStreamGetId=query)
        with patch.object(IDENTITY["ctypes"], "CDLL", return_value=library) as load:
            result = IDENTITY["loading_stream_identity"](SimpleNamespace(npu_stream=123456), 0)
        load.assert_called_once_with("libascendcl.so")
        query.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stream_id"], 54)
        self.assertNotIn("123456", json.dumps(result))

    def test_query_failure_does_not_invent_an_identity_or_break_inference(self):
        for library in (SimpleNamespace(), SimpleNamespace(aclrtStreamGetId=Mock(return_value=1))):
            with patch.object(IDENTITY["ctypes"], "CDLL", return_value=library):
                result = IDENTITY["loading_stream_identity"](SimpleNamespace(npu_stream=123456), 0)
            self.assertEqual(result["status"], "unavailable")
            self.assertNotIn("stream_id", result)
        with patch.object(IDENTITY["ctypes"], "CDLL") as load:
            result = IDENTITY["loading_stream_identity"](SimpleNamespace(npu_stream=0), 0)
        load.assert_not_called()
        self.assertEqual(result["status"], "unavailable")

    def test_marker_records_identity_once_per_window_not_in_update(self):
        module = ModuleType("vllm_ascend.expert_offload.profile_loading_stream")
        module.mark_loading_stream = Mock()
        collector = TIMING["CallbackTimings"]("cpp", mstx=Mock())
        collector.loading_stream = dict(status="ok", schema=1, stream_id=54, device=0)
        with patch.dict(sys.modules, {module.__name__: module}):
            collector.start()
            # A profiled request can contain many Host tasks; entry does not
            # requery the ACL stream or emit another loading identity.
            function = collector.wrap("load_submit", lambda: 7)
            for _ in range(10):
                self.assertEqual(function(), 7)
            result = collector.stop()
        module.mark_loading_stream.assert_called_once_with(collector.loading_stream)
        self.assertEqual(result["loading_stream"]["stream_id"], 54)
        self.assertEqual(result["mstx_errors"], 0)
        self.assertEqual(collector.marker_attempts, 1)

    def test_native_control_records_stream_and_combines_marker_errors(self):
        timing_module = ModuleType("vllm_ascend.expert_offload.callback_timing")
        timing_module.CallbackTimings = TIMING["CallbackTimings"]
        identity_module = ModuleType("vllm_ascend.expert_offload.profile_loading_stream")
        identity_module.mark_loading_stream = Mock(side_effect=RuntimeError("marker failed"))
        with patch.dict(
            sys.modules, {timing_module.__name__: timing_module, identity_module.__name__: identity_module}
        ):
            native = runpy.run_path(str(ROOT / "vllm_ascend/expert_offload/native_profile.py"))
            profiler = Mock()
            profiler.stop.return_value = dict(count=170, marker_errors=2)
            collector = native["NativeTimings"](profiler)
            collector.loading_stream = dict(status="ok", schema=1, stream_id=54, device=0)
            collector.start()
            result = collector.stop()
        self.assertEqual(result["count"], 170)
        self.assertEqual(result["mstx_errors"], 3)
        self.assertEqual(result["loading_stream"]["stream_id"], 54)
        identity_module.mark_loading_stream.assert_called_once()

    def test_no_marker_when_mstx_disabled_and_marker_errors_reported(self):
        module = ModuleType("vllm_ascend.expert_offload.profile_loading_stream")
        module.mark_loading_stream = Mock(side_effect=RuntimeError("marker failed"))
        with patch.dict(sys.modules, {module.__name__: module}):
            for enabled in (False, True):
                collector = TIMING["CallbackTimings"]("cpp", mstx=Mock() if enabled else None)
                collector.loading_stream = dict(status="ok", schema=1, stream_id=54, device=0)
                collector.start()
                self.assertEqual(collector.stop()["mstx_errors"], int(enabled))
        self.assertEqual(module.mark_loading_stream.call_count, 1)

    def test_mstx_marker_does_not_create_a_device_task(self):
        npu = ModuleType("torch_npu.npu")
        npu.mstx = Mock()
        npu.mstx.range_start.return_value = 123
        with patch.dict(sys.modules, {npu.__name__: npu}):
            IDENTITY["mark_loading_stream"](dict(schema=1, stream_id=54, device=0))
        npu.mstx.range_start.assert_called_once_with("moe_offload/loading_stream|schema=1|stream_id=54|device=0")
        npu.mstx.range_end.assert_called_once_with(123)

    def inspect(self, data, stream="auto", **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                gaps.inspect_offload_trace(path, stream, 10, 5000, **kwargs)
            return output.getvalue()

    def test_stream_changes_and_larger_unrelated_track_does_not_win(self):
        for stream_id in (54, 56, 102):
            with self.subTest(stream_id=stream_id):
                data = fixture()
                # Keep tid=56 so it cannot be confused with the display name.
                data["traceEvents"][1]["args"]["name"] = f"Stream {stream_id}"
                data["traceEvents"].append(marker(stream_id))
                data["traceEvents"].append(dict(ph="M", name="thread_name", pid=2, tid=99, args=dict(name="Stream 99")))
                data["traceEvents"].extend(
                    dict(ph="X", name="MEMCPY_ASYNC", pid=2, tid=99, ts=n * 2, dur=1) for n in range(1000)
                )
                output = self.inspect(data)
                selection = json.loads(
                    next(
                        line.split(" ", 1)[1]
                        for line in output.splitlines()
                        if line.startswith("LOADING_STREAM_SELECTION ")
                    )
                )
                self.assertEqual(selection["method"], "runtime_acl_id")
                self.assertEqual((selection["name"], selection["tid"]), (f"Stream {stream_id}", "56"))
                full = json.loads(
                    next(line.split(" ", 1)[1] for line in output.splitlines() if line.startswith("TARGET_GAPS_US "))
                )
                self.assertEqual(full["kept"]["mean"], 95)

    def test_absent_conflicting_foreign_or_malformed_markers_never_guess(self):
        for marks in ([], [marker(56, pid=9)], [marker(56), marker(54)], [marker("invalid")]):
            data = fixture()
            data["traceEvents"].extend(marks)
            with self.subTest(marks=marks), self.assertRaises(ValueError):
                self.inspect(data)
        data = fixture()
        data["traceEvents"].append(marker(56))
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.inspect(data, "Stream 54")

    def test_ambiguous_hardware_tracks_fail_and_legacy_explicit_still_works(self):
        data = fixture()
        self.assertIn('"method":"explicit_legacy"', self.inspect(data, "Stream 56"))
        data["traceEvents"].extend(
            [
                marker(56),
                dict(ph="M", name="process_name", pid=3, args=dict(name="Ascend Hardware")),
                dict(ph="M", name="thread_name", pid=3, tid=56, args=dict(name="Stream 56")),
                dict(ph="X", name="MEMCPY_ASYNC", pid=3, tid=56, ts=0, dur=1),
            ]
        )
        with self.assertRaisesRegex(ValueError, "不唯一"):
            self.inspect(data)
        # Even a uniquely named track is insufficient across devices: the
        # marker uses logical device IDs, while exports may use physical IDs.
        data["traceEvents"][-2]["args"]["name"] = "Stream 108"
        with self.assertRaisesRegex(ValueError, "不唯一"):
            self.inspect(data)


if __name__ == "__main__":
    unittest.main()
