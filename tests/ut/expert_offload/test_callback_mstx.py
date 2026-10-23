"""CPU-only tests of real diagnostic wrappers; no Ascend installation needed."""

import ast
import contextlib
import io
import json
import runpy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[3]
TIMING = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/callback_timing.py"))
RUNNER = runpy.run_path(str(REPO / "script/run_verified_cpp_tests.py"))


class FakeMstx:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def range_start(self, message):
        # Intentionally accepts no stream: a device range would fail this test.
        with self.lock:
            rid = len(self.events) + 1
            self.events.append(("start", rid, message, threading.get_ident()))
            return rid

    def range_end(self, rid):
        with self.lock:
            self.events.append(("end", rid, "", threading.get_ident()))


class TestMstx(unittest.TestCase):
    def test_load_detail_nested_timing_and_counts(self):
        now = [0]
        collector = TIMING["CallbackTimings"]("python", clock=lambda: now[0], mstx=FakeMstx())
        manager = SimpleNamespace(_callback_timing=collector)

        def update(args):
            for _ in range(2):
                with TIMING["callback_stage"](manager, "quant_adapter"):
                    now[0] += 1000
                    with TIMING["callback_stage"](manager, "quant_allocate"):
                        now[0] += 2000
                    with TIMING["callback_stage"](manager, "quant_pack"):
                        now[0] += 5000

        collector.start()
        collector.wrap_update(update)(self.args())
        report = collector.stop()
        details = report["groups"][0]["load_details"]
        self.assertEqual(details["quant_adapter"]["mean"], 16)
        self.assertEqual(details["quant_pack"]["mean"], 10)
        self.assertEqual(details["quant_allocate"]["mean"], 4)
        self.assertEqual(details["quant_allocate"]["calls"], 2)
        self.assertEqual(details["quant_get_format"]["calls"], 0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            RUNNER["print_inference_report"]("python", report, [(1000, {}, "digest")])
        self.assertIn("LOAD DETAILS", output.getvalue())
        self.assertIn("quant_adapter_residual_mean_us=2.000", output.getvalue())

    def test_outer_callback_context_and_snapshot(self):
        mstx = FakeMstx()
        collector = TIMING["CallbackTimings"]("cpp", mstx=mstx)
        manager = SimpleNamespace(_callback_timing=collector)
        inner = collector.wrap_update(lambda args: 5)

        def outer(args):
            with TIMING["callback_boundary_stage"](manager, "prefetch_snapshot"):
                pass
            return inner(args)

        call = collector.wrap_callback(outer)
        call(self.args(prefetch=True))
        self.assertEqual(mstx.events, [])
        collector.start()
        self.assertEqual(call(self.args(prefetch=True)), 5)
        report = collector.stop()
        self.assertEqual(report["count"], 1)
        starts = [e[2] for e in mstx.events if e[0] == "start"]
        self.assertEqual(
            [s.split("|")[0] for s in starts],
            ["moe_offload/host_callback", "moe_offload/prefetch_snapshot", "moe_offload/update_weights"],
        )
        self.assertEqual(len({s.partition("|")[2] for s in starts}), 1)
        self.assertEqual(collector.inflight, 0)
        self.assertIsNone(collector.local.outer_context)

    def test_outer_exception_drains_and_closes(self):
        collector = TIMING["CallbackTimings"]("python", mstx=FakeMstx())

        def fail(args):
            raise ValueError("snapshot failed")

        collector.start()
        with self.assertRaisesRegex(ValueError, "snapshot failed"):
            collector.wrap_callback(fail)(self.args())
        self.assertEqual(collector.inflight, 0)
        self.assertEqual([e[0] for e in collector.mstx.events], ["start", "end"])
        collector.stop()

    def test_gap_analysis_union_and_boundaries(self):
        module = runpy.run_path(str(REPO / "script/analyze_callback_gaps.py"))

        def event(name, call, kind, ts, dur):
            return dict(
                name=f"moe_offload/{name}|call={call}|kind={kind}|layer=2", ph="X", pid=1, tid=1, ts=ts, dur=dur
            )

        data = [
            event("host_callback", 1, "prefetch", 0, 15),
            event("update_weights", 1, "prefetch", 2, 10),
            event("host_callback", 2, "reactive", 25, 15),
            event("update_weights", 2, "reactive", 28, 10),
            dict(name="MEMCPY_ASYNC", ph="X", ts=14, dur=6),
            dict(name="MEMCPY_ASYNC", ph="X", ts=18, dur=4),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path, out = Path(folder) / "trace.json", Path(folder) / "overlay.json"
            path.write_text(json.dumps(dict(traceEvents=data)), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                rows = module["analyze"](path, out)
            row = rows[0]
            self.assertEqual(
                (row["gap_us"], row["callback_tail_us"], row["between_callbacks_us"], row["callback_head_us"]),
                (16, 3, 10, 3),
            )
            self.assertEqual(row["memcpy_overlap_us"], 8)
            self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))["traceEvents"]), len(data) + 3)
            path.write_text('{"traceEvents":[{"ph":"X"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "完整"):
                list(module["events"](path))

    def test_gap_steps_timing_disabled_and_exception(self):
        now = [0]
        mstx = FakeMstx()
        collector = TIMING["CallbackTimings"]("cpp", clock=lambda: now[0], mstx=mstx)
        manager = SimpleNamespace(_callback_timing=collector)

        def update(args):
            for index, name in enumerate(TIMING["GAP_METRICS"], 1):
                with TIMING["callback_stage"](manager, name):
                    now[0] += index * 1000

        call = collector.wrap_update(update)
        call(self.args())
        self.assertEqual(mstx.events, [])
        collector.start()
        call(self.args())
        gap = collector.stop()["groups"][0]["gap_metrics"]
        for index, name in enumerate(TIMING["GAP_METRICS"], 1):
            self.assertEqual(gap[name]["mean"], index)

        def fail(args):
            with TIMING["callback_stage"](manager, "gap_debug"):
                raise RuntimeError("log failure")

        collector.start()
        with self.assertRaisesRegex(RuntimeError, "log failure"):
            collector.wrap_update(fail)(self.args())
        self.assertEqual(collector.stop()["errors"], 1)
        self.assertEqual(sum(e[0] == "start" for e in mstx.events), sum(e[0] == "end" for e in mstx.events))
        collector.mstx = None
        collector.start()
        call(self.args())
        self.assertEqual(collector.stop()["groups"][0]["gap_metrics"], {})

    @staticmethod
    def args(layer=2, prefetch=False):
        return (SimpleNamespace(shape=(3, 8)), None, None, layer, None, prefetch)

    def test_nested_ranges_and_warmup(self):
        mstx = FakeMstx()
        collector = TIMING["CallbackTimings"]("cpp", mstx=mstx)
        observe = collector.wrap("observe", lambda: 17)
        load = collector.wrap("load_submit", collector.wrap("copy_submit", lambda: 23))
        call = collector.wrap_update(lambda args: (observe(), load()))
        self.assertEqual(call(self.args()), (17, 23))
        self.assertEqual(mstx.events, [])
        collector.start()
        self.assertEqual(call(self.args()), (17, 23))
        report = collector.stop()
        self.assertTrue(report["mstx_enabled"])
        self.assertEqual(report["mstx_errors"], 0)
        stack = []
        names = []
        for action, rid, label, _ in mstx.events:
            if action == "start":
                stack.append(rid)
                names.append(label.split("|")[0])
                self.assertIn("backend=cpp|layer=2|kind=reactive|rows=3|call=1", label)
            else:
                self.assertEqual(stack.pop(), rid)
        self.assertEqual(stack, [])
        self.assertEqual(
            names, ["moe_offload/" + n for n in ("update_weights", "observe", "load_submit", "copy_submit")]
        )

    def test_exceptions_close_ranges_and_drain(self):
        mstx = FakeMstx()
        collector = TIMING["CallbackTimings"]("python", mstx=mstx)

        def fail():
            raise ValueError("original failure")

        inner = collector.wrap("choose_victims", fail)
        collector.start()
        with self.assertRaisesRegex(ValueError, "original failure"):
            collector.wrap_update(lambda args: inner())(self.args(prefetch=True))
        self.assertEqual([e[0] for e in mstx.events], ["start", "start", "end", "end"])
        self.assertEqual(collector.stop()["errors"], 1)
        self.assertEqual(collector.inflight, 0)

    def test_broken_marker_preserves_callback(self):
        for value in (None, 0):
            mstx = MagicMock()
            mstx.range_start.return_value = value
            collector = TIMING["CallbackTimings"]("python", mstx=mstx)
            collector.start()
            self.assertEqual(collector.wrap_update(lambda args: 5)(self.args()), 5)
            self.assertEqual(collector.stop()["mstx_errors"], 1)
            mstx.range_end.assert_not_called()

    def test_threads_have_distinct_contexts(self):
        mstx = FakeMstx()
        collector = TIMING["CallbackTimings"]("python", mstx=mstx)
        barrier = threading.Barrier(2)
        inner = collector.wrap("observe", lambda: None)

        def update(args):
            barrier.wait(timeout=5)
            inner()

        call = collector.wrap_update(update)
        collector.start()
        threads = [threading.Thread(target=call, args=(self.args(layer),)) for layer in (2, 4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(collector.stop()["count"], 2)
        contexts = {}
        for action, _, label, tid in mstx.events:
            if action == "start":
                context = label.partition("|")[2]
                self.assertEqual(contexts.setdefault(tid, context), context)
        self.assertEqual(len(set(contexts.values())), 2)

    def test_launcher_opt_in(self):
        source = (REPO / "script/test_graph2.sh").read_text(encoding="utf-8")
        for enabled in (False, True):
            _, words, _, _ = RUNNER["launch_spec"](source, "cpp", 1, 8155, 18155, mstx=enabled)
            config = json.loads(words[words.index("--additional-config") + 1])
            self.assertEqual(config["expert_offload_config"].get("callback_mstx_enabled", False), enabled)

    def test_trace_presence_across_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace_view.json"
            path.write_bytes(b" " * (1024 * 1024 - 4) + b"moe_offload/update_weights moe_offload/observe")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(RUNNER["inspect_mstx_trace"](Path(folder)), 2)

    def test_profiler_capture_flag(self):
        # Execute the actual wrapper class with mocked dependencies, avoiding
        # package imports that require vLLM and NPU hardware on the blue machine.
        path = REPO / "vllm_ascend/profiler/torch_npu_profiler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        for enabled in (False, True):
            npu = MagicMock()
            config = SimpleNamespace(
                msmonitor_use_daemon=False,
                expert_offload_config=SimpleNamespace(callback_mstx_enabled=enabled),
            )
            namespace = dict(
                torch_npu=npu,
                WorkerProfiler=object,
                ProfilerConfig=object,
                Any=object,
                suppress=contextlib.suppress,
                get_ascend_config=lambda config=config: config,
                envs_ascend=SimpleNamespace(MSMONITOR_USE_DAEMON=False),
            )
            exec(compile(tree, str(path), "exec"), namespace)
            namespace["TorchNPUProfilerWrapper"]._create_profiler(
                SimpleNamespace(
                    profiler="torch",
                    torch_profiler_dir="traces",
                    torch_profiler_with_memory=False,
                    torch_profiler_with_stack=False,
                ),
                "test",
            )
            kwargs = npu.profiler._ExperimentalConfig.call_args.kwargs
            self.assertEqual(kwargs.get("mstx", False), enabled)


if __name__ == "__main__":
    unittest.main()
