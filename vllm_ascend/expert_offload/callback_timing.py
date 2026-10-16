"""Opt-in, bounded, in-memory timing of actual single-card host callbacks.

The localhost control endpoint opens/closes a measurement window after model
warmup. No profiles, request content or timing files are written.
"""

import functools
import json
import math
import os
import statistics
import threading
import time
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_SAMPLES = 50000
GAP_METRICS = ("gap_mapping", "gap_missing", "gap_hit_stats", "gap_cache_stats", "gap_slots", "gap_debug")
DRAIN_SECONDS = 30
METRICS = (
    "total",
    "observe",
    "choose_victims",
    "synchronize",
    "outer",
    "host_without_sync",
    "load_submit",
    "fused_plan",
    "load_prepare",
    "copy_submit",
    "scale_refresh",
    "load_other",
)


def summarize(values):
    values = sorted(values)

    def percentile(q):
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return dict(
        mean=statistics.mean(values),
        p50=percentile(0.5),
        p95=percentile(0.95),
        p99=percentile(0.99),
        min=values[0],
        max=values[-1],
    )


def callback_stage(manager, name):
    """Fine-grained diagnostics only inside an active MSTX callback window."""
    collector = getattr(manager, "_callback_timing", None)
    if collector is None or collector.mstx is None or getattr(collector.local, "values", None) is None:
        return nullcontext()
    return collector.stage(name)


def callback_boundary_stage(manager, name):
    collector = getattr(manager, "_callback_timing", None)
    if collector is None or not getattr(collector.local, "outer_context", None):
        return nullcontext()
    return collector.boundary_stage(name)


class CallbackTimings:
    def __init__(self, backend, clock=time.perf_counter_ns, mstx=None):
        self.backend = backend
        self.clock = clock
        self.mstx = mstx
        self.marker_errors = 0
        self.call_id = 0
        self.local = threading.local()
        self.condition = threading.Condition()
        self.active = False
        self.inflight = 0
        self.samples = []
        self.errors = self.dropped = 0
        self.server = None

    def start(self):
        with self.condition:
            if self.active or self.inflight:
                raise ValueError("A measurement window is already active or draining")
            self.samples.clear()
            self.errors = self.dropped = 0
            self.marker_errors = 0
            self.active = True

    def range_start(self, name):
        if self.mstx is None:
            return None
        try:
            # No stream argument: measure CPU execution/waiting, not NPU work.
            range_id = self.mstx.range_start(f"moe_offload/{name}|{self.local.context}")
            if not isinstance(range_id, int) or range_id <= 0:
                raise RuntimeError("MSTX did not return a valid range ID")
            return range_id
        except Exception:
            # Diagnostics must not change callback execution or mask its errors.
            with self.condition:
                self.marker_errors += 1
            return None

    def range_end(self, range_id):
        if range_id is not None:
            try:
                self.mstx.range_end(range_id)
            except Exception:
                with self.condition:
                    self.marker_errors += 1

    def wrap(self, name, fn, count_loads=False):
        @functools.wraps(fn)
        def measured(*args, **kwargs):
            values = getattr(self.local, "values", None)
            if values is None:
                return fn(*args, **kwargs)
            if count_loads:
                values["loads"] += len(args[2])  # manager passes planned_loads list
            range_id = self.range_start(name)
            begin = self.clock()
            try:
                return fn(*args, **kwargs)
            finally:
                values[name] += self.clock() - begin
                self.range_end(range_id)

        return measured

    @contextmanager
    def boundary_stage(self, name):
        range_id = self.range_start(name)
        try:
            yield
        finally:
            self.range_end(range_id)

    def wrap_callback(self, fn):
        @functools.wraps(fn)
        def measured(args):
            with self.condition:
                enabled = self.active and self.mstx is not None
                if enabled:
                    self.inflight += 1
                    self.call_id += 1
                    call_id = self.call_id
            if not enabled:
                return fn(args)
            previous = getattr(self.local, "context", None)
            kind = "prefetch" if args[5] else "reactive"
            self.local.context = (
                f"backend={self.backend}|layer={args[3]}|kind={kind}|rows={args[0].shape[0]}|call={call_id}"
            )
            self.local.outer_context = self.local.context
            try:
                with self.boundary_stage("host_callback"):
                    return fn(args)
            finally:
                self.local.outer_context = None
                self.local.context = previous
                with self.condition:
                    self.inflight -= 1
                    self.condition.notify_all()

        return measured

    @contextmanager
    def stage(self, name):
        range_id = self.range_start(name)
        begin = self.clock()
        try:
            yield
        finally:
            self.local.values[name] += self.clock() - begin
            self.range_end(range_id)

    def wrap_update(self, fn):
        @functools.wraps(fn)
        def measured(args):
            with self.condition:
                enabled = self.active or bool(getattr(self.local, "outer_context", None))
                if enabled:
                    self.inflight += 1
                    self.call_id += 1
                    call_id = self.call_id
            if not enabled:
                return fn(args)
            values = dict.fromkeys((*METRICS, *GAP_METRICS, "loads"), 0)
            self.local.values = values
            if self.mstx is not None and not getattr(self.local, "outer_context", None):
                kind = "prefetch" if args[5] else "reactive"
                self.local.context = (
                    f"backend={self.backend}|layer={args[3]}|kind={kind}|rows={args[0].shape[0]}|call={call_id}"
                )
            range_id = self.range_start("update_weights")
            success = False
            begin = self.clock()
            try:
                result = fn(args)
                success = True
                return result
            finally:
                values["total"] = self.clock() - begin
                self.range_end(range_id)
                self.local.values = None
                values["host_without_sync"] = values["total"] - values["synchronize"]
                values["outer"] = values["host_without_sync"] - values["observe"] - values["choose_victims"]
                values["load_other"] = values["load_submit"] - sum(
                    values[k] for k in ("load_prepare", "copy_submit", "scale_refresh")
                )
                # Only shapes/indices are retained, never routes, weights or text.
                with self.condition:
                    try:
                        if not success:
                            self.errors += 1
                        elif len(self.samples) >= MAX_SAMPLES:
                            self.dropped += 1
                        else:
                            group = ("prefetch" if args[5] else "reactive", int(args[0].shape[0]))
                            self.samples.append((group, int(args[3]), values))
                    finally:
                        self.inflight -= 1
                        self.condition.notify_all()

        return measured

    def stop(self):
        with self.condition:
            self.active = False
            if not self.condition.wait_for(lambda: self.inflight == 0, timeout=DRAIN_SECONDS):
                raise ValueError("Timed out draining callbacks; results are incomplete")
            samples = list(self.samples)
            errors, dropped = self.errors, self.dropped
        groups = {}
        for key, layer, values in samples:
            group = groups.setdefault(key, [])
            group.append((layer, values))
        reports = []
        for (kind, rows), records in sorted(groups.items()):
            reports.append(
                dict(
                    kind=kind,
                    token_rows=rows,
                    count=len(records),
                    layers=sorted({r[0] for r in records}),
                    loads=summarize([r[1]["loads"] for r in records]),
                    metrics={key: summarize([r[1][key] / 1000 for r in records]) for key in METRICS},
                    gap_metrics=(
                        {key: summarize([r[1][key] / 1000 for r in records]) for key in GAP_METRICS}
                        if self.mstx is not None
                        else {}
                    ),
                )
            )
        return dict(
            backend=self.backend,
            count=len(samples),
            errors=errors,
            dropped=dropped,
            groups=reports,
            units="us",
            mstx_enabled=self.mstx is not None,
            mstx_errors=self.marker_errors,
        )

    def serve(self, port):
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/status":
                    self.respond(404, {"error": "unknown endpoint"})
                    return
                self.respond(
                    200,
                    dict(
                        backend=collector.backend,
                        pid=os.getpid(),
                        active=collector.active,
                        policy="LRC only",
                        transport="torch",
                        load_backend="python",
                        mstx_enabled=collector.mstx is not None,
                    ),
                )

            def do_POST(self):
                try:
                    # Consume the control request body before closing HTTP/1.0;
                    # unread bytes can cause a TCP reset instead of a response.
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    if self.path == "/start":
                        collector.start()
                        self.respond(200, {"started": True})
                    elif self.path == "/stop":
                        self.respond(200, collector.stop())
                    else:
                        self.respond(404, {"error": "unknown endpoint"})
                except ValueError as exc:
                    self.respond(409, {"error": str(exc)})

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True, name="offload-timing-control").start()

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def install_callback_timing(manager, port):
    """Explicit diagnostic mode; original execution is untouched when disabled."""
    if hasattr(manager, "_callback_timing"):
        raise RuntimeError("Callback timing installed twice on this manager")
    backend = manager.offload_config.cache_policy_backend
    if backend not in ("python", "cpp") or manager.enable_multi_card or manager.cache_policy is None:
        raise ValueError("Callback timing requires single-card python/cpp LRC")
    if manager.offload_config.h2d_backend != "torch":
        raise ValueError("Callback timing requires original TorchCopy transport")
    mstx = None
    if manager.offload_config.callback_mstx_enabled:
        # Lazy import: standalone CPU benchmarks and disabled mode need no NPU API.
        from torch_npu.npu import mstx

        if not all(callable(getattr(mstx, name, None)) for name in ("range_start", "range_end")):
            raise RuntimeError("Installed torch_npu does not provide MSTX host ranges")
    collector = CallbackTimings(backend, mstx=mstx)
    if mstx is not None:
        manager._update_weights_guarded = collector.wrap_callback(manager._update_weights_guarded)
    for name in ("observe", "choose_victims"):
        setattr(manager.cache_policy, name, collector.wrap(name, getattr(manager.cache_policy, name)))
    for method, metric in (
        ("_synchronize_h2d", "synchronize"),
        ("_load_expert_weights_into_slots", "load_submit"),
        ("_build_expert_h2d_tasks", "load_prepare"),
        ("_refresh_expert_fp32_scale", "scale_refresh"),
    ):
        setattr(manager, method, collector.wrap(metric, getattr(manager, method), count_loads=metric == "load_submit"))
    transport = manager._get_h2d_transport()
    transport.copy_batch = collector.wrap("copy_submit", transport.copy_batch)
    manager._update_weights = collector.wrap_update(manager._update_weights)
    collector.serve(port)
    manager._callback_timing = collector
