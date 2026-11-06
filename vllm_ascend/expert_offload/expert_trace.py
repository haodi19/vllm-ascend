"""Explicit, request-scoped CPU diagnostics. Never used for performance data."""

import functools
import json
import threading
import time
from collections import Counter

MAX_RECORDS = 20000
DRAIN_SECONDS = 30


def resident_map(mapping):
    return {int(eid): int(slot) for eid, slot in enumerate(mapping) if slot >= 0}


class ExpertTrace:
    """Read existing Host buffers; do not introduce device reads or waits.

    layer_visit is an ordinal for each (kind, layer), not an accepted-token
    number. A prefetch is associated with the following reactive visit by its
    record ID, rather than assuming identical global step counters.
    """

    def __init__(self, manager):
        self.manager = manager
        self.condition = threading.Condition()
        self.local = threading.local()
        self.active = False
        self.inflight = 0
        self.records = []
        self.visits = Counter()
        self.pending_prefetch = {}
        self.dropped = 0
        self.metadata = {}
        self.runtime_identity = {}

    def start(self):
        with self.condition:
            if self.active or self.inflight:
                raise ValueError("Expert trace is already active or draining")
            m = self.manager
            self.metadata = dict(
                runtime=self.runtime_identity,
                scope="one measured request; startup/capture/warmup excluded",
                layer_index_base=0,
                step_definition="layer_visit: per-kind per-layer visit ordinal; not output token or scheduler step",
                moe_layers=len(m.moe_layers),
                experts_per_layer=m.num_total_experts,
                top_k=m.topk,
                decode_cache_slots=[m.num_device_experts_for_layer(i) for i in range(len(m.moe_layers))],
                prefill_pool_layers=m.num_device_layers,
                prefetch_load_cap=m.prefetch_topk,
                offload_config=dict(m.offload_config.config),
                cpu_weight_storage_bytes=m._cpu_tensor_storage_bytes((m.w13_weights_cpu, m.w2_weights_cpu)),
                cpu_quant_storage_bytes=m._cpu_tensor_storage_bytes(
                    (m.scale_cpu_buffers, m.offset_cpu_buffers, m.scale_bias_cpu_buffers)
                ),
                host_memory_snapshot=m._host_memory_snapshot(),
                host_memory_fields=["process_rss_bytes", "process_peak_rss_bytes", "host_available_bytes"],
            )
            self.records = []
            self.visits.clear()
            self.pending_prefetch.clear()
            self.dropped = 0
            self.active = True

    def stop(self):
        with self.condition:
            self.active = False
            if not self.condition.wait_for(lambda: self.inflight == 0, timeout=DRAIN_SECONDS):
                raise ValueError("Expert trace callbacks did not drain; service log retains failure location")
            errors = sum(record.get("status") != "completed" for record in self.records)
            return dict(
                metadata=self.metadata,
                complete=bool(self.records) and not self.dropped and not errors,
                dropped=self.dropped,
                errors=errors,
                records=self.records,
            )

    def wrap_call(self, fn, prefill=False):
        @functools.wraps(fn)
        def traced(*args, **kwargs):
            with self.condition:
                if not self.active:
                    enabled = False
                elif len(self.records) >= MAX_RECORDS:
                    self.dropped += 1
                    enabled = False
                else:
                    enabled = True
                    layer = int(args[0] if prefill else args[0][3])
                    kind = "prefill_pool" if prefill else ("prefetch" if args[0][5] else "reactive")
                    self.visits[kind, layer] += 1
                    record = dict(
                        event_id=len(self.records) + 1,
                        kind=kind,
                        layer=layer,
                        layer_visit=self.visits[kind, layer],
                        start_ns=time.perf_counter_ns(),
                        thread_id=threading.get_native_id(),
                        status="entered",
                        loads=[],
                        copies=[],
                        submit_returned=False,
                        h2d_wait_returned=False,
                    )
                    self.records.append(record)
                    self.inflight += 1
            if not enabled:
                return fn(*args, **kwargs)
            self.local.record = record
            try:
                if prefill:
                    record["pool_slot"] = layer % self.manager.num_device_layers
                    record["full_pool_experts"] = self.manager.num_total_experts
                else:
                    values = args[0]
                    before = resident_map(values[1].tolist())
                    rows = values[0].tolist()
                    needed = {int(e) for row in rows for e in row if 0 <= e < self.manager.num_total_experts}
                    record.update(
                        route_rows=rows,
                        router_weights=values[4].tolist() if values[4] is not None else None,
                        needed_ids=sorted(needed),
                        hit_ids=sorted(needed & before.keys()),
                        missing_ids=sorted(needed - before.keys()),
                        resident_before=before,
                        route_rows_meaning="predicted rows"
                        if kind == "prefetch"
                        else "actual routed rows; may include MTP positions/padding",
                    )
                    if kind == "reactive":
                        with self.condition:
                            previous = self.pending_prefetch.pop(layer, None)
                        record["preceding_prefetch_event"] = previous[0] if previous else None
                        record["prefetched_ids_used"] = sorted(needed & previous[1]) if previous else []
                        record["prefetched_ids_unused"] = sorted(previous[1] - needed) if previous else []
                print("[EXPERT-TRACE BEGIN] " + json.dumps(record), flush=True)
                result = fn(*args, **kwargs)
                if not prefill:
                    after = resident_map(args[0][1].tolist())
                    loaded = {int(eid) for _, eid in record["loads"]}
                    record.update(
                        resident_after=after,
                        loaded_ids=sorted(loaded),
                        evicted_ids=sorted(before.keys() - after.keys()),
                        missing_not_loaded=sorted(needed - before.keys() - loaded),
                        still_missing_ids=sorted(needed - after.keys()),
                    )
                    if kind == "prefetch":
                        with self.condition:
                            self.pending_prefetch[layer] = (record["event_id"], loaded)
                record["status"] = "completed" if record["h2d_wait_returned"] else "missing_completion_wait"
                if kind == "reactive" and record.get("still_missing_ids"):
                    record["status"] = "reactive_missing_after_load"
                return result
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["end_ns"] = time.perf_counter_ns()
                record["logical_h2d_bytes"] = sum(task["bytes"] for task in record["copies"])
                print("[EXPERT-TRACE END] " + json.dumps(record), flush=True)
                self.local.record = None
                with self.condition:
                    self.inflight -= 1
                    self.condition.notify_all()

        return traced

    def wrap_load(self, fn):
        @functools.wraps(fn)
        def traced(layer, layer_idx, loads):
            record = getattr(self.local, "record", None)
            if record is not None:
                loads = list(loads)
                record["loads"].extend([[int(slot), int(eid)] for slot, eid in loads])
            return fn(layer, layer_idx, loads)

        return traced

    def wrap_copy(self, fn):
        @functools.wraps(fn)
        def traced(tasks):
            record = getattr(self.local, "record", None)
            if record is not None:
                tasks = list(tasks)
                record["copies"].extend(dict(name=t.name, bytes=int(t.nbytes)) for t in tasks)
                print(
                    "[EXPERT-TRACE COPY] " + json.dumps(dict(event_id=record["event_id"], copies=record["copies"])),
                    flush=True,
                )
            result = fn(tasks)
            if record is not None:
                record["submit_returned"] = True
            return result

        return traced

    def wrap_wait(self, fn):
        @functools.wraps(fn)
        def traced(*args, **kwargs):
            result = fn(*args, **kwargs)
            record = getattr(self.local, "record", None)
            if record is not None:
                record["h2d_wait_returned"] = True
            return result

        return traced


def install_expert_trace(manager):
    """Install only in diagnostic mode, before graph callbacks bind methods."""
    collector = ExpertTrace(manager)
    manager._update_weights = collector.wrap_call(manager._update_weights)
    manager._prefill_load_layer = collector.wrap_call(manager._prefill_load_layer, prefill=True)
    manager._load_expert_weights_into_slots = collector.wrap_load(manager._load_expert_weights_into_slots)
    transport = manager._get_h2d_transport()
    transport.copy_batch = collector.wrap_copy(transport.copy_batch)
    manager._synchronize_h2d = collector.wrap_wait(manager._synchronize_h2d)
    return collector
