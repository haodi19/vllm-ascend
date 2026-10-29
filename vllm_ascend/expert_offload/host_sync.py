# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-card graph callback dispatch through registered MTE signals."""

import atexit
import ctypes
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path


def create_offload_load_stream(npu, backend):
    """Keep polling-worker H2D off the model's Host submission queue."""
    if backend == "callback":
        return npu.Stream()
    if backend != "mte":
        raise ValueError("host_sync_backend must be callback or mte")
    stream_type = getattr(npu, "SyncLaunchStream", None)
    if stream_type is None:
        raise RuntimeError("MTE host sync requires torch_npu.npu.SyncLaunchStream; no ordinary-stream fallback")
    # Sync launch means direct Host submission, not waiting after every copy.
    # _synchronize_h2d still waits for the actual device transfers to finish.
    return stream_type()


def validate_host_sync_config(config):
    cores = config.get("host_sync_diagnose_load_vector_cores", 0)
    if type(cores) is not int or not 0 <= cores <= 56:
        raise ValueError("diagnostic load vector cores must be an integer in 0..56")
    if cores and not (config.get("host_sync_diagnose_load") and config.get("host_sync_diagnose_native_wait")):
        raise ValueError("diagnostic load vector cores requires load and native wait diagnosis")
    budget = config.get("host_sync_diagnose_prefetch_poll_budget", 0)
    if type(budget) is not int or not 0 <= budget <= 10_000_000:
        raise ValueError("host_sync_diagnose_prefetch_poll_budget must be an integer in 0..10000000")
    if budget and not config.get("host_sync_diagnose_prefetch_report_wait", False):
        raise ValueError("bounded prefetch wait requires report completion guard")
    if not isinstance(config.get("host_sync_diagnose_prefetch_report_wait", False), bool):
        raise ValueError("host_sync_diagnose_prefetch_report_wait must be boolean")
    if not isinstance(config.get("host_sync_diagnose_scale_context", False), bool):
        raise ValueError("host_sync_diagnose_scale_context must be boolean")
    if config.get("host_sync_diagnose_scale_context") and not config.get("host_sync_diagnose_split_scale"):
        raise ValueError("scale context diagnosis requires split scale diagnosis")
    if not isinstance(config.get("host_sync_diagnose_split_scale", False), bool):
        raise ValueError("host_sync_diagnose_split_scale must be boolean")
    if config.get("host_sync_diagnose_split_scale") and not (
        config.get("host_sync_diagnose_native_wait") and config.get("host_sync_diagnose_load")
    ):
        raise ValueError("split scale diagnosis requires load and native wait diagnosis")
    if not isinstance(config.get("host_sync_diagnose_native_wait", False), bool):
        raise ValueError("host_sync_diagnose_native_wait must be boolean")
    for key, choices, default in (
        ("host_sync_scan_mode", ("all", "allocated"), "all"),
        ("host_sync_idle_probe", ("locked", "atomic"), "locked"),
        ("host_sync_idle_mode", ("yield", "spin"), "yield"),
    ):
        if config.get(key, default) not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    if not isinstance(config.get("host_sync_diagnose_load", False), bool):
        raise ValueError("host_sync_diagnose_load must be boolean")
    backend = config.get("host_sync_backend", "callback")
    if backend not in ("callback", "mte"):
        raise ValueError("host_sync_backend must be callback or mte")
    if backend == "callback":
        return
    if config.get("enable_multi_card") or config.get("h2d_backend", "torch") != "torch":
        raise ValueError("MTE host sync currently requires single-card offload and torch H2D")
    if not config.get("expert_offload"):
        raise ValueError("MTE host sync requires expert_offload")
    library = config.get("host_sync_library", "")
    if not isinstance(library, str) or not Path(library).is_absolute() or not Path(library).is_file():
        raise ValueError("host_sync_library must point to the built absolute libhost_sync_device.so path")


def describe_copy_object(value):
    """Metadata only: never read tensor values or synchronize the device."""
    result = {"type": type(value).__name__}
    for name in ("shape", "dtype", "device"):
        if hasattr(value, name):
            item = getattr(value, name)
            result[name] = list(item) if name == "shape" else str(item)
    for name in ("stride", "storage_offset", "is_contiguous", "data_ptr"):
        method = getattr(value, name, None)
        if callable(method):
            result[name] = method()
    if str(getattr(value, "device", "")) == "cpu" and callable(getattr(value, "is_pinned", None)):
        result["is_pinned"] = value.is_pinned()
    return result


def recent_acl_error(library):
    """Read thread-local Runtime detail immediately after failure, on that worker."""
    try:
        getter = library.aclGetRecentErrMsg
        getter.argtypes, getter.restype = [], ctypes.c_char_p
        message = getter()
        return message.decode("utf-8", errors="replace") if message else "<no recent ACL error>"
    except Exception as exc:
        return f"<ACL error detail unavailable: {exc}>"


def runtime_stream_identity(library, pointer):
    """Query Runtime ID, not the Python stream index or pointer address."""
    try:
        query = library.aclrtStreamGetId
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        query.restype = ctypes.c_int
        stream_id = ctypes.c_int32(-1)
        code = query(pointer, ctypes.byref(stream_id))
        return {"pointer": pointer, "code": code, "stream_id": stream_id.value if code == 0 else None}
    except Exception as exc:
        return {"pointer": pointer, "query_error": str(exc)}


def apply_load_vector_limit(library, pointer, cores):
    """Diagnostic stream-local resource request, bound to the calling worker."""
    vector_core_type = 1  # ACL_RT_DEV_RES_VECTOR_CORE
    setter = library.aclrtSetStreamResLimit
    setter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    setter.restype = ctypes.c_int
    bind = library.aclrtUseStreamResInCurrentThread
    bind.argtypes, bind.restype = [ctypes.c_void_p], ctypes.c_int
    code = setter(pointer, vector_core_type, cores)
    if code:
        raise RuntimeError(f"load vector limit set failed: {code}")
    code = bind(pointer)
    if code:
        raise RuntimeError(f"load vector limit thread binding failed: {code}")


def runtime_last_task(library):
    """Last task submitted by this thread; not proof of execution or stream ownership."""
    result = {"host_tid": threading.get_native_id()}
    try:
        query = library.aclrtGetThreadLastTaskId
        query.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_int
        task_id = ctypes.c_uint32()
        code = query(ctypes.byref(task_id))
        result.update(code=code, task_id=task_id.value if code == 0 else None)
    except Exception as exc:
        result["query_error"] = str(exc)
    return result


def describe_npu_tensor(torch_module, value):
    # Diagnostic metadata only; no cpu(), item(), tensor contents or synchronization.
    import torch_npu

    result = describe_copy_object(value)
    try:
        result["npu_format"] = torch_npu.get_npu_format(value)
    except Exception as exc:
        result["npu_format_error"] = str(exc)
    result["grad_enabled"] = torch_module.is_grad_enabled()
    result["inference_mode"] = torch_module.is_inference_mode_enabled()
    return result


def configure_mte_worker(session, config):
    """Configure opt-in worker policies before start; keep old defaults usable."""
    scan = getattr(config, "host_sync_scan_mode", "all")
    probe = getattr(config, "host_sync_idle_probe", "locked")
    idle = getattr(config, "host_sync_idle_mode", "yield")
    required = []
    if scan == "allocated":
        required.extend(("configure_scan_prefix", "expand_scan_prefix"))
    if probe != "locked":
        required.append("configure_idle_probe")
    if idle != "yield":
        required.append("configure_polling")
    if any(not callable(getattr(session, name, None)) for name in required):
        raise RuntimeError("Rebuild csrc/host_sync in this checkout for selected MTE worker policies")
    if scan == "allocated":
        session.configure_scan_prefix(1)  # slot 0: initialization handshake
    if probe != "locked":
        session.configure_idle_probe(probe)
    if idle != "yield":
        session.configure_polling(idle, False)
    return scan == "allocated"


class MteHostDispatcher:
    """One worker, fixed slots retained for the lifetime of captured graphs.

    A slot is allocated per captured callback, not per replay. Different graph
    sizes and prefetch/reactive callbacks do not overwrite one another's args.
    """

    SLOT_CAPACITY = 2048
    STATE_BYTES = 32
    STALL_SECONDS = 120
    WARMUP_MAX_POLLS = 1000000
    DIAGNOSTIC_SECONDS = 10
    DIAGNOSTIC_WAIT_TIMEOUT_MS = 10_000
    REPORT_WAIT_TIMEOUT_MS = 30_000
    WAIT_EXIT_SUMMARY_INTERVAL = 32

    def __init__(self, manager, library_path):
        # Optional runtime dependencies are loaded only for the opt-in backend.
        import torch
        import torch_npu
        import vllm_ascend_host_sync as hs

        if not torch_npu.__version__.startswith("2.10.") or hs.API_VERSION < 2:
            raise RuntimeError("MTE MoE integration requires torch_npu 2.10 and Host extension API >= 2")
        if not torch.npu.get_device_name().startswith("Ascend950"):
            raise RuntimeError("MTE MoE integration currently supports Ascend950 only")
        self.manager, self.torch = manager, torch
        self.library = ctypes.CDLL(library_path)
        self.publish = self.library.host_sync_publish_launch
        self.wait = self.library.host_sync_wait_strict_launch
        self.warmup_wait = self.library.host_sync_wait_launch
        for fn in (self.publish, self.wait):
            fn.argtypes, fn.restype = [ctypes.c_void_p] * 3, None
        self.warmup_wait.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64]
        self.warmup_wait.restype = None
        self.entries = [None] * self.SLOT_CAPACITY
        self.entry_streams = [None] * self.SLOT_CAPACITY
        self.report_wait_streams = set()
        self.report_wait_samples = 0
        self.next_slot = 1
        self.completed = 0
        self.load_diagnostic_batches = 0
        self.active_callback = None
        self.closed = False
        self.stopping = threading.Event()
        self.states = torch.zeros((self.SLOT_CAPACITY, 4), dtype=torch.int64, device="npu")
        self.wait_states_h = None
        self.wait_exit_counts = {"completed": 0, "budget": 0}
        if getattr(manager.offload_config, "host_sync_diagnose_prefetch_poll_budget", 0):
            self.wait_states_h = torch.empty((self.SLOT_CAPACITY, 4), dtype=torch.int64, pin_memory=True)
        torch.npu.synchronize()
        self.state_base = self.states.data_ptr()
        self.session = hs.Session(slots=self.SLOT_CAPACITY, backend="acl", device_id=torch.npu.current_device())
        info = self.session.info()
        if info["protocol_version"] != 1 or info["slot_stride"] != 1024 or info["done_offset"] != 512:
            raise RuntimeError("Unsupported MTE control-memory layout")
        self.stride, self.done_offset = info["slot_stride"], info["done_offset"]
        for slot in range(self.SLOT_CAPACITY):
            self.session.register_task(slot, lambda slot=slot: self._run(slot))
        self.allocated_scan = configure_mte_worker(self.session, manager.offload_config)
        self.session.start()
        self.alias = self.session.begin_device_access()
        print(
            f"[MOE-MTE WORKER] scan={getattr(manager.offload_config, 'host_sync_scan_mode', 'all')} "
            f"probe={getattr(manager.offload_config, 'host_sync_idle_probe', 'locked')} "
            f"idle={getattr(manager.offload_config, 'host_sync_idle_mode', 'yield')}",
            flush=True,
        )
        self.entries[0] = (lambda args: None, None)
        stream = torch.npu.current_stream()
        pointer = int(stream.npu_stream)
        self.publish(pointer, self.alias, self.state_base)
        self.warmup_wait(pointer, self.alias + self.done_offset, self.state_base, self.WARMUP_MAX_POLLS)
        stream.synchronize()
        if self.states[0].cpu().tolist()[:3] != [1, 1, 1]:
            self.session.end_device_access(stream_synchronized=True)
            self.session.stop()
            self.session.close()
            raise RuntimeError("MTE dispatcher initialization handshake failed")
        self.watchdog = threading.Thread(target=self._watch, name="moe-mte-watchdog", daemon=True)
        self.watchdog.start()
        atexit.register(self.close)

    @staticmethod
    def _fatal(message):
        print(
            f"[MOE-MTE FATAL] {message}; terminating engine to prevent stale-weight execution",
            file=sys.stderr,
            flush=True,
        )
        os._exit(70)

    def _run(self, slot):
        try:
            self.active_callback = (slot, threading.get_ident())
            fn, args = self.entries[slot]
            failures = getattr(self.manager, "_cb_failures", 0)
            fn(args)
            if getattr(self.manager, "_cb_failures", 0) != failures:
                raise RuntimeError("guarded offload callback reported a paging failure")
            if slot:
                self.completed += 1
                if self.completed == 1:
                    print("[MOE-MTE] first real offload callback completed", flush=True)
        except BaseException as exc:
            self._fatal(f"slot={slot}: {type(exc).__name__}: {exc}")
        finally:
            self.active_callback = None

    def diagnose_load(self, tasks, loads, layer, layer_idx):
        """Temporarily serialize real loads to locate the first blocked task.

        Only explicit diagnostic mode uses this path. It changes overlap and
        is not a performance measurement. Limit output to eight batches.
        """
        enabled = getattr(self.manager.offload_config, "host_sync_diagnose_load", False)
        cores = getattr(self.manager.offload_config, "host_sync_diagnose_load_vector_cores", 0)
        if cores and enabled and self.active_callback is not None and tasks:
            pointer = int(self.manager.load_stream.npu_stream)
            apply_load_vector_limit(ctypes.CDLL("libascendcl.so"), pointer, cores)
            if self.load_diagnostic_batches < 8:
                print(
                    f"[MOE-MTE VECTOR_LIMIT] requested={cores} stream={pointer}; "
                    "verify actual kernel coreDim in Runtime log",
                    flush=True,
                )
        if not enabled or self.active_callback is None or not tasks or self.load_diagnostic_batches >= 8:
            return False
        self.load_diagnostic_batches += 1
        batch = self.load_diagnostic_batches
        stream = self.manager.load_stream
        native_wait = getattr(self.manager.offload_config, "host_sync_diagnose_native_wait", False)
        raw_stream = None
        sync_native = None
        if native_wait:
            print(f"[MOE-MTE LOAD] batch={batch} STREAM_HANDLE_BEGIN", flush=True)
            raw_stream = int(stream.npu_stream)
            import torch_npu  # Lazy diagnostic import; never from the watchdog.

            current = int(torch_npu.npu.current_stream().npu_stream)
            slot = self.active_callback[0]
            print(
                f"[MOE-MTE LOAD] batch={batch} STREAM_HANDLE_READY "
                f"load={raw_stream} current={current} handoff={self.entry_streams[slot]} "
                f"slot={slot} tid={threading.get_native_id()}",
                flush=True,
            )
            library = ctypes.CDLL("libascendcl.so")
            identities = {"load": runtime_stream_identity(library, raw_stream)}
            for entry_slot in range(1, self.next_slot):
                pointer = self.entry_streams[entry_slot]
                if pointer is not None and str(pointer) not in identities:
                    fn, _ = self.entries[entry_slot]
                    identities[str(pointer)] = {
                        **runtime_stream_identity(library, pointer),
                        "example_slot": entry_slot,
                        "callback": getattr(fn, "__name__", type(fn).__name__),
                    }
            print("[MOE-MTE STREAM_MAP] " + json.dumps(identities), flush=True)
            sync_native = library.aclrtSynchronizeStreamWithTimeout
            sync_native.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            sync_native.restype = ctypes.c_int

        def checkpoint(label, operation):
            prefix = f"[MOE-MTE LOAD] batch={batch} layer={layer_idx} {label}"
            print(f"{prefix} BEGIN", flush=True)
            operation()
            print(f"{prefix} SUBMITTED", flush=True)
            if sync_native is not None:
                print(f"{prefix} LAST_TASK " + json.dumps(runtime_last_task(library)), flush=True)
                print(
                    f"{prefix} ACL_WAIT_BEGIN stream={raw_stream} timeout_ms={self.DIAGNOSTIC_WAIT_TIMEOUT_MS}",
                    flush=True,
                )
                result = sync_native(raw_stream, self.DIAGNOSTIC_WAIT_TIMEOUT_MS)
                print(f"{prefix} ACL_WAIT_RETURN code={result}", flush=True)
                if result != 0:
                    print("[MOE-MTE ACL_DETAIL] " + json.dumps(recent_acl_error(library)), flush=True)
                    self._dump_stall(self.active_callback[0], self.session.snapshot(self.active_callback[0]))
                    self._fatal(f"diagnostic native stream wait failed code={result}; no inference continuation")
                print(f"{prefix} TORCH_WAIT_BEGIN", flush=True)
            stream.synchronize()
            print(f"{prefix} COMPLETE", flush=True)

        checkpoint("prior_stream_work", lambda: None)
        transport = self.manager._get_h2d_transport()
        for index, task in enumerate(tasks):
            if native_wait:
                print(
                    "[MOE-MTE LAYOUT] "
                    + json.dumps(
                        {
                            "batch": batch,
                            "layer": layer_idx,
                            "index": index,
                            "name": task.name,
                            "bytes": task.nbytes,
                            "non_blocking": task.non_blocking,
                            "source": describe_copy_object(task.source),
                            "destination": describe_copy_object(task.destination),
                        }
                    ),
                    flush=True,
                )
            label = f"copy[{index}] name={task.name} bytes={task.nbytes}"
            if getattr(self.manager.offload_config, "host_sync_diagnose_split_scale", False) and (
                task.name.startswith(("w13_weight_scale[", "w2_weight_scale["))
            ):
                self._diagnose_split_scale(task, label, checkpoint)
            else:
                checkpoint(label, lambda task=task: transport.copy_batch([task]))
        for target_slot, _ in loads:
            checkpoint(
                f"scale_refresh slot={target_slot}",
                lambda target_slot=target_slot: self.manager._refresh_expert_fp32_scale(layer, target_slot),
            )
        return True

    def _diagnose_split_scale(self, task, label, checkpoint):
        # Diagnostic only: preserve the destination and logical values, but
        # introduce an explicit temporary and synchronization boundaries.
        # Passing this path does not prove the original copy path is repaired.
        print(f"[MOE-MTE LOAD] {label} TEMP_ALLOC_BEGIN", flush=True)
        temporary = self.torch.empty_like(task.destination, memory_format=self.torch.contiguous_format)
        print(f"[MOE-MTE LOAD] {label} TEMP_ALLOC_READY", flush=True)
        checkpoint(f"{label} H2D_CONTIGUOUS", lambda: temporary.copy_(task.source, non_blocking=task.non_blocking))
        if getattr(self.manager.offload_config, "host_sync_diagnose_scale_context", False):
            self._diagnose_scale_targets(task, temporary, label, checkpoint)
        checkpoint(f"{label} D2D_TARGET_LAYOUT", lambda: task.destination.copy_(temporary, non_blocking=True))
        # Retain temporary until both checkpoints finish; never publish done
        # or proceed to scale refresh after a failed checkpoint.

    def _diagnose_scale_targets(self, task, temporary, label, checkpoint):
        print(
            "[MOE-MTE SCALE_CONTEXT] "
            + json.dumps(
                {
                    "name": task.name,
                    "temporary": describe_npu_tensor(self.torch, temporary),
                    "original": describe_npu_tensor(self.torch, task.destination),
                }
            ),
            flush=True,
        )
        for kind, template, memory_format in (
            ("CONTIGUOUS", temporary, self.torch.contiguous_format),
            ("INDEPENDENT_STRIDED", task.destination, self.torch.preserve_format),
        ):
            print(f"[MOE-MTE LOAD] {label} {kind}_ALLOC_BEGIN", flush=True)
            probe = self.torch.empty_like(template, memory_format=memory_format)
            print(
                "[MOE-MTE PROBE_TARGET] "
                + json.dumps(
                    {
                        "kind": kind,
                        "metadata": describe_npu_tensor(self.torch, probe),
                    }
                ),
                flush=True,
            )
            if kind == "INDEPENDENT_STRIDED" and probe.stride() != task.destination.stride():
                raise RuntimeError("Independent scale probe did not preserve destination stride")
            checkpoint(f"{label} D2D_{kind}", lambda probe=probe: probe.copy_(temporary, non_blocking=True))
            # checkpoint finishes before probe can be released/reallocated.

    def _dump_stall(self, slot, state):
        # Only on a stalled request, never on the normal callback hot path.
        # Do not query NPU tensors or stream handles here: that could block
        # the watchdog on the same device work it is trying to diagnose.
        active = self.active_callback
        lines = [
            f"[MOE-MTE DIAGNOSTIC BEGIN] pending_slot={slot} state={state}",
            f"active_callback={active}; completed={self.completed}; captured_slots={self.next_slot - 1}",
        ]
        pending_entries = []
        for entry_slot in range(1, self.next_slot):
            snapshot = self.session.snapshot(entry_slot)
            if not isinstance(snapshot, dict):
                continue
            if snapshot.get("request_generation", 0) > snapshot.get("done_generation", 0):
                fn, args = self.entries[entry_slot]
                pending_entries.append(
                    {
                        "slot": entry_slot,
                        "stream_pointer": self.entry_streams[entry_slot],
                        "callback": getattr(fn, "__name__", type(fn).__name__),
                        "layer": args[3] if isinstance(args, tuple) and len(args) > 3 else "unknown",
                        "state": snapshot,
                    }
                )
        lines.append("[MOE-MTE PENDING] " + json.dumps(pending_entries))
        for selected in sorted({slot, active[0] if active else slot}):
            fn, args = self.entries[selected]
            layer = args[3] if isinstance(args, tuple) and len(args) > 3 else "unknown"
            lines.append(
                f"slot={selected} callback={getattr(fn, '__name__', type(fn).__name__)} "
                f"layer={layer} state={self.session.snapshot(selected)}"
            )
        frames = sys._current_frames()
        # The active callback and Python main thread distinguish a worker
        # blocked inside copy/sync from a request not yet picked up by worker.
        ids = [active[1]] if active else []
        main_id = threading.main_thread().ident
        if main_id not in ids:
            ids.append(main_id)
        for ident in ids:
            frame = frames.get(ident)
            lines.append(f"Python thread={ident}; active_worker={bool(active and ident == active[1])}")
            if frame is not None:
                lines.extend(line.rstrip() for line in traceback.format_stack(frame, limit=24))
            else:
                lines.append("No Python frame available; native wait cannot be resolved from this snapshot.")
        lines.append("[MOE-MTE DIAGNOSTIC END]")
        print("\n".join(lines), file=sys.stderr, flush=True)

    def launch(self, stream, callback, args):
        if self.closed or self.next_slot >= self.SLOT_CAPACITY:
            raise RuntimeError("MTE dispatcher closed or graph callback slot capacity exceeded")
        slot = self.next_slot
        self.entries[slot] = (callback, args)
        if self.allocated_scan:
            # Publish the initialized entry's range before any device request.
            # Older captured graphs keep their slots inside the growing prefix.
            self.session.expand_scan_prefix(slot + 1)
        self.next_slot += 1
        # torch_npu 2.10's npu_stream property calls NPUStream::stream(), which
        # drains its Host submission queue. Read it HERE, after preceding copy_
        # and event submissions; never cache it across model operations.
        pointer = int(stream.npu_stream)
        self.entry_streams[slot] = pointer
        request = self.alias + slot * self.stride
        state = self.state_base + slot * self.STATE_BYTES
        self.publish(pointer, request, state)
        if (
            getattr(self.manager.offload_config, "host_sync_diagnose_prefetch_report_wait", False)
            and stream == self.manager._prefetch_stream
        ):
            import torch_npu  # Optional diagnostic path only.

            budget = getattr(self.manager.offload_config, "host_sync_diagnose_prefetch_poll_budget", 0)
            if budget:
                # A report callback immediately after this kernel still guards
                # consumers until actual completion. Never use bounded wait alone.
                self.warmup_wait(pointer, request + self.done_offset, state, budget)
                # Capture the 32-byte device exit record on the same stream,
                # before the report callback. Never read device values from
                # the report thread, which could introduce another dependency.
                self.wait_states_h[slot].copy_(self.states[slot], non_blocking=True)
            if pointer not in self.report_wait_streams:
                torch_npu.npu._subscribe_report(stream)
                self.report_wait_streams.add(pointer)
                print(
                    "[MOE-MTE WAIT_CONTROL] prefetch=report_wait; request=MTE; worker=native; reactive=AIV; "
                    f"bounded_aiv_polls={budget}",
                    flush=True,
                )
            torch_npu.npu._launch_host_func(stream, self._report_wait_completion, slot)
        else:
            self.wait(pointer, request + self.done_offset, state)

    def _report_wait_completion(self, slot):
        # Control experiment: report thread only waits; the native worker still
        # executes the original callback and publishes done after H2D sync.
        # Session.wait releases GIL and its mutex while waiting, so the worker
        # can run. Failure must never release the graph to consume stale data.
        try:
            snapshot = self.session.snapshot(slot)
            generation = snapshot["request_generation"]
            if generation <= 0:
                raise RuntimeError("report wait ran before request publication")
            budget = getattr(self.manager.offload_config, "host_sync_diagnose_prefetch_poll_budget", 0)
            device_exit = None
            if budget:
                device_exit = self.wait_states_h[slot].tolist()  # Pinned CPU data, D2H precedes callback.
                if device_exit[0] != generation or device_exit[1] not in (1, 3):
                    raise RuntimeError(f"unexpected AIV exit record: {device_exit}; generation={generation}")
                if device_exit[1] == 3 and device_exit[3] != budget:
                    raise RuntimeError(f"AIV poll count differs from budget: {device_exit}")
                self.wait_exit_counts["completed" if device_exit[1] == 1 else "budget"] += 1
            record = budget and self.report_wait_samples < 8
            begin = time.perf_counter_ns()
            if record:
                self.report_wait_samples += 1
                print(
                    "[MOE-MTE AIV_EXIT] "
                    + json.dumps(
                        {
                            "slot": slot,
                            "poll_budget": budget,
                            "generation": generation,
                            "done_at_report_entry": snapshot.get("done_generation"),
                            "error_at_report_entry": snapshot.get("error_generation"),
                            "active_worker": self.active_callback,
                            "device_exit": device_exit,
                        }
                    ),
                    flush=True,
                )
            result = self.session.wait(slot, generation, timeout_ms=self.REPORT_WAIT_TIMEOUT_MS)
            if result["done_generation"] != generation:
                raise RuntimeError("report wait returned without matching completion")
            if record:
                print(
                    "[MOE-MTE AFTER_AIV_EXIT] "
                    + json.dumps(
                        {
                            "slot": slot,
                            "generation": generation,
                            "done": result["done_generation"],
                            "report_guard_us": (time.perf_counter_ns() - begin) / 1000,
                        }
                    ),
                    flush=True,
                )
            if budget and sum(self.wait_exit_counts.values()) % self.WAIT_EXIT_SUMMARY_INTERVAL == 0:
                print("[MOE-MTE EXIT_COUNTS] " + json.dumps(self.wait_exit_counts), flush=True)
        except BaseException as exc:
            self._fatal(f"report wait slot={slot}: {exc}")

    def _watch(self):
        pending = {}
        diagnosed = False
        while not self.stopping.wait(1):
            try:
                now = time.monotonic()
                for slot in range(1, self.next_slot):
                    state = self.session.snapshot(slot)
                    generation = state["request_generation"]
                    if generation == 0 or state["done_generation"] == generation:
                        pending.pop(slot, None)
                        continue
                    if state["error_generation"] == generation:
                        self._fatal(f"slot={slot}: {state['error']}")
                    previous = pending.get(slot)
                    if previous is None or previous[0] != generation:
                        pending[slot] = (generation, now)
                    elif now - previous[1] > self.STALL_SECONDS:
                        self._fatal(f"slot={slot} generation={generation} exceeded {self.STALL_SECONDS}s")
                    elif not diagnosed and now - previous[1] > self.DIAGNOSTIC_SECONDS:
                        self._dump_stall(slot, state)
                        diagnosed = True
            except BaseException as exc:
                self._fatal(f"watchdog failed: {exc}")

    def close(self):
        if self.closed:
            return
        # Keep the watchdog active until all submitted graph work has drained.
        self.torch.npu.synchronize()
        self.stopping.set()
        self.watchdog.join()
        self.session.end_device_access(stream_synchronized=True)
        self.session.stop()
        self.session.close()
        self.closed = True
        atexit.unregister(self.close)
