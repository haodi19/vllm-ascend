# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-card graph callback dispatch through registered MTE signals."""

import atexit
import ctypes
import os
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path


def dense_strides(shape, strides):
    expected = 1
    for stride, size in sorted((stride, size) for size, stride in zip(shape, strides) if size > 1):
        if stride != expected:
            return False
        expected *= size
    return all(size > 0 for size in shape)


def storage_quant_copy(task, get_format, allocate_host=None, stage=None):
    """Match Tensor.copy_ semantics for dense ND views, packing on CPU only."""
    source, destination = task.source, task.destination
    mark = stage if stage is not None else lambda name: nullcontext()
    with mark("quant_get_format"):
        dst_format = get_format(destination)
    if (
        source.device.type != "cpu"
        or destination.device.type != "npu"
        or source.dtype != destination.dtype
        or tuple(source.shape) != tuple(destination.shape)
        or not dense_strides(destination.shape, destination.stride())
        or dst_format != 2
        or source.numel() * source.element_size() != task.nbytes
        or destination.numel() * destination.element_size() != task.nbytes
    ):
        raise RuntimeError(
            f"MTE quant storage copy requires matching dtype/shape and dense CPU->NPU ND layout: "
            f"{task.name}; src_dtype={source.dtype}; dst_dtype={destination.dtype}; "
            f"src_shape={tuple(source.shape)}; dst_shape={tuple(destination.shape)}; "
            f"src_stride={source.stride()}; dst_stride={destination.stride()}; dst_format={dst_format}"
        )

    if not dense_strides(source.shape, source.stride()) or any(
        size > 1 and src_stride != dst_stride
        for size, src_stride, dst_stride in zip(source.shape, source.stride(), destination.stride())
    ):
        if allocate_host is None:
            # Optional dependency; CPU packing never launches a device kernel.
            import torch

            allocate_host = torch.empty_strided
        with mark("quant_allocate"):
            packed = allocate_host(
                tuple(destination.shape), tuple(destination.stride()), dtype=source.dtype, device="cpu", pin_memory=True
            )
        with mark("quant_pack"):
            packed.copy_(source)
        source = packed

    def byte_view(tensor):
        start = tensor.storage_offset() * tensor.element_size()
        storage = tensor.untyped_storage()
        if start + task.nbytes > storage.nbytes():
            raise RuntimeError(f"MTE quant storage copy exceeds storage: {task.name}")
        return storage[start : start + task.nbytes]

    with mark("quant_source_view"):
        source_view = byte_view(source)
    with mark("quant_destination_view"):
        destination_view = byte_view(destination)
    with mark("quant_task_replace"):
        result = replace(task, source=source_view, destination=destination_view)
    return result


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
        self.next_slot = 1
        self.completed = 0
        self.load_diagnostic_batches = 0
        self.active_callback = None
        self.closed = False
        self.stopping = threading.Event()
        self.states = torch.zeros((self.SLOT_CAPACITY, 4), dtype=torch.int64, device="npu")
        torch.npu.synchronize()
        self.state_base = self.states.data_ptr()
        self.session = hs.Session(slots=self.SLOT_CAPACITY, backend="acl", device_id=torch.npu.current_device())
        info = self.session.info()
        if info["protocol_version"] != 1 or info["slot_stride"] != 1024 or info["done_offset"] != 512:
            raise RuntimeError("Unsupported MTE control-memory layout")
        self.stride, self.done_offset = info["slot_stride"], info["done_offset"]
        for slot in range(self.SLOT_CAPACITY):
            self.session.register_task(slot, lambda slot=slot: self._run(slot))
        self.session.start()
        self.alias = self.session.begin_device_access()
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
        if not enabled or self.active_callback is None or not tasks or self.load_diagnostic_batches >= 8:
            return False
        self.load_diagnostic_batches += 1
        batch = self.load_diagnostic_batches
        stream = self.manager.load_stream

        def checkpoint(label, operation):
            prefix = f"[MOE-MTE LOAD] batch={batch} layer={layer_idx} {label}"
            print(f"{prefix} BEGIN", flush=True)
            operation()
            print(f"{prefix} SUBMITTED", flush=True)
            stream.synchronize()
            print(f"{prefix} COMPLETE", flush=True)

        checkpoint("prior_stream_work", lambda: None)
        transport = self.manager._get_h2d_transport()
        for index, task in enumerate(tasks):
            checkpoint(
                f"copy[{index}] name={task.name} bytes={task.nbytes}", lambda task=task: transport.copy_batch([task])
            )
        for target_slot, _ in loads:
            checkpoint(
                f"scale_refresh slot={target_slot}",
                lambda target_slot=target_slot: self.manager._refresh_expert_fp32_scale(layer, target_slot),
            )
        return True

    def _dump_stall(self, slot, state):
        # Only on a stalled request, never on the normal callback hot path.
        # Do not query NPU tensors or stream handles here: that could block
        # the watchdog on the same device work it is trying to diagnose.
        active = self.active_callback
        lines = [
            f"[MOE-MTE DIAGNOSTIC BEGIN] pending_slot={slot} state={state}",
            f"active_callback={active}; completed={self.completed}; captured_slots={self.next_slot - 1}",
        ]
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
        self.next_slot += 1
        # torch_npu 2.10's npu_stream property calls NPUStream::stream(), which
        # drains its Host submission queue. Read it HERE, after preceding copy_
        # and event submissions; never cache it across model operations.
        pointer = int(stream.npu_stream)
        request = self.alias + slot * self.stride
        state = self.state_base + slot * self.STATE_BYTES
        self.publish(pointer, request, state)
        self.wait(pointer, request + self.done_offset, state)

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
