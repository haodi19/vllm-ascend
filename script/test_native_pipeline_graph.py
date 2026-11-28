# SPDX-License-Identifier: Apache-2.0
"""NPU acceptance gate: early prediction, queued DMA, strict ready and replay.

No model. Every replay checks expert bytes, non-contiguous scale physical
layout, mapping and device completion, including hits and zero-load prefetch.
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

from moe_host_affinity import resolve_host_cpus


def run(library_path, host_cpus="none"):
    import _vllm_ascend_lrc as native
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    library = ctypes.CDLL(str(library_path.resolve()))
    publish = library.host_sync_publish_launch
    wait = library.host_sync_wait_strict_launch
    for fn in (publish, wait):
        fn.argtypes, fn.restype = [ctypes.c_void_p] * 3, None
    compute, prefetch, load = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
    states = torch.zeros((2, 4), dtype=torch.int64, device="npu")
    ready = torch.zeros_like(states)
    completed = torch.zeros((2, 4), dtype=torch.int64, pin_memory=True)
    routes = torch.tensor([[2]], dtype=torch.int64, device="npu")
    policy = native.Policy(2, 4, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
    cpus = resolve_host_cpus(host_cpus)
    engine = native.NativePipeline(str(library_path.resolve()), False, cpus[1] if cpus else -1)
    session = hs.Session(slots=2, backend="acl", device_id=0)
    if cpus:
        session.configure_worker_cpu(cpus[0])
    info = session.info()
    channels, owners = [], []
    for index in range(2):
        source = torch.empty((4, 2 * 1024 * 1024), dtype=torch.uint8, pin_memory=True)
        for eid in range(4):
            source[eid].fill_(eid + index * 10)
        target = source[:2].to("npu")
        scales = torch.empty_strided((4, 16, 64, 2), (2048, 2, 32, 1), dtype=torch.uint8, pin_memory=True)
        for eid in range(4):
            scales[eid].copy_((torch.arange(2048).reshape(16, 64, 2) + eid).to(torch.uint8))
        device_scales = torch.empty_strided((2, 16, 64, 2), (2048, 2, 32, 1), dtype=torch.uint8, device="npu")
        device_scales.copy_(scales[:2])
        if device_scales.stride() != scales.stride():
            raise RuntimeError("NPU changed requested scale strides")
        ids = torch.empty((1, 1), dtype=torch.int64, pin_memory=True)
        mapping = torch.tensor([0, 1, -1, -1], dtype=torch.int32).pin_memory()
        device_map = mapping.to("npu")
        table = native.NativeTransfers(
            [[source[e].data_ptr(), scales[e].data_ptr()] for e in range(4)],
            [[target[s].data_ptr(), device_scales[s].data_ptr()] for s in range(2)],
            [source.shape[1], 2048],
            int(load.npu_stream),
            "",
            False,
        )
        old = native.native_update_task(
            policy,
            table,
            index,
            ids.numpy(),
            None,
            mapping.numpy(),
            bool(index),
            1,
            (source, target, scales, device_scales, ids, mapping),
        )
        task, channel = native.native_pipeline_task(
            old,
            engine,
            info["host_address"] + index * info["slot_stride"],
            ready[index].data_ptr(),
            completed[index].data_ptr(),
            device_map.data_ptr(),
            0 if index else -1,
        )
        if channel != index:
            raise RuntimeError("unexpected pipeline channel order")
        session.register_task(index, lambda: None)
        session.install_native_task(index, task)
        channels.append((ids, mapping, device_map, source, target, scales, device_scales))
        owners.extend((old, task, table))
    torch.npu.synchronize()
    session.start()
    if cpus:
        assert session.worker_info()["worker_affinity"] == [cpus[0]]
        assert engine.worker_info()["worker_affinity"] == [cpus[1]]
        print(f"PIPELINE_AFFINITY=PASS; planner={cpus[0]}; DMA={cpus[1]}", flush=True)
    alias = session.begin_device_access()
    observations = []

    def consume(index):
        _, _, device_map, _, target, _, device_scales = channels[index]
        # Snapshot on the consumer stream immediately after its wait. Checking
        # only final buffers after a global sync would miss an early release.
        observations.append((device_map.clone(), target.clone(), device_scales.clone()))

    def enqueue():
        ids, mapping, device_map, *_ = channels[0]
        ids.copy_(routes, non_blocking=True)
        mapping.copy_(device_map, non_blocking=True)
        publish(int(compute.npu_stream), alias, states[0].data_ptr())
        # Stand-in for independent next-layer prediction while demand is live.
        predicted = routes + 0
        ids, mapping, device_map, *_ = channels[1]
        ids.copy_(predicted, non_blocking=True)
        mapping.copy_(device_map, non_blocking=True)
        inputs_ready = compute.record_event()
        with torch.npu.stream(prefetch):
            prefetch.wait_event(inputs_ready)
            publish(int(prefetch.npu_stream), alias + info["slot_stride"], states[1].data_ptr())
        wait(int(compute.npu_stream), ready[0].data_ptr(), states[0].data_ptr())
        consume(0)
        consumer = compute.record_event()
        with torch.npu.stream(prefetch):
            prefetch.wait_event(consumer)
            wait(int(prefetch.npu_stream), ready[1].data_ptr(), states[1].data_ptr())
            finished = prefetch.record_event()
        compute.wait_event(finished)
        consume(1)
        return inputs_ready, consumer, finished

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=compute):
        events = enqueue()
    compute.synchronize()
    if completed.any():
        raise RuntimeError("capture unexpectedly executed DMA")
    for generation, eid in enumerate((2, 3, 3, 0, 0, 1, 2, 3), 1):
        routes.fill_(eid)
        torch.npu.synchronize()
        with torch.npu.stream(compute):
            graph.replay()
        flags = [ctypes.c_uint64.from_address(completed[i].data_ptr()) for i in range(2)]
        previous = sys.getswitchinterval()
        try:
            sys.setswitchinterval(60)
            deadline = time.monotonic() + 30
            while any(flag.value != generation for flag in flags) and time.monotonic() < deadline:
                pass
            if any(flag.value != generation for flag in flags):
                raise RuntimeError("pipeline stalled while Python held GIL; do not start model")
        finally:
            sys.setswitchinterval(previous)
        compute.synchronize()
        engine.check_progress(30)
        for index, (_, mapping, device_map, source, target, scales, device_scales) in enumerate(channels):
            if states[index].cpu().tolist()[:3] != [generation, 1, generation]:
                raise RuntimeError("device consumer did not observe matching completion generation")
            torch.testing.assert_close(device_map.cpu(), mapping, rtol=0, atol=0)
            observed_map, observed_weights, observed_scales = observations[index]
            torch.testing.assert_close(observed_map.cpu(), mapping, rtol=0, atol=0)
            for expert, slot in enumerate(mapping.tolist()):
                if slot >= 0:
                    torch.testing.assert_close(target[slot].cpu(), source[expert], rtol=0, atol=0)
                    torch.testing.assert_close(device_scales[slot].cpu(), scales[expert], rtol=0, atol=0)
                    torch.testing.assert_close(observed_weights[slot].cpu(), source[expert], rtol=0, atol=0)
                    torch.testing.assert_close(observed_scales[slot].cpu(), scales[expert], rtol=0, atol=0)
            if mapping[eid] < 0:
                raise RuntimeError("missing requested expert after strict completion")
    if engine.submitted_count() != 16:
        raise RuntimeError("missing/duplicate pipeline jobs")
    torch.npu.synchronize()
    engine.close()
    session.end_device_access(stream_synchronized=True)
    session.stop()
    session.close()
    del events, graph, owners
    print("PIPELINE_GRAPH=PASS; replay=8; jobs=16; bytes/scales/mapping/strict-generation/GIL=PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--host-cpus", default="none")
    args = parser.parse_args()
    run(args.library, args.host_cpus)
