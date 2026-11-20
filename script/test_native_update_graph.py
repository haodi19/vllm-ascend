# SPDX-License-Identifier: Apache-2.0
"""Isolated real NPU D2H/publish/native update/strict wait/mapping regression."""

import argparse
import ctypes
import os
import sys
import time
import traceback
from pathlib import Path


def run(library_path):
    import _vllm_ascend_lrc as native
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    library = ctypes.CDLL(str(library_path.resolve()))
    publish = library.host_sync_publish_launch
    wait = library.host_sync_wait_strict_launch
    for fn in (publish, wait):
        fn.argtypes = [ctypes.c_void_p] * 3
        fn.restype = None
    acl = ctypes.CDLL("libascendcl.so")
    copy = acl.aclrtMemcpyAsync
    copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    copy.restype = ctypes.c_int
    compute, load = torch.npu.Stream(), torch.npu.Stream()
    source = torch.empty((4, 2 * 1024 * 1024), dtype=torch.uint8, pin_memory=True)
    for eid in range(4):
        source[eid].fill_(eid + 11)
    target = torch.full((2, source.shape[1]), 11, dtype=torch.uint8, device="npu")
    target[1].fill_(12)
    scale_source = torch.empty_strided((4, 16, 64, 2), (2048, 2, 32, 1), dtype=torch.uint8, pin_memory=True)
    for eid in range(4):
        scale_source[eid].copy_((torch.arange(2048).reshape(16, 64, 2) + eid).to(torch.uint8))
    scale_target = torch.empty_strided((2, 16, 64, 2), (2048, 2, 32, 1), dtype=torch.uint8, device="npu")
    scale_target.copy_(scale_source[:2])
    ids_host = torch.empty((1, 1), dtype=torch.int64, pin_memory=True)
    ids_device = torch.tensor([[2]], dtype=torch.int64, device="npu")
    map_host = torch.tensor([0, 1, -1, -1], dtype=torch.int32).pin_memory()
    map_device = map_host.to("npu")
    state = torch.zeros(4, dtype=torch.int64, device="npu")
    torch.npu.synchronize()
    if scale_target.stride() != scale_source.stride():
        raise RuntimeError("NPU changed requested scale strides")
    table = native.NativeTransfers(
        [[source[e].data_ptr(), scale_source[e].data_ptr()] for e in range(4)],
        [[target[s].data_ptr(), scale_target[s].data_ptr()] for s in range(2)],
        [source.shape[1], 2048],
        int(load.npu_stream),
        "",
        False,
    )
    policy = native.Policy(1, 4, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
    task = native.native_update_task(
        policy,
        table,
        0,
        ids_host.numpy(),
        None,
        map_host.numpy(),
        False,
        1,
        (source, target, scale_source, scale_target, ids_host, map_host),
    )
    session = hs.Session(slots=1, backend="acl", device_id=0)
    info = session.info()
    if (info["protocol_version"], info["request_offset"], info["done_offset"], info["error_offset"]) != (
        1,
        0,
        512,
        520,
    ):
        raise RuntimeError("unknown handshake protocol layout")

    def forbidden_callback():
        raise RuntimeError("native graph called Python fallback")

    session.register_task(0, forbidden_callback)
    session.install_native_task(0, task)
    session.start()
    alias = session.begin_device_access()
    done = ctypes.c_uint64.from_address(info["host_address"] + info["done_offset"])
    error = ctypes.c_uint64.from_address(info["host_address"] + info["error_offset"])

    def transfer(dst, src, kind):
        size = src.numel() * src.element_size()
        code = copy(dst.data_ptr(), size, src.data_ptr(), size, kind, int(compute.npu_stream))
        if code:
            raise RuntimeError(f"test raw copy failed: {code}")

    def enqueue():
        transfer(ids_host, ids_device, 2)
        transfer(map_host, map_device, 2)
        publish(int(compute.npu_stream), alias, state.data_ptr())
        wait(int(compute.npu_stream), alias + info["done_offset"], state.data_ptr())
        transfer(map_device, map_host, 1)

    def validate(generation, eid):
        if error.value or done.value != generation or state.cpu().tolist()[1:3] != [1, generation]:
            raise RuntimeError("native graph failed strict completion")
        torch.testing.assert_close(map_device.cpu(), map_host, rtol=0, atol=0)
        for resident in range(4):
            slot = int(map_host[resident])
            if slot >= 0:
                torch.testing.assert_close(target[slot].cpu(), source[resident], rtol=0, atol=0)
                torch.testing.assert_close(scale_target[slot].cpu(), scale_source[resident], rtol=0, atol=0)
        if int(map_host[eid]) < 0:
            raise RuntimeError("requested expert was not loaded")

    enqueue()
    compute.synchronize()
    validate(1, 2)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=compute):
        enqueue()
    compute.synchronize()
    if done.value != 1:
        raise RuntimeError("capture unexpectedly executed native update")
    completions_under_gil = 0
    for generation, eid in enumerate((3, 3, 0, 1, 2, 3, 0, 1), start=2):
        ids_device.fill_(eid)
        torch.npu.synchronize()
        with torch.npu.stream(compute):
            graph.replay()
        previous_interval = sys.getswitchinterval()
        try:
            sys.setswitchinterval(60.0)
            before = done.value
            deadline = time.perf_counter() + 35
            while done.value != generation and not error.value and time.perf_counter() < deadline:
                pass  # Python holds GIL; no sleep or extension wait.
            if done.value != generation or error.value:
                raise RuntimeError("native task did not complete while Python held the GIL")
            completions_under_gil += before < generation
        finally:
            sys.setswitchinterval(previous_interval)
        compute.synchronize()
        validate(generation, eid)
    graph.reset()
    session.end_device_access(stream_synchronized=True)
    session.stop()
    session.close()
    if not completions_under_gil:
        raise RuntimeError("all tasks finished before GIL bracket; GIL independence test inconclusive")
    print(
        f"PASS native strict graph: replays=8; completions_under_gil={completions_under_gil}; values/map match",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.library)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Do not free memory still reachable from a failed strict device wait.
        # This isolated process is also guarded by the runner's hard deadline.
        os._exit(1)


if __name__ == "__main__":
    main()
