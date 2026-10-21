# SPDX-License-Identifier: Apache-2.0
"""Isolated stage 2 NPU experiment. Never import this into a serving process."""

import argparse
import ctypes
import json
import os
import threading
import traceback
from pathlib import Path


def check_result(result, generation, expected_code):
    if len(result) != 4 or result[0] != generation or result[1] != expected_code:
        raise RuntimeError(f"Unexpected NPU result: {result}; expected generation={generation}, code={expected_code}")
    if expected_code == 1 and result[2] != generation:
        raise RuntimeError(f"NPU did not observe the matching Host completion: {result}")
    if result[3] < 1:
        raise RuntimeError(f"NPU did not execute a completion poll: {result}")


def run_probe(args):
    # Lazy imports keep CPU checks/help independent of the NPU runtime.
    import torch
    import torch_npu
    import vllm_ascend_host_sync as hs

    if hs.API_VERSION < 2:
        raise RuntimeError("Rebuild csrc/host_sync: stage 2 requires extension API >= 2")
    torch.npu.set_device(0)
    name = torch.npu.get_device_name(0)
    if not name.startswith("Ascend950"):
        raise RuntimeError(f"Stage 2 currently targets Ascend950, found {name}")
    print(f"device={name}; torch_npu={torch_npu.__version__}; case={args.case}", flush=True)
    library = ctypes.CDLL(str(args.library.resolve()))
    publish = library.host_sync_publish_launch
    publish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    publish.restype = None
    wait = library.host_sync_wait_launch
    wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    wait.restype = None

    # No Torch operations are queued between raw launches. Drain allocation /
    # initialization before using the runtime stream directly. Graph integration
    # and mixed Torch/native task-queue ordering are intentionally not tested yet.
    stream = torch.npu.current_stream()
    state = torch.zeros(4, dtype=torch.int64, device="npu:0")
    stream.synchronize()
    stream_pointer = int(stream.npu_stream)
    session = hs.Session(slots=1, backend="acl", device_id=0)
    info = session.info()
    expected_layout = {
        "protocol_version": 1,
        "slot_stride": 1024,
        "request_offset": 0,
        "done_offset": 512,
        "error_offset": 520,
    }
    for key, expected in expected_layout.items():
        if info[key] != expected:
            raise RuntimeError(f"Unsupported protocol layout {key}={info[key]}, expected {expected}")
    calls = []
    release_callback = threading.Event()

    def callback():
        generation = session.snapshot(0)["request_generation"]
        if generation != len(calls) + 1:
            raise RuntimeError(f"Host observed out-of-order generation {generation}")
        calls.append(generation)
        if args.case == "callback-error":
            raise RuntimeError("deliberate stage 2 callback failure")
        if args.case == "poll-limit" and not release_callback.wait(30):
            raise RuntimeError("test did not release the gated callback")

    session.register_task(0, callback)
    session.start()
    alias = session.begin_device_access()
    for forbidden in (
        session.stop,
        session.close,
        session.begin_device_access,
        lambda: session.submit_for_test(0, 1),
        lambda: session.end_device_access(stream_synchronized=False),
    ):
        try:
            forbidden()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Device access lease failed to prevent an unsafe operation")
    # From here on failure exits this isolated process without freeing memory
    # potentially still used by the device. The parent enforces a hard deadline.
    request = alias + info["request_offset"]
    completion = alias + info["done_offset"]
    rounds = args.rounds if args.case == "normal" else 1
    code = {"normal": 1, "callback-error": 2, "poll-limit": 3}[args.case]
    poll_counts = []
    for generation in range(1, rounds + 1):
        publish(stream_pointer, request, state.data_ptr())
        wait(stream_pointer, completion, state.data_ptr(), 8 if args.case == "poll-limit" else args.max_polls)
        stream.synchronize()
        result = state.cpu().tolist()
        check_result(result, generation, code)
        poll_counts.append(result[3])
    # The device no longer accesses the Host allocation. Release a gated Host
    # callback before asking stop() to join it.
    release_callback.set()
    session.end_device_access(stream_synchronized=True)
    try:
        session.wait(0, rounds)
    except RuntimeError as exc:
        if args.case != "callback-error" or "deliberate stage 2 callback failure" not in str(exc):
            raise
    session.stop()
    snapshot = session.snapshot(0)
    if calls != list(range(1, rounds + 1)):
        raise RuntimeError(f"Callback count/order mismatch: expected {rounds}, observed {calls}")
    if args.case == "callback-error":
        if snapshot["error_generation"] != 1 or snapshot["done_generation"] == 1:
            raise RuntimeError(f"Callback error incorrectly published as success: {snapshot}")
    elif snapshot["done_generation"] != rounds:
        raise RuntimeError(f"Host completion mismatch: {snapshot}")
    session.close()
    print(
        "RESULT "
        + json.dumps(
            {
                "case": args.case,
                "result": "PASS",
                "npu_generated_requests": rounds,
                "host_callbacks": len(calls),
                "last_npu_result": result,
                "poll_count_min": min(poll_counts),
                "poll_count_max": max(poll_counts),
                "memory_released_after_device_sync": session.info()["state"] == "closed",
            }
        ),
        flush=True,
    )
    print("Poll counts are not time measurements. Graph replay/MoE/performance: NOT TESTED.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--case", choices=("normal", "callback-error", "poll-limit"), default="normal")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-polls", type=int, default=100000)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10000 or not 1 <= args.max_polls <= 10000000:
        parser.error("rounds must be 1..10000 and max-polls must be 1..10000000")
    try:
        run_probe(args)
    except BaseException:
        traceback.print_exc()
        print("FAIL: isolated device probe; no success or performance claim.", flush=True)
        # Do not run atexit handlers that could free registered buffers while an
        # unsuccessful device stream still holds their addresses.
        os._exit(1)


if __name__ == "__main__":
    main()
