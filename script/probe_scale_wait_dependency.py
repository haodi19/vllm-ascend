# SPDX-License-Identifier: Apache-2.0
"""隔离真实 scale 布局的 D2D 与 AIV 等待依赖；不加载模型，不测性能。"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def capture_exchange(npu, compute, handoff, launch, fork_join):
    """Record the model's ready -> prefetch -> done -> compute join edges."""
    if not fork_join:
        launch()
        return ()
    ready, done = npu.Event(), npu.Event()
    compute.record_event(ready)
    with npu.stream(handoff):
        handoff.wait_event(ready)
        launch()
        handoff.record_event(done)
    compute.wait_event(done)
    return ready, done


def child(args):
    # Optional device dependencies belong only to the isolated child.
    import torch
    import torch_npu
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    print(f"DEVICE={torch.npu.get_device_name()}; torch_npu={torch_npu.__version__}", flush=True)
    library = ctypes.CDLL(str(args.library))
    publish = library.host_sync_publish_launch
    publish.argtypes, publish.restype = [ctypes.c_void_p] * 3, None
    wait = library.host_sync_wait_launch
    wait.argtypes, wait.restype = [ctypes.c_void_p] * 3 + [ctypes.c_uint64], None
    acl = ctypes.CDLL("libascendcl.so")
    sync = acl.aclrtSynchronizeStreamWithTimeout
    sync.argtypes, sync.restype = [ctypes.c_void_p, ctypes.c_int32], ctypes.c_int
    handoff = torch.npu.Stream()
    compute = torch.npu.Stream()
    capture_stream = compute if args.topology == "fork_join" else handoff
    load = torch_npu.npu.SyncLaunchStream()
    state = torch.zeros(4, dtype=torch.int64, device="npu")
    # Same shape, stride and element offset as the failed layer-1 scale slot.
    pool = torch.zeros((24, 4096, 64, 2), dtype=torch.uint8, device="npu")
    target = pool.permute(0, 2, 1, 3)[20]
    source = torch.arange(64 * 4096 * 2, dtype=torch.int64).to(torch.uint8).reshape(64, 4096, 2)
    source_device = source.to("npu")
    torch.npu.synchronize()
    handoff_pointer = int(handoff.npu_stream)
    timings = []
    errors = []
    session = hs.Session(slots=1, backend="acl", device_id=0)

    def copy_scale():
        try:
            with torch_npu.npu.stream(load):
                pointer = int(load.npu_stream)
                begin = time.perf_counter_ns()
                print("WORKER D2D_BEGIN", flush=True)
                target.copy_(source_device, non_blocking=True)
                submitted = time.perf_counter_ns()
                print("WORKER D2D_SUBMITTED", flush=True)
                code = sync(pointer, 10_000)
                end = time.perf_counter_ns()
                print(f"WORKER D2D_ACL_RETURN code={code}", flush=True)
                if code:
                    raise RuntimeError(f"load stream ACL wait returned {code}")
                load.synchronize()
                timings.append({"submit_us": (submitted - begin) / 1000, "wait_us": (end - submitted) / 1000})
        except BaseException as exc:
            errors.append(repr(exc))
            raise

    session.register_task(0, copy_scale)
    session.start()
    alias = session.begin_device_access()
    completion = alias + session.info()["done_offset"]
    # Initialize launch and copy paths before capture, without overlapping wait.
    publish(handoff_pointer, alias, state.data_ptr())
    handoff.synchronize()
    session.wait(0, 1, timeout_ms=15_000)
    wait(handoff_pointer, completion, state.data_ptr(), 100_000)
    handoff.synchronize()
    graph = torch.npu.NPUGraph()

    def launch():
        # Flush the ready-event submission before raw kernel launches, just
        # like MteHostDispatcher.launch in the real prefetch path.
        pointer = int(handoff.npu_stream)
        publish(pointer, alias, state.data_ptr())
        if args.polls:
            # Bounded exit is safe ONLY in this isolated graph: there are no
            # expert/model consumers after wait. Never use it in MoE inference.
            wait(pointer, completion, state.data_ptr(), args.polls)

    with torch.npu.graph(graph, stream=capture_stream):
        events = capture_exchange(torch.npu, compute, handoff, launch, args.topology == "fork_join")
    capture_stream.synchronize()
    for generation in range(2, 5):
        target.zero_()
        torch.npu.synchronize()
        with torch.npu.stream(capture_stream):
            graph.replay()
        code = sync(int(capture_stream.npu_stream), 15_000)
        if code:
            raise RuntimeError(f"handoff ACL wait returned {code}; worker errors={errors}")
        session.wait(0, generation, timeout_ms=15_000)
        load.synchronize()
        result = state.cpu().tolist()
        if not torch.equal(target.cpu(), source):
            raise RuntimeError("D2D value mismatch")
        if torch.count_nonzero(pool[19].cpu()).item() or torch.count_nonzero(pool[21].cpu()).item():
            raise RuntimeError("Adjacent slot overwritten")
        print(
            "RESULT "
            + json.dumps(
                {
                    "topology": args.topology,
                    "polls": args.polls,
                    "round": generation,
                    "state": result,
                    "worker": timings[-1],
                    "values_match": True,
                }
            ),
            flush=True,
        )
    graph.reset()
    del events  # Retain captured events through every replay and graph reset.
    session.end_device_access(stream_synchronized=True)
    session.stop()
    session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--polls", type=int, default=0)
    parser.add_argument("--topology", choices=("independent", "fork_join"), default="independent")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.npu < 0:
        parser.error("NPU must be nonnegative")
    if args.child:
        try:
            child(args)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Failed device work can still hold registered memory addresses.
            os._exit(1)
        return 0
    root = Path(__file__).resolve().parents[1]
    if args.library is None:
        candidates = list((root / "build/host_sync_device_stage3").glob("run_*/lib/libhost_sync_device.so"))
        if not candidates:
            parser.error("No built MTE library; supply --library from the last model run")
        args.library = max(candidates, key=lambda path: path.stat().st_mtime)
    args.library = args.library.resolve(strict=True)
    print(f"Library={args.library}; NPU={args.npu}; no model/profile; console output only", flush=True)
    print("Compare independent graph with compute -> prefetch -> compute fork/join; no model consumers.", flush=True)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    results = []
    for topology, polls in (
        ("independent", 100_000),
        ("fork_join", 0),
        ("fork_join", 100_000),
        ("fork_join", 1_000_000),
    ):
        command = [
            sys.executable,
            "-u",
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            "--npu",
            str(args.npu),
            "--library",
            str(args.library),
            "--polls",
            str(polls),
            "--topology",
            topology,
        ]
        print(f"CASE topology={topology} polls={polls}; timeout=90s", flush=True)
        try:
            completed = subprocess.run(command, env=env, timeout=90, check=False)
            results.append({"topology": topology, "polls": polls, "exit": completed.returncode})
        except subprocess.TimeoutExpired:
            results.append({"topology": topology, "polls": polls, "timeout": True})
        if results[-1].get("exit") != 0:
            break
    print("SUMMARY " + json.dumps(results), flush=True)
    print("state[1]=1: completion observed; 3: bounded wait expired. Success here does not prove model safety.")
    return 0 if len(results) == 4 and all(item.get("exit") == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
