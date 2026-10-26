"""隔离检查常驻 AIV 等待是否妨碍其他流计算；不加载模型、不写文件。"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("callback", "publish_only", "mte")
HOLD_SECONDS = 0.1
MAX_POLLS = 1000000


def classify(result):
    if result["compute_done_ns"] < result["callback_exit_ns"]:
        return "OVERLAP_CONFIRMED"
    return "NO_OVERLAP_OBSERVED"


def child(args):
    # Imports stay isolated so the parent can bound stalled device operations.
    import torch
    import torch_npu
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    control = torch_npu.npu.SyncLaunchStream()
    compute = torch_npu.npu.SyncLaunchStream()
    pointer = int(control.npu_stream)
    x = torch.ones(8 * 1024 * 1024, dtype=torch.float16, device="npu")
    y = torch.empty_like(x)
    state = torch.zeros(4, dtype=torch.int64, device="npu")
    started = threading.Event()
    exited = threading.Event()
    stamps = {}

    def callback(*unused):
        stamps["callback_enter_ns"] = time.monotonic_ns()
        started.set()
        # Release GIL; do not submit any copy, NPU operation or logging here.
        time.sleep(HOLD_SECONDS)
        stamps["callback_exit_ns"] = time.monotonic_ns()
        exited.set()

    session = None
    if args.backend == "callback":
        torch_npu.npu._subscribe_report(control)
    else:
        lib = ctypes.CDLL(str(args.library))
        publish, wait = lib.host_sync_publish_launch, lib.host_sync_wait_launch
        publish.argtypes, publish.restype = [ctypes.c_void_p] * 3, None
        wait.argtypes, wait.restype = [ctypes.c_void_p] * 3 + [ctypes.c_uint64], None
        session = hs.Session(slots=1, backend="acl", device_id=0)
        session.register_task(0, callback)
        session.start()
        alias = session.begin_device_access()
        info = session.info()
    torch.npu.synchronize()
    # Warm up the exact independent computation, outside the held callback.
    with torch.npu.stream(compute):
        torch.add(x, 1, out=y)
    compute.synchronize()
    work = torch.npu.NPUGraph()
    handoff = torch.npu.NPUGraph()
    with torch.npu.graph(work, stream=compute):
        torch.add(x, 1, out=y)
    with torch.npu.graph(handoff, stream=control):
        if args.backend == "callback":
            torch_npu.npu._launch_host_func(control, callback, ())
        else:
            publish(pointer, alias + info["request_offset"], state.data_ptr())
            if args.backend == "mte":
                wait(pointer, alias + info["done_offset"], state.data_ptr(), MAX_POLLS)
    torch.npu.synchronize()
    if started.is_set():
        raise RuntimeError("Capture executed the callback unexpectedly")
    try:
        for generation in range(1, 4):
            started.clear()
            exited.clear()
            stamps.clear()
            with torch.npu.stream(control):
                handoff.replay()
            if not started.wait(5):
                raise RuntimeError("Callback did not start")
            # Allow the wait kernel to become resident after publish; the
            # control graph has no dependency on the independent work graph.
            time.sleep(0.005)
            stamps["submit_begin_ns"] = time.monotonic_ns()
            with torch.npu.stream(compute):
                work.replay()
            stamps["submit_return_ns"] = time.monotonic_ns()
            compute.synchronize()
            stamps["compute_done_ns"] = time.monotonic_ns()
            if not exited.wait(5):
                raise RuntimeError("Callback did not exit")
            control.synchronize()
            if session is not None:
                session.wait(0, generation, timeout_ms=5000)
            if args.backend == "mte":
                values = state.cpu().tolist()
                if values[:3] != [generation, 1, generation]:
                    raise RuntimeError(f"Wait did not finish successfully: {values}")
            if not torch.equal(y.cpu(), torch.full((x.numel(),), 2, dtype=torch.float16)):
                raise RuntimeError("Independent compute produced incorrect values")
            result = dict(
                backend=args.backend,
                round=generation,
                outcome=classify(stamps),
                callback_ms=(stamps["callback_exit_ns"] - stamps["callback_enter_ns"]) / 1e6,
                submit_ms=(stamps["submit_return_ns"] - stamps["submit_begin_ns"]) / 1e6,
                compute_wait_ms=(stamps["compute_done_ns"] - stamps["submit_return_ns"]) / 1e6,
                compute_done_before_callback_exit_ms=(stamps["callback_exit_ns"] - stamps["compute_done_ns"]) / 1e6,
            )
            print("RESULT " + json.dumps(result), flush=True)
    finally:
        torch.npu.synchronize()
        work.reset()
        handoff.reset()
        if session is not None:
            session.end_device_access(stream_synchronized=True)
            session.stop()
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux" or args.npu < 0:
        parser.error("在 Linux 容器指定物理 NPU 编号执行")
    if args.library is None:
        candidates = list((ROOT / "build/host_sync_device_stage3").glob("run_*/lib/libhost_sync_device.so"))
        if not candidates:
            parser.error("未找到已构建的 MTE 库，请用 --library 指定")
        args.library = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
    if args.backend:
        child(args)
        return 0
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    print(f"Library={args.library}; physical NPU={args.npu}; logical NPU=0", flush=True)
    print("独立图上的并行能力隔离测试，不是模型性能验收；CPU 回调故意等待 100ms。", flush=True)
    for backend in BACKENDS:
        print(f"CASE {backend}", flush=True)
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--npu",
            str(args.npu),
            "--library",
            str(args.library),
            "--backend",
            backend,
        ]
        try:
            result = subprocess.run(command, env=env, timeout=90, check=False)
        except subprocess.TimeoutExpired:
            print("TIMEOUT: 已终止子进程，停止后续测试。", flush=True)
            return 1
        if result.returncode:
            return result.returncode
    print("完成。OVERLAP_CONFIRMED 表示计算在 CPU 回调返回前完成；未观察到重叠仍需区分提交阻塞和设备等待。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
