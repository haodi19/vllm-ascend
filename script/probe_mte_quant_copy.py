"""隔离验证原始 Tensor.copy_ 是否与 AIV 等待冲突；不加载模型、不采集 profile。"""

import argparse
import ctypes
import json
import os
import runpy
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_POLLS = 1000000
CASES = (
    ("no_wait", "tensor", "different"),
    ("wait", "tensor", "same"),
    ("wait", "tensor", "different"),
    ("wait", "storage", "different"),
)


@dataclass
class CopyTask:
    source: object
    destination: object
    nbytes: int
    name: str = "probe_scale"


def check_state(state, generation):
    if state[:3] != [generation, 1, generation]:
        raise RuntimeError(f"AIV 未等到正常完成（不能把超时后 copy 成功算通过）: {state}")


def child(args):
    # NPU imports stay in isolated children; parent can enforce a hard timeout.
    import torch
    import torch_npu
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    adapter = runpy.run_path(str(ROOT / "vllm_ascend/expert_offload/host_sync.py"))["storage_quant_copy"]
    compute = torch.npu.Stream()
    transfer = torch_npu.npu.SyncLaunchStream()
    pointer = int(compute.npu_stream)
    # Same reshape/transpose structure as w4a8_mxfp4; 512 KiB, not actual model data.
    backing = torch.zeros((2, 512, 512, 2), dtype=torch.uint8, device="npu")
    destination = backing[1].transpose(0, 1)
    strides = destination.stride() if args.layout == "same" else (1024, 2, 1)
    source = torch.empty_strided(destination.shape, strides, dtype=torch.uint8, pin_memory=True)
    expected = (torch.arange(source.numel(), dtype=torch.int64) % 251).to(torch.uint8).reshape(source.shape)
    source.copy_(expected)
    state = torch.zeros(4, dtype=torch.int64, device="npu")
    torch.npu.synchronize()
    print(
        f"LAYOUT src={source.stride()} dst={destination.stride()} bytes={source.numel()} "
        f"format={torch_npu.get_npu_format(destination)}",
        flush=True,
    )
    library = ctypes.CDLL(str(args.library))
    publish, wait = library.host_sync_publish_launch, library.host_sync_wait_launch
    publish.argtypes, publish.restype = [ctypes.c_void_p] * 3, None
    wait.argtypes, wait.restype = [ctypes.c_void_p] * 3 + [ctypes.c_uint64], None
    finished = threading.Event()
    errors = []
    calls = []

    def callback():
        try:
            task = CopyTask(source, destination, source.numel())
            if args.copy == "storage":
                task = adapter(task, torch_npu.get_npu_format)
            print("WORKER copy_begin", flush=True)
            begin = time.perf_counter()
            with torch.npu.stream(transfer):
                task.destination.copy_(task.source, non_blocking=True)
            print(f"WORKER copy_return_ms={(time.perf_counter() - begin) * 1000:.3f}", flush=True)
            transfer.synchronize()
            print(f"WORKER sync_return_ms={(time.perf_counter() - begin) * 1000:.3f}", flush=True)
            calls.append(True)
        except Exception as exc:
            errors.append(repr(exc))
            raise
        finally:
            finished.set()

    session = hs.Session(slots=1, backend="acl", device_id=0)
    session.register_task(0, callback)
    session.start()
    alias = session.begin_device_access()
    info = session.info()

    def submit():
        publish(pointer, alias + info["request_offset"], state.data_ptr())
        if args.handoff == "wait":
            wait(pointer, alias + info["done_offset"], state.data_ptr(), MAX_POLLS)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=compute):
        submit()
    compute.synchronize()
    if calls:
        raise RuntimeError("Capture unexpectedly executed callback")
    for generation in range(1, 4):
        finished.clear()
        source.copy_((expected.to(torch.int16) + generation).remainder(251).to(torch.uint8))
        with torch.npu.stream(compute):
            graph.replay()
        compute.synchronize()
        if not finished.wait(15) or errors:
            raise RuntimeError(f"Worker failed or stalled: {errors}")
        if args.handoff == "wait":
            check_state(state.cpu().tolist(), generation)
        # Callback's event precedes native completion publication by a few instructions.
        deadline = time.monotonic() + 5
        while session.snapshot(0)["done_generation"] != generation:
            if time.monotonic() > deadline:
                raise RuntimeError("Native worker did not publish completion")
            time.sleep(0.001)
        if not torch.equal(destination.cpu(), source):
            raise RuntimeError("Copied values differ")
        if torch.count_nonzero(backing[0].cpu()).item():
            raise RuntimeError("Adjacent slot overwritten")
    graph.reset()
    session.end_device_access(stream_synchronized=True)
    session.stop()
    session.close()
    print("PROBE PASS: 3 graph replays, values and adjacent slot verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--handoff", choices=("wait", "no_wait"))
    parser.add_argument("--copy", choices=("tensor", "storage"))
    parser.add_argument("--layout", choices=("same", "different"))
    args = parser.parse_args()
    if sys.platform != "linux" or args.npu < 0:
        parser.error("在 Linux 容器指定物理 NPU 编号执行")
    if args.library is None:
        libraries = list((ROOT / "build/host_sync_device_stage3").glob("run_*/lib/libhost_sync_device.so"))
        if not libraries:
            parser.error("未找到已构建的 MTE 库，请用 --library 指定绝对路径")
        args.library = max(libraries, key=lambda path: path.stat().st_mtime).resolve()
    if args.child:
        child(args)
        return 0
    print(f"Library: {args.library}; NPU={args.npu}; no model/profile/files", flush=True)
    print("合成布局隔离测试；不是实际模型验收，也不是性能对比。", flush=True)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    results = []
    for handoff, copy, layout in CASES:
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            "--npu",
            str(args.npu),
            "--library",
            str(args.library),
            "--handoff",
            handoff,
            "--copy",
            copy,
            "--layout",
            layout,
        ]
        print(f"\nCASE {handoff}/{copy}/{layout}", flush=True)
        try:
            result = subprocess.run(command, env=env, timeout=90, check=False)
        except subprocess.TimeoutExpired:
            print("TIMEOUT: 子进程已终止；停止后续用例，请检查设备恢复状态。", flush=True)
            return 1
        results.append({"handoff": handoff, "copy": copy, "layout": layout, "pass": result.returncode == 0})
    print("PROBE SUMMARY " + json.dumps(results), flush=True)
    return int(any(not result["pass"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
