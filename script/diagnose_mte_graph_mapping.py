# SPDX-License-Identifier: Apache-2.0
"""纯 MTE 复现并打印捕获图任务信息，生成精简定位文本。"""

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from uuid import uuid4

from run_moe_host_sync import failure_excerpt


def graph_record(line):
    return bool(re.search(r"DebugPrint|DebugDump|PrintModel|ModelPrint|PrintTaskInfo|PrintStreamInfo", line, re.I))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--routing-cores", type=int, default=0, choices=range(57))
    parser.add_argument("--compute-cube-cores", type=int, default=0, choices=range(29))
    parser.add_argument("--dsa-cube-cores", type=int, default=0, choices=range(29))
    parser.add_argument("--native-stream-limits", action="store_true")
    parser.add_argument("--prefetch-wait-at-join", action="store_true")
    parser.add_argument("--trace-startup", action="store_true")
    parser.add_argument("--no-load-diagnostics", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.trace_startup and not args.prefetch_wait_at_join:
        parser.error("--trace-startup requires --prefetch-wait-at-join")
    if args.no_load_diagnostics and not args.prefetch_wait_at_join:
        parser.error("--no-load-diagnostics requires --prefetch-wait-at-join")
    if args.dsa_cube_cores and not args.compute_cube_cores:
        parser.error("--dsa-cube-cores requires --compute-cube-cores")
    if args.native_stream_limits and not args.compute_cube_cores:
        parser.error("--native-stream-limits requires --compute-cube-cores")
    if args.prefetch_wait_at_join and any(
        (args.routing_cores, args.compute_cube_cores, args.dsa_cube_cores, args.native_stream_limits)
    ):
        parser.error("wait placement diagnosis must not change core limits")
    # Fail before loading the model if this build has no Python entry point.
    import torch_npu

    graph = torch_npu.npu.NPUGraph
    if args.native_stream_limits or args.dsa_cube_cores:
        if not all(callable(getattr(torch_npu.npu, name, None)) for name in ("get_stream_limit", "set_stream_limit")):
            raise RuntimeError("缺少 Torch get/set_stream_limit；未启动模型")
        inspect.signature(torch_npu.npu.set_stream_limit).bind(None, cube_num=14, vector_num=28)
        inspect.signature(torch_npu.npu.get_stream_limit).bind(None)
    if not all(callable(getattr(graph, name, None)) for name in ("enable_debug_mode", "debug_dump")):
        raise RuntimeError("当前 Torch NPU 没有图调试入口；未启动模型")
    signature = inspect.signature(graph.debug_dump)
    signature.bind(None, "diagnostic_graph_path")
    print(f"NPUGraph.debug_dump signature={signature}; 已检查路径参数，尚未调用设备接口", flush=True)
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir or root / "bench_results" / f"mte_graph_mapping_{uuid4().hex[:8]}").resolve()
    if output.exists():
        raise ValueError(f"诊断目录已存在，拒绝覆盖：{output}")
    env = os.environ.copy()
    env["ASCEND_MODULE_LOG_LEVEL"] = "RUNTIME=0"
    command = [
        sys.executable,
        "-B",
        "script/run_moe_host_sync.py",
        "--npu",
        str(args.npu),
        "--backend",
        "mte",
        "--diagnose-native-wait",
        "--diagnose-graph-dump",
        "--requests",
        "1",
        "--warmup-requests",
        "1",
        "--profile-timeout",
        "30",
        "--runtime-source",
        "/home/l00992607/vllm-ascend",
        "--test-script",
        "/home/l00992607/vllm-ascend/script/test_graph2.sh",
        "--output-dir",
        str(output),
    ]
    if args.routing_cores:
        command += ["--diagnose-routing-vector-cores", str(args.routing_cores)]
    if args.compute_cube_cores:
        command += ["--diagnose-compute-cube-cores", str(args.compute_cube_cores)]
    if args.dsa_cube_cores:
        command += ["--diagnose-dsa-cube-cores", str(args.dsa_cube_cores)]
    if args.native_stream_limits:
        command += ["--diagnose-native-stream-limits"]
    if args.prefetch_wait_at_join:
        command += ["--diagnose-prefetch-wait-at-join", "--skip-profile"]
        command[command.index("--requests") + 1] = "3"
    if args.trace_startup:
        command += ["--trace-startup"]
    if args.no_load_diagnostics:
        command.remove("--diagnose-native-wait")
        command += ["--no-load-diagnostics"]
    print(f"纯 MTE 图诊断；routing_cores={args.routing_cores}（0不改）；原始拷贝、严格预取等待。", flush=True)
    print(f"完整构建/服务日志保留在 {output}", flush=True)
    with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) as proc:
        elapsed = 0
        while True:
            try:
                code = proc.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                elapsed += 15
                print(f"WAIT {elapsed}s；运行器自身的启动/请求超时继续生效", flush=True)
    service = output / "mte/inference_python.log"
    text = service.read_text(encoding="utf-8", errors="replace") if service.is_file() else ""
    report = output / "graph_diagnosis.txt"
    if not output.is_dir():
        raise RuntimeError(f"运行器退出 {code}，没有生成实验目录")
    with report.open("w", encoding="utf-8") as dest:
        dest.write(f"RESOURCE_API={'torch' if args.native_stream_limits or args.dsa_cube_cores else 'raw_acl'}\n")
        dest.write(f"DSA_CUBE_REQUESTED={args.dsa_cube_cores}; Vector=2*Cube；0表示不改\n")
        dest.write(f"COMPUTE_CUBE_REQUESTED={args.compute_cube_cores}; Vector=2*Cube；0表示不改\n")
        dest.write(f"ROUTING_CORES_REQUESTED={args.routing_cores}; 以 GRAPH_KERNEL 实际值判断是否生效\n")
        dest.write(f"RUNNER_EXIT={code}\n" + failure_excerpt(text) + "\n")
        dest.write(f"PREFETCH_WAIT_AT_JOIN={args.prefetch_wait_at_join}\n")
        for line in text.splitlines():
            if "[MOE-MTE COMPUTE_LIMIT]" in line:
                dest.write(line + "\n")
        summary = output / "summary.txt"
        if summary.is_file():
            for line in summary.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MOE_RESULT ") or line.startswith("FAILED:"):
                    dest.write(line + "\n")
        starts = sum("[MOE-MTE GRAPH_DUMP_BEGIN]" in line for line in text.splitlines())
        ends = sum("[MOE-MTE GRAPH_DUMP_END]" in line for line in text.splitlines())
        dest.write(f"GRAPH_DUMP begin={starts} end={ends}; 接口返回不保证所需映射已包含\n")
        graph_root = output / "mte/graph_debug"
        for path in sorted(graph_root.rglob("*")):
            if path.is_file():
                dest.write(f"GRAPH_FILE {path.relative_to(output)} bytes={path.stat().st_size}\n")
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    events = data.get("traceEvents", []) if isinstance(data, dict) else data
                    signatures = set()
                    for event in events:
                        name = event.get("name", "")
                        if any(marker in name for marker in ("MoeInitRouting", "host_sync_wait", "HcPre", "MatMulV3")):
                            info = event.get("args", {})
                            signatures.add((info.get("Stream Id"), name, info.get("Numblocks"), info.get("Schem Mode")))
                    for signature in sorted(signatures, key=str):
                        dest.write("GRAPH_KERNEL " + json.dumps(signature) + "\n")
                except (ValueError, UnicodeError, TypeError, AttributeError) as exc:
                    dest.write(f"GRAPH_SCHEMA_UNAVAILABLE {exc}\n")
                # Text preview only; binary artifacts stay in the green environment.
                with path.open("rb") as source:
                    head = source.read(4096)
                if b"\0" not in head:
                    dest.write(head.decode("utf-8", errors="replace")[:2000] + "\n")
        for line in text.splitlines():
            if graph_record(line):
                dest.write(line[:1800] + "\n")
        pids = set(re.findall(r"EngineCore pid=(\d+)", text))
        records = deque(maxlen=70)
        count = 0
        log_root = Path.home() / "ascend/log/debug/plog"
        for path in sorted(log_root.glob("*")):
            if not path.is_file() or not any(re.search(rf"(?<!\d){pid}(?!\d)", path.name) for pid in pids):
                continue
            with path.open(encoding="utf-8", errors="replace") as source:
                for number, line in enumerate(source, 1):
                    if graph_record(line):
                        count += 1
                        records.append(f"{path.name}:{number}: {line.rstrip()[:1800]}")
        dest.write(f"RUNTIME graph debug candidates={count}; last_at_most=70\n")
        for line in records:
            dest.write(line + "\n")
        if not count:
            dest.write("未命中图打印记录；需区分日志位置/格式或接口输出，不解释为图没有任务。\n")
        if "[MOE-MTE ACL_DETAIL]" in text:
            dest.write("\n=== 本轮超时快照与图内任务匹配 ===\n")
            dest.flush()
            try:
                mapped = subprocess.run(
                    [sys.executable, "-B", str(root / "script/inspect_mte_graph_dump.py"), str(output), "--compact"],
                    cwd=root,
                    stdout=dest,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=False,
                )
                dest.write(f"SNAPSHOT_MAPPING_EXIT={mapped.returncode}\n")
            except subprocess.TimeoutExpired:
                dest.write("SNAPSHOT_MAPPING_TIMEOUT；保留原始日志，不解释为没有任务映射。\n")
    print(f"仅返回此文件：{report}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
