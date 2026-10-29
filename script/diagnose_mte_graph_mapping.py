# SPDX-License-Identifier: Apache-2.0
"""纯 MTE 复现并打印捕获图任务信息，生成精简定位文本。"""

import argparse
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
    args = parser.parse_args()
    # Fail before loading the model if this build has no Python entry point.
    import torch_npu

    graph = torch_npu.npu.NPUGraph
    if not all(callable(getattr(graph, name, None)) for name in ("enable_debug_mode", "debug_dump")):
        raise RuntimeError("当前 Torch NPU 没有图调试入口；未启动模型")
    root = Path(__file__).resolve().parents[1]
    output = root / "bench_results" / f"mte_graph_mapping_{uuid4().hex[:8]}"
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
    print("启动纯 MTE 图映射诊断；不分核、不更换拷贝、不替换预取等待。", flush=True)
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
        dest.write(f"RUNNER_EXIT={code}\n" + failure_excerpt(text) + "\n")
        starts = sum("[MOE-MTE GRAPH_DUMP_BEGIN]" in line for line in text.splitlines())
        ends = sum("[MOE-MTE GRAPH_DUMP_END]" in line for line in text.splitlines())
        dest.write(f"GRAPH_DUMP begin={starts} end={ends}; 接口返回不保证所需映射已包含\n")
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
    print(f"仅返回此文件：{report}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
