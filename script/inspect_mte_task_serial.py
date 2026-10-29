# SPDX-License-Identifier: Apache-2.0
"""只读按提交序号提取 Runtime 下发上下文，避免循环 task_id 混淆。"""

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

from inspect_mte_runtime_task import failure_identity

CONTEXT_LINES = 55
OUTPUT_LINES = 100
ADDRESS_LINES = 30


def last_submission(text):
    lines = [line for line in text.splitlines() if " LAST_TASK " in line]
    if not lines:
        raise ValueError("没有 LAST_TASK 记录")
    label, raw = lines[-1].split(" LAST_TASK ", 1)
    record = json.loads(raw)
    if record.get("code") != 0 or record.get("task_id") in (None, 4294967295):
        raise ValueError("最后一次任务查询无有效编号")
    return label, record


def serial_context(path, serial):
    pattern = re.compile(rf"task_sn\s*[:=]\s*{serial}\b", re.I)
    before = deque(maxlen=CONTEXT_LINES)
    selected = {}
    remaining = 0
    with path.open(encoding="utf-8", errors="replace") as source:
        for number, line in enumerate(source, 1):
            if pattern.search(line):
                selected.update(before)
                remaining = CONTEXT_LINES
            if remaining:
                selected[number] = line.rstrip()
                remaining -= 1
            before.append((number, line.rstrip()))
    return sorted(selected.items())


def kernel_signature(line, stream):
    if "StreamLaunchKernelV2:" not in line or not re.search(rf"stream_id={stream}\b", line):
        return None
    name = re.search(r"kernel_name=([^, ]+)", line)
    cores = re.search(r"coreDim=(\d+)", line)
    return (name.group(1), int(cores.group(1))) if name and cores else None


def summarize_kernels(text, root):
    maps = [line for line in text.splitlines() if "[MOE-MTE STREAM_MAP]" in line]
    if not maps:
        raise ValueError("服务未到达原始拷贝诊断，缺少 STREAM_MAP")
    pid = int(re.search(r"EngineCore pid=(\d+)", maps[-1]).group(1))
    stream = json.loads(maps[-1].split("[MOE-MTE STREAM_MAP]", 1)[1])["load"]["stream_id"]
    counts = Counter()
    pid_pattern = re.compile(rf"(?<!\d){pid}(?!\d)")
    for path in root.rglob("*"):
        if not path.is_file() or not pid_pattern.search(path.name) or path.suffix in (".gz", ".zip"):
            continue
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                signature = kernel_signature(line, stream)
                if signature:
                    counts[signature] += 1
    print(f"RUNTIME LOAD KERNELS pid={pid} stream={stream}; 下发记录数量，非执行完成次数")
    for (name, cores), count in counts.most_common(12):
        print(f"kernel={name} coreDim={cores} records={count}")
    if not counts:
        print("未找到下发记录，不能声称核数限制已生效。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path.home() / "ascend/log")
    parser.add_argument("--kernel-summary", action="store_true")
    args = parser.parse_args()
    text = (args.run_dir / "mte/inference_python.log").read_text(encoding="utf-8", errors="replace")
    if args.kernel_summary:
        summarize_kernels(text, args.log_root)
        return
    pid, stream, task = failure_identity(text)
    label, record = last_submission(text)
    serial = record["task_id"]
    print(f"只读：PID={pid}; timeout stream={stream} task={task}; submission serial candidate={serial}")
    print(label)
    print("按 task_sn 核对映射；task_id 可循环复用。下发成功不等于 kernel 已开始执行。")
    pid_pattern = re.compile(rf"(?<!\d){pid}(?!\d)")
    files = sorted(
        p
        for p in args.log_root.rglob("*")
        if p.is_file() and pid_pattern.search(p.name) and p.suffix not in (".gz", ".zip")
    )
    addresses = set()
    count = 0
    hits = 0
    for path in files:
        selected = serial_context(path, serial)
        if not selected:
            continue
        hits += 1
        print(f"CONTEXT {path}")
        for number, line in selected:
            if "GetFunctionDevAddr" in line:
                addresses.update(re.findall(r"(?:baseAddr is|func1 address is)\s+(0x[0-9a-fA-F]+)", line))
            # Repeated queue polling obscures API/argument/kernel submission records.
            if re.search(r"GetTaskInfo:|JudgeIsExecuted:|GetFinishedTaskIdBySqHead:", line):
                continue
            if count < OUTPUT_LINES:
                print(f"{number}: {line[:1600]}")
                count += 1
    print(f"matched_files={hits}; printed_context_lines={count}; addresses={sorted(addresses)}")
    if not hits:
        print("未找到序号映射：可能日志已轮转或不在指定目录；不要据此推断任务不存在。")
    printed = 0
    if addresses:
        for path in files:
            with path.open(encoding="utf-8", errors="replace") as source:
                for number, line in enumerate(source, 1):
                    if any(address.lower() in line.lower() for address in addresses) and re.search(
                        r"register|kernel.*name|function.*name|binary", line, re.I
                    ):
                        if printed < ADDRESS_LINES:
                            print(f"ADDRESS {path.name}:{number}: {line.rstrip()[:1600]}")
                            printed += 1
    print(f"address_registration_lines={printed}; 输出有上限，未命中不代表没有注册记录。")
    print("完成：无模型启动、无文件写入。上下文仅帮助关联，不自动证明设备资源竞争。")


if __name__ == "__main__":
    main()
