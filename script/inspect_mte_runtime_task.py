# SPDX-License-Identifier: Apache-2.0
"""只读提取失败进程的 Runtime 任务线索，不启动模型、不写文件。"""

import argparse
import os
import re
from pathlib import Path

TAIL_BYTES = 16 * 1024 * 1024
MAX_FILES = 40
MAX_LINES = 70


def failure_identity(text):
    details = [line for line in text.splitlines() if "[MOE-MTE ACL_DETAIL]" in line]
    if not details:
        raise ValueError("日志没有 MOE-MTE ACL_DETAIL；请指定本次实验目录")
    detail = details[-1]
    pid = re.search(r"\[PID:\s*(\d+)\]", detail)
    stream = re.search(r"stream_id\s*=\s*(\d+)", detail)
    task = re.search(r"task_id\s*=\s*(\d+)", detail)
    if not all((pid, stream, task)):
        raise ValueError("无法从 ACL_DETAIL 提取 PID、stream_id、task_id")
    return tuple(int(match.group(1)) for match in (pid, stream, task))


def select_lines(lines, stream, task, fallback=True):
    stream_pattern = re.compile(rf"stream[_ ]?id\s*[:=]\s*{stream}\b", re.I)
    task_pattern = re.compile(rf"task[_ ]?id\s*[:=]\s*{task}\b", re.I)
    selected = set()
    for index, line in enumerate(lines):
        if task_pattern.search(line) and stream_pattern.search(line):
            selected.update(range(max(0, index - 3), min(len(lines), index + 4)))
    if not selected and fallback:
        # Explicitly a fallback, not proof that a nearby kernel is the failed task.
        candidates = [
            index
            for index, line in enumerate(lines)
            if re.search(
                r"KERNEL_AIVEC|InplaceCopy|host_sync_wait|synchronize.*timeout|task.*(error|exception)", line, re.I
            )
        ]
        selected.update(candidates[-12:])
    return [(index + 1, lines[index][:1200]) for index in sorted(selected)[-MAX_LINES:]]


def inspect_run(run_dir, log_roots=()):
    service = Path(run_dir) / "mte/inference_python.log"
    if not service.is_file():
        raise ValueError(f"缺少服务日志：{service}")
    text = service.read_text(encoding="utf-8", errors="replace")
    pid, stream, task = failure_identity(text)
    print(f"READ ONLY: pid={pid}; stream_id={stream}; task_id={task}")
    print("stream/task 编号只在本次进程上下文中解释；相邻行不代表因果关系。")
    print("SERVICE HINTS:")
    for line in text.splitlines():
        if "[MOE-MTE ACL_DETAIL]" in line or "LAST_TASK" in line or "[MOE-MTE STREAM_MAP]" in line:
            print(line[:2400])
    roots = log_roots or [Path.home() / "ascend/log", Path("/var/log/npu/slog"), Path("/usr/slog")]
    files = []
    seen = set()
    pid_pattern = re.compile(rf"(?<!\d){pid}(?!\d)")
    for root in roots:
        print(f"SEARCH ROOT: {root}; exists={root.exists()}")
        if not root.is_dir():
            continue
        for directory, dirs, names in os.walk(root):
            if len(Path(directory).relative_to(root).parts) >= 5:
                dirs[:] = []
            for name in names:
                if not pid_pattern.search(name) or name.endswith((".gz", ".zip")):
                    continue
                path = Path(directory) / name
                try:
                    resolved = path.resolve()
                    if resolved not in seen and path.is_file():
                        seen.add(resolved)
                        files.append(path)
                except OSError:
                    continue
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"PID-named logs={len(files)}; reading at most {MAX_FILES}; last {TAIL_BYTES // (1024 * 1024)} MiB/file")
    records = []
    for path in files[:MAX_FILES]:
        try:
            with path.open("rb") as source:
                size = source.seek(0, 2)
                offset = max(0, size - TAIL_BYTES)
                source.seek(offset)
                lines = source.read().decode("utf-8", errors="replace").splitlines()
            exact = select_lines(lines, stream, task, fallback=False)
            selected = exact or select_lines(lines, stream, task)
            records.append((bool(exact), path, size, offset, selected))
        except OSError as exc:
            print(f"UNREADABLE {path}: {exc}")
    # Search every bounded file before printing: unrelated errors must not consume
    # the output budget before older DEBUG submission records are inspected.
    records.sort(key=lambda record: not record[0])
    budget = MAX_LINES
    for exact, path, size, offset, selected in records:
        print(f"LOG {path}; bytes={size}; tail_offset={offset}; exact_match={exact}; selected={len(selected)}")
        for number, line in selected[:budget]:
            print(f"tail_line={number}: {line}")
        budget -= min(len(selected), budget)
        if budget == 0:
            print("达到输出上限；未打印的日志不等于没有相关任务。")
            break
    print("仅搜索上述数量与尾部范围；未命中不能排除记录位于更早或已轮转的日志。")
    if not files:
        print("未发现该 PID 命名的日志；可能目录不同、日志轮转或日志未开启，不能据此判断任务不存在。")
    print("完成；无模型启动、无文件写入。请返回以上控制台输出。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    inspect_run(args.run_dir, args.log_root)


if __name__ == "__main__":
    main()
