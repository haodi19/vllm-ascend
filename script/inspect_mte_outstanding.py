# SPDX-License-Identifier: Apache-2.0
"""只读提取超时快照内其他 AIVEC 任务的下发线索，不再启动模型。"""

import argparse
import re
from collections import deque
from pathlib import Path

from inspect_mte_runtime_task import failure_identity

CONTEXT = 35


def task_pair(line):
    stream = re.search(r"stream[_ ]?id\s*[:=]\s*(\d+)", line, re.I)
    task = re.search(r"task[_ ]?id\s*[:=]\s*(\d+)", line, re.I)
    return (int(stream.group(1)), int(task.group(1))) if stream and task else None


def snapshot_targets(lines):
    return {
        task_pair(line)
        for line in lines
        if "PrintStreamTimeoutSnapshotInfo:" in line and "KERNEL_AIVEC" in line and task_pair(line)
    }


def submission_context(files, targets):
    # Keep the most recent *submission* occurrence, never overwrite it with
    # timeout/GetTaskInfo lookups. A circular task position alone is not unique.
    before = deque(maxlen=CONTEXT)
    results = {}
    pending = {}
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as source:
            for number, raw in enumerate(source, 1):
                line = raw.rstrip()
                entry = (path.name, number, line)
                for key in list(pending):
                    results[key].append(entry)
                    pending[key] -= 1
                    if not pending[key]:
                        del pending[key]
                key = task_pair(line)
                if key in targets and re.search(r"AllocTaskInfo:|StarsAddTaskToStream:|DavidSendTask:", line):
                    results[key] = [*before, entry]
                    pending[key] = 8
                before.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path.home() / "ascend/log/debug/plog")
    args = parser.parse_args()
    service = (args.run_dir / "mte/inference_python.log").read_text(encoding="utf-8", errors="replace")
    pid, load_stream, load_task = failure_identity(service)
    pid_pattern = re.compile(rf"(?<!\d){pid}(?!\d)")
    files = sorted(
        p
        for p in args.log_root.rglob("*")
        if p.is_file() and pid_pattern.search(p.name) and p.suffix not in (".gz", ".zip")
    )
    targets = set()
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as source:
            targets.update(snapshot_targets(source))
    print(f"PID={pid}; timeout_load=({load_stream},{load_task}); AIVEC snapshot targets={sorted(targets)}")
    print("以下是各循环 task 位置最近一次下发上下文候选；相邻线程日志不自动构成任务映射。")
    results = submission_context(files, targets)
    for key in sorted(targets):
        print(f"TASK stream={key[0]} task={key[1]}")
        entries = results.get(key, [])
        if not entries:
            print("NO_SUBMISSION: 未找到下发映射，可能是图内部任务或日志已轮转；不能猜测 kernel 身份。")
        for path, number, line in entries:
            if re.search(
                r"StreamLaunchKernelV2:|LaunchKernelByArgsWithType:|ConstructCommonAicAivSqePart:|"
                r"ConstructAivSqePart:|AllocTaskInfo:|StarsAddTaskToStream:|DavidSendTask:|"
                r"kernel_name=|host_sync_wait|host_sync_publish",
                line,
            ):
                print(f"{path}:{number}: {line[:1500]}")
    # Graph internals may be remapped rather than submitted on their capture IDs.
    # Print a small set of explicit mapping lines, not arbitrary neighboring waits.
    mappings = deque(maxlen=15)
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as source:
            for number, line in enumerate(source, 1):
                if re.search(r"host_sync_wait|host_sync_publish", line) and re.search(
                    r"kernel_name|register|function", line, re.I
                ):
                    mappings.append(f"{path.name}:{number}: {line.rstrip()[:1500]}")
    print("HANDSHAKE KERNEL RECORDS（独立线索，不能直接对应上述任务）")
    for line in mappings:
        print(line)
    print("只读结束；不写文件、不占用 NPU。未找到映射时，本轮不能据此判断是哪一个图内算子。")


if __name__ == "__main__":
    main()
