# SPDX-License-Identifier: Apache-2.0
"""只读捕获图：关联本轮超时任务、相邻节点与 Event 名称候选，不使用伪时间作性能分析。"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from inspect_mte_outstanding import snapshot_targets
from inspect_mte_runtime_task import failure_identity


def identity(event):
    args = event.get("args", {})
    try:
        return int(args["Model Id"]), int(args["Stream Id"]), int(args["Task Id"])
    except (KeyError, ValueError, TypeError):
        return None


def describe(event):
    args = event.get("args", {})
    result = {"id": identity(event), "name": event.get("name"), "type": args.get("Task Type")}
    for key in ("Numblocks", "Schem Mode", "Kernel Args", "Kernel Args Size"):
        if key in args:
            if key == "Kernel Args" and not event.get("name", "").startswith("host_sync_"):
                continue
            result[key] = args[key]
    return json.dumps(result, ensure_ascii=False)


def event_candidates(nodes, event):
    match = re.fullmatch(r"EVENT_(WAIT|RECORD)_(\d+)", event.get("name", ""))
    key = identity(event)
    if not match or key is None:
        return []
    other = "RECORD" if match.group(1) == "WAIT" else "WAIT"
    name = f"EVENT_{other}_{match.group(2)}"
    return [
        node
        for node in nodes
        if identity(node) is not None and identity(node)[0] == key[0] and node.get("name") == name
    ]


def report_graph(events, targets, compact=False):
    nodes = [event for event in events if isinstance(event, dict) and identity(event) is not None]
    print(f"nodes={len(nodes)}; models={sorted({identity(node)[0] for node in nodes})}")
    print("Task Type counts=" + json.dumps(Counter(node.get("args", {}).get("Task Type") for node in nodes)))
    for stream, task in sorted(targets):
        matches = [node for node in nodes if identity(node)[1:] == (stream, task)]
        print(f"TARGET stream={stream} task={task} matches={len(matches)}")
        if not matches:
            print("NOT_IN_THIS_GRAPH：可能属于其他图或单算子流，不自行补全身份。")
        for event in matches[:4]:
            model = identity(event)[0]
            track = sorted(
                [node for node in nodes if identity(node)[:2] == (model, stream)], key=lambda node: identity(node)[2]
            )
            index = next(i for i, node in enumerate(track) if node is event)
            print("MATCH " + describe(event))
            radius = 1 if compact else 5
            window = track[max(0, index - radius) : index + (2 if compact else 7)]
            for node in window:
                print("  NEIGHBOR " + describe(node))
                candidates = event_candidates(nodes, node)
                if candidates:
                    print(f"  EVENT_NAME_CANDIDATES={len(candidates)}（同图同编号名称；复用时不能认定唯一依赖）")
                    for candidate in candidates[:4]:
                        print("    " + describe(candidate))
            # Include nearest explicit Event wait before target and record after target,
            # even when model computation makes them fall outside the small window.
            for direction, sequence in (
                ("PREVIOUS_EVENT", reversed(track[:index])),
                ("NEXT_EVENT", iter(track[index + 1 :])),
            ):
                edge = next((node for node in sequence if node.get("name", "").startswith("EVENT_")), None)
                if edge:
                    print(direction + " " + describe(edge))
                    for candidate in event_candidates(nodes, edge)[:4]:
                        print("  NAME_CANDIDATE " + describe(candidate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--compact", action="store_true", help="只输出目标及紧邻节点，保留 Event 依赖候选")
    parser.add_argument("--log-root", type=Path, default=Path.home() / "ascend/log/debug/plog")
    args = parser.parse_args()
    service = (args.run_dir / "mte/inference_python.log").read_text(encoding="utf-8", errors="replace")
    pid, stream, task = failure_identity(service)
    targets = {(stream, task)}
    pid_pattern = re.compile(rf"(?<!\d){pid}(?!\d)")
    snapshot_lines = set()
    for path in sorted(args.log_root.rglob("*")):
        if path.is_file() and pid_pattern.search(path.name) and path.suffix not in (".gz", ".zip"):
            with path.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    if "PrintStreamTimeoutSnapshotInfo:" in line:
                        snapshot_lines.add(line.strip())
                        targets.update(snapshot_targets([line]))
    print(f"本轮 PID={pid}; targets={sorted(targets)}；仅使用本轮快照，不套用旧进程编号。")
    print(f"SNAPSHOT_LINES={len(snapshot_lines)}；若有多个超时时刻，以下目标为并集，不能视作同一瞬间。")
    for line in sorted(snapshot_lines)[:30]:
        print("SNAPSHOT " + line[:1600])
    print("图 dump 表达捕获结构；ts/dur 不作为实测时延、开始执行或完成证据。")
    paths = sorted((args.run_dir / "mte/graph_debug").rglob("graph_dump*"))
    for path in paths:
        if not path.is_file():
            continue
        print(f"GRAPH {path}; bytes={path.stat().st_size}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            events = data.get("traceEvents", []) if isinstance(data, dict) else data
            if not isinstance(events, list):
                raise ValueError("不是事件列表")
            report_graph(events, targets, compact=args.compact)
        except (ValueError, UnicodeError) as exc:
            print(f"UNREADABLE_GRAPH {exc}")
    if not paths:
        print("没有找到图文件。")
    print("只读结束；不加载模型、不写文件。请返回控制台内容。")


if __name__ == "__main__":
    main()
