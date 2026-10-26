"""只读分析搬运准备与回调间隔；不启动模型、不写入或修改 profile。"""

import argparse
import collections
import math
import runpy
import statistics
from bisect import bisect_left, bisect_right
from pathlib import Path

MAX_EVENTS = 500000
MAX_FILES = 64


def number(event, name):
    value = event.get(name)
    if isinstance(value, bool):
        raise ValueError(f"非法时间字段 {name}: {value}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法时间字段 {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"非有限时间字段 {name}")
    return value


def load_trace(path):
    reader = runpy.run_path(str(Path(__file__).with_name("analyze_callback_gaps.py")))["events"]
    before = path.stat()
    processes = {}
    print("READ 1/2: 识别设备轨道", flush=True)
    for event in reader(path):
        if event.get("ph") == "M" and event.get("name") == "process_name":
            processes[str(event.get("pid"))] = str(event.get("args", {}).get("name", ""))
            if len(processes) > 256:
                raise ValueError("进程数量超过上限")
    hardware = {pid for pid, name in processes.items() if "ascend hardware" in name.lower()}
    callbacks, stages, device = [], [], []
    print(f"READ 2/2: Ascend Hardware={sorted(hardware)}", flush=True)
    for event in reader(path):
        if event.get("ph") != "X":
            continue
        name, pid = str(event.get("name", "")), str(event.get("pid"))
        is_callback = name.startswith("moe_offload/host_callback|")
        is_stage = name.startswith(("moe_offload/load_prepare|", "moe_offload/copy_submit|", "moe_offload/quant_"))
        if pid not in hardware and not is_callback and not is_stage:
            continue
        start, duration = number(event, "ts"), number(event, "dur")
        if duration < 0:
            raise ValueError("存在负时长，不能可靠分解")
        row = dict(name=name, pid=pid, tid=str(event.get("tid")), start=start, end=start + duration, dur=duration)
        if pid in hardware:
            device.append(row)
        elif is_callback or is_stage:
            row["fields"] = dict(part.split("=", 1) for part in name.split("|")[1:] if "=" in part)
            (callbacks if is_callback else stages).append(row)
        if len(callbacks) + len(stages) + len(device) > MAX_EVENTS:
            raise ValueError("保留事件超过上限，请选择较短采集")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("时间线仍在变化，请等导出完成后重试")
    return callbacks, stages, device, processes, hardware


def pair_callbacks(callbacks):
    """Match adjacent callbacks only when their layer relationship is valid."""
    groups = collections.defaultdict(list)
    for row in callbacks:
        groups[row["pid"]].append(row)
    pairs, skipped = [], 0
    for rows in groups.values():
        rows.sort(key=lambda row: row["start"])
        for left, right in zip(rows, rows[1:]):
            a, b = left["fields"], right["fields"]
            direction = f"{a.get('kind')}->{b.get('kind')}"
            try:
                x, y = int(a["layer"]), int(b["layer"])
            except (KeyError, ValueError):
                skipped += 1
                continue
            valid = (direction == "prefetch->reactive" and x == y) or (direction == "reactive->prefetch" and y == x + 1)
            if not valid or right["start"] < left["end"]:
                skipped += 1
                continue
            pairs.append(dict(left=left, right=right, direction=direction, gap_us=right["start"] - left["end"]))
    return pairs, skipped


def split_pairs(pairs, device):
    """Conservative temporal candidates, never guess when multiple tasks qualify."""
    waits = sorted((e for e in device if "host_sync_wait" in e["name"].lower()), key=lambda e: e["end"])
    publishes = sorted((e for e in device if "host_sync_publish" in e["name"].lower()), key=lambda e: e["start"])
    wait_ends, publish_starts = [e["end"] for e in waits], [e["start"] for e in publishes]
    for pair in pairs:
        begin, end = pair["left"]["end"], pair["right"]["start"]
        ws = waits[bisect_left(wait_ends, begin) : bisect_right(wait_ends, end)]
        ps = publishes[bisect_left(publish_starts, begin) : bisect_right(publish_starts, end)]
        pair["wait_candidates"], pair["publish_candidates"] = len(ws), len(ps)
        # A missing wait endpoint must not discard an independently observable
        # publish boundary. Keep the same conservative temporal qualification.
        pair["publish_resolved"] = len(ps) == 1 and ps[0]["end"] <= end
        if pair["publish_resolved"]:
            pair.update(
                before_publish_us=ps[0]["start"] - begin,
                publish_duration_us=ps[0]["dur"],
                after_publish_us=end - ps[0]["end"],
            )
        if len(ws) != 1 or len(ps) != 1 or ws[0]["end"] > ps[0]["start"] or ps[0]["end"] > end:
            pair["resolved"] = False
            continue
        wait, publish = ws[0], ps[0]
        pair.update(
            resolved=True,
            completion_to_wait_exit_us=wait["end"] - begin,
            after_wait_to_publish_us=publish["start"] - wait["end"],
            publish_us=publish["dur"],
            publish_exit_to_callback_us=end - publish["end"],
            wait_stream=wait["tid"],
            publish_stream=publish["tid"],
        )
    return pairs


def nearby_handoffs(pair, device):
    """Show neighbors without asserting that their task IDs match a callback."""
    begin, end = pair["left"]["end"], pair["right"]["start"]
    for token in ("host_sync_wait", "host_sync_publish"):
        candidates = [e for e in device if token in e["name"].lower()]
        nearest = sorted(candidates, key=lambda e: min(abs(e["end"] - begin), abs(e["start"] - end)))[:3]
        for event in sorted(nearest, key=lambda e: e["start"]):
            print(
                f"  NEIGHBOR stream={event['tid']} name={event['name']} "
                f"start_from_left_end={event['start'] - begin:.3f} "
                f"end_from_left_end={event['end'] - begin:.3f} dur={event['dur']:.3f} us"
            )


def preparation_details(callbacks, stages):
    for kind in ("prefetch", "reactive"):
        totals = collections.defaultdict(float)
        counts = collections.Counter()
        for event in stages:
            if event["fields"].get("kind") == kind:
                name = event["name"].split("|", 1)[0].split("/", 1)[1]
                totals[name] += event["dur"]
                counts[name] += 1
        callback_count = sum(e["fields"].get("kind") == kind for e in callbacks)
        print(f"PREPARATION {kind}: callbacks={callback_count}; 子项相互嵌套，不能相加")
        for name in sorted(totals):
            normalized = totals[name] / callback_count if callback_count else float("nan")
            print(f"  {name}: events={counts[name]} total_us={totals[name]:.3f} per_callback_us={normalized:.3f}")


def stats(label, values):
    if not values:
        print(f"{label}: 无样本")
        return
    values = sorted(values)
    p95 = values[math.ceil(0.95 * len(values)) - 1]
    print(
        f"{label}: count={len(values)} mean={statistics.mean(values):.3f} p50={statistics.median(values):.3f} "
        f"p95={p95:.3f} max={values[-1]:.3f} us"
    )


def interval_coverage(events, begin, end):
    intervals = sorted(
        (max(begin, e["start"]), min(end, e["end"])) for e in events if e["start"] < end and e["end"] > begin
    )
    total, cursor = 0.0, begin
    for start, stop in intervals:
        total += max(0, stop - max(cursor, start))
        cursor = max(cursor, stop)
    return total


def device_pipeline(device):
    """Device-clock-only intervals; no CPU timestamps or inferred clock shift."""
    publishes = sorted((e for e in device if "host_sync_publish" in e["name"].lower()), key=lambda e: e["start"])
    waits = collections.defaultdict(list)
    work_by_stream = collections.defaultdict(list)
    for event in sorted(device, key=lambda e: e["start"]):
        if "host_sync_wait" in event["name"].lower():
            waits[event["tid"]].append(event)
        if not any(token in event["name"].upper() for token in ("WAIT", "EVENT", "NOTIFY", "MEMCPY", "HOST_SYNC")):
            work_by_stream[event["tid"]].append(event)
    results = []
    for index, (left, right) in enumerate(zip(publishes, publishes[1:])):
        # Bound matching by the next publish on this same stream, rather than
        # by a CPU marker which may have a different clock calibration.
        following = next((e["start"] for e in publishes[index + 1 :] if e["tid"] == left["tid"]), float("inf"))
        candidates = [e for e in waits[left["tid"]] if left["start"] <= e["start"] < following]
        if len(candidates) != 1:
            continue
        wait = candidates[0]
        target_work = work_by_stream[right["tid"]]
        gap = right["start"] - wait["end"]
        work = interval_coverage(target_work, wait["end"], right["start"]) if gap >= 0 else 0
        results.append(
            dict(
                streams=f"{left['tid']}->{right['tid']}",
                wait_to_publish_us=gap,
                work_after_wait_us=work,
                other_after_wait_us=max(0, gap) - work,
                work_during_wait_us=interval_coverage(target_work, wait["start"], wait["end"]),
            )
        )
    return results


def report_device_pipeline(device):
    rows = device_pipeline(device)
    print("DEVICE-ONLY: 仅设备时钟；非等待/拷贝/事件任务视为计算候选，不代表完整因果归因")
    for streams in sorted({row["streams"] for row in rows}):
        selected = [row for row in rows if row["streams"] == streams]
        print(
            f"STREAM TRANSITION {streams}: count={len(selected)}; "
            f"next_publish_before_previous_wait_exit={sum(row['wait_to_publish_us'] < 0 for row in selected)}"
        )
        for key in ("wait_to_publish_us", "work_during_wait_us", "work_after_wait_us", "other_after_wait_us"):
            stats(key, [row[key] for row in selected])
    if not rows:
        print("没有可匹配的 MTE 设备发布/等待序列")


def report(path):
    print(f"\n=== {path} ===", flush=True)
    callbacks, stages, device, processes, hardware = load_trace(path)
    preparation_details(callbacks, stages)
    print("CPU 阶段计时含 profiler/MSTX 扰动；性能比较仍以 summary.txt 的无 profiler 窗口为准。")
    for kind in ("prefetch", "reactive"):
        for stage in ("load_prepare", "copy_submit"):
            stats(
                f"{kind}/{stage}",
                [
                    e["dur"]
                    for e in stages
                    if e["fields"].get("kind") == kind and e["name"].startswith(f"moe_offload/{stage}|")
                ],
            )
    pairs, skipped = pair_callbacks(callbacks)
    print(f"同层预取→按需、相邻层按需→预取：pairs={len(pairs)} skipped={skipped}")
    if not callbacks:
        raise ValueError("没有 host_callback 完整区间，无法关联")
    if len(hardware) != 1:
        print(f"DEVICE UNKNOWN: 需要唯一设备进程；metadata={processes}")
        return
    split_pairs(pairs, device)
    report_device_pipeline(device)
    print("以下是同一导出时间线中的唯一时间候选，不是关联 ID 证明的因果关系；候选不唯一就不分解。")
    print("completion_to_wait_exit 包含 CPU 发布完成、标记可见、设备轮询与退出；不是纯设备轮询时间。")
    print("after_wait_to_publish 包含映射写回、事件汇合、路由回传及调度；可能等待主流其他计算。")
    print("publish_exit_to_callback 包含发现请求、工作线程调度、GIL 与 Python 入口；不是纯 GIL 时间。")
    for direction in ("prefetch->reactive", "reactive->prefetch"):
        rows = [p for p in pairs if p["direction"] == direction]
        selected = [p for p in rows if p["resolved"]]
        print(f"\nDIRECTION {direction}: all={len(rows)} uniquely_matched={len(selected)}")
        stats("between_host_callbacks", [p["gap_us"] for p in rows])
        for key in ("before_publish_us", "publish_duration_us", "after_publish_us"):
            stats(key, [p[key] for p in rows if p["publish_resolved"]])
        for key in (
            "completion_to_wait_exit_us",
            "after_wait_to_publish_us",
            "publish_us",
            "publish_exit_to_callback_us",
        ):
            stats(key, [p[key] for p in selected])
        for pair in sorted(rows, key=lambda p: -p["gap_us"])[:3]:
            begin, end = pair["left"]["end"], pair["right"]["start"]
            print(
                f"PAIR layers={pair['left']['fields'].get('layer')}->{pair['right']['fields'].get('layer')} "
                f"calls={pair['left']['fields'].get('call')}->{pair['right']['fields'].get('call')} "
                f"ts={begin:.3f}..{end:.3f} gap={pair['gap_us']:.3f} "
                f"waits={pair['wait_candidates']} publishes={pair['publish_candidates']}"
            )
            overlap = collections.defaultdict(float)
            nearby_handoffs(pair, device)
            for event in device:
                amount = max(0, min(end, event["end"]) - max(begin, event["start"]))
                if amount:
                    overlap[(event["tid"], event["name"])] += amount
            for (stream, name), value in sorted(overlap.items(), key=lambda item: -item[1])[:6]:
                print(f"  device stream={stream} clipped_sum_us={value:.3f} name={name[:160]}")
            # Long unrelated idle-stream waits otherwise occupy every output
            # position, hiding the copies and kernels that advance this gap.
            active = [(key, value) for key, value in overlap.items() if "WAIT" not in key[1].upper()]
            for (stream, name), value in sorted(active, key=lambda item: -item[1])[:8]:
                print(f"  NON_WAIT stream={stream} clipped_sum_us={value:.3f} name={name[:160]}")
    inventory = collections.Counter(e["name"] for e in device if "host_sync" in e["name"].lower())
    print(f"HANDOFF KERNEL INVENTORY: {dict(inventory.most_common(12))}")
    if not inventory:
        names = collections.Counter(e["name"] for e in device)
        print(f"DEVICE NAME SAMPLES: {names.most_common(12)}")
    print("设备重叠时间不能相加归因；MEMCPY 空隙还包括回调内部的准备与提交，不能只看以上回调间隔。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="实验目录、profile 目录或原始 trace_view.json")
    args = parser.parse_args()
    paths = [args.profile] if args.profile.is_file() else []
    if not paths:
        for path in args.profile.rglob("trace_view.json"):
            if path.parent.name == "ASCEND_PROFILER_OUTPUT":
                paths.append(path)
                if len(paths) > MAX_FILES:
                    parser.error("文件超过上限，请指定单次实验目录")
    if not paths:
        parser.error("未找到原始 trace_view.json")
    failed = 0
    for path in sorted(paths):
        try:
            report(path)
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
