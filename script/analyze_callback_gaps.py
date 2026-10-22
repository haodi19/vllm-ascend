"""从导出时间线分解相邻回调间隔；不把重叠传输当作因果归因。"""

import argparse
import collections
import json
from pathlib import Path

MAX_INTERVALS = 1000000


def events(path):
    """流式读取 traceEvents 数组，避免将数 GB 的时间线全部载入内存。"""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as source:
        buf = source.read(65536).lstrip()
        while not buf:
            chunk = source.read(65536)
            if not chunk:
                raise ValueError("时间线为空")
            buf = chunk.lstrip()
        # CANN exports may use a root array instead of a traceEvents object.
        if not buf.startswith("["):
            while '"traceEvents"' not in buf:
                chunk = source.read(65536)
                if not chunk:
                    raise ValueError("找不到 traceEvents 数组")
                buf += chunk
                if len(buf) > 1024 * 1024:
                    raise ValueError("traceEvents 不在文件头 1MB 内")
            buf = buf[buf.index('"traceEvents"') + len('"traceEvents"') :]
        while "[" not in buf:
            chunk = source.read(65536)
            if not chunk:
                raise ValueError("traceEvents 数组不完整")
            buf += chunk
        buf = buf[buf.index("[") + 1 :]
        while True:
            buf = buf.lstrip(" \n\r\t,")
            if buf.startswith("]"):
                return
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = source.read(65536)
                if not chunk:
                    raise ValueError("时间线未完整导出，无法完成分析") from None
                buf += chunk
                if len(buf) > 32 * 1024 * 1024:
                    raise ValueError("单个时间线事件超过 32MB")
                continue
            yield item
            buf = buf[end:]


def fields(event):
    return dict(part.split("=", 1) for part in event["name"].split("|")[1:] if "=" in part)


def analyze(path, output):
    updates, callbacks, copies = [], {}, []
    phases = collections.Counter()
    for event in events(path):
        name = str(event.get("name", ""))
        if name.startswith("moe_offload/"):
            phases[event.get("ph")] += 1
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        if event["dur"] < 0:
            continue
        if name.startswith("moe_offload/update_weights|"):
            updates.append(event)
        elif name.startswith("moe_offload/host_callback|"):
            callbacks[(event.get("pid"), fields(event).get("call"))] = event
        elif "MEMCPY_ASYNC" in name.upper():
            copies.append((event["ts"], event["ts"] + event["dur"]))
        if len(updates) + len(callbacks) + len(copies) > MAX_INTERVALS:
            raise ValueError("事件超过分析上限，请减少采集请求数")
    if not updates or not callbacks:
        raise ValueError(f"缺少 X 型 update_weights/host_callback 事件，MSTX phase 分布={dict(phases)}")
    # 合并传输区间，避免多流并行重复计时。仅表示时间覆盖，不表示因果。
    merged = []
    for start, end in sorted(copies):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    groups = collections.defaultdict(list)
    for event in updates:
        groups[event.get("pid")].append(event)
    rows, overlay, skipped = [], [], 0
    for pid, records in groups.items():
        records.sort(key=lambda event: event["ts"])
        for previous, current in zip(records, records[1:]):
            a, b = fields(previous), fields(current)
            if a.get("kind") == b.get("kind"):
                continue
            left = callbacks.get((pid, a.get("call")))
            right = callbacks.get((pid, b.get("call")))
            start, end = previous["ts"] + previous["dur"], current["ts"]
            if not left or not right:
                skipped += 1
                continue
            tail = left["ts"] + left["dur"]
            head = right["ts"]
            if not (left["ts"] <= previous["ts"] <= start <= tail <= head <= end <= right["ts"] + right["dur"]):
                skipped += 1
                continue
            coverage = sum(max(0, min(end, y) - max(start, x)) for x, y in merged if x < end and y > start)
            row = dict(
                direction=f"{a['kind']}->{b['kind']}",
                pid=pid,
                from_layer=a.get("layer"),
                to_layer=b.get("layer"),
                from_call=a.get("call"),
                to_call=b.get("call"),
                gap_us=end - start,
                callback_tail_us=tail - start,
                between_callbacks_us=head - tail,
                callback_head_us=end - head,
                memcpy_overlap_us=coverage,
            )
            rows.append(row)
            for label, begin, finish in (
                ("callback_tail", start, tail),
                ("between_callbacks_unattributed", tail, head),
                ("callback_head", head, end),
            ):
                overlay.append(
                    dict(
                        ph="X",
                        name=f"gap/{label}|{row['direction']}|calls={a['call']}->{b['call']}",
                        pid="offload_gap_analysis",
                        tid=str(pid),
                        ts=begin,
                        dur=finish - begin,
                        args=row,
                    )
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == path.resolve() or output.exists():
        raise ValueError("输出必须是尚不存在的新文件，不能覆盖原始时间线")
    with output.open("w", encoding="utf-8") as dest:
        dest.write('{"traceEvents":[')
        first = True
        for event in events(path):
            if not first:
                dest.write(",")
            json.dump(event, dest, ensure_ascii=False)
            first = False
        for event in overlay:
            if not first:
                dest.write(",")
            json.dump(event, dest, ensure_ascii=False)
            first = False
        dest.write("]}")
    print(f"GAP TRACE: {output}; pairs={len(rows)}; skipped={skipped}; MEMCPY intervals={len(copies)}")
    print("单位 us；按进程时间相邻配对，不保证属于同一层/请求；传输覆盖来自全设备轨道，不能相加归因。")
    for direction in sorted({row["direction"] for row in rows}):
        subset = [row for row in rows if row["direction"] == direction]
        print(f"GAP {direction}: count={len(subset)}")
        for key in ("gap_us", "callback_tail_us", "between_callbacks_us", "callback_head_us", "memcpy_overlap_us"):
            values = sorted(row[key] for row in subset)
            print(f"  {key}: mean={sum(values) / len(values):.3f}; max={values[-1]:.3f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.trace, args.output)
