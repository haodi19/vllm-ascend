"""从导出时间线分解相邻回调间隔；不把重叠传输当作因果归因。"""

import argparse
import collections
import json
import math
from pathlib import Path

MAX_INTERVALS = 1000000
READ_CHUNK_CHARS = 64 * 1024
MAX_VALUE_CHARS = 32 * 1024 * 1024


def _reject_constant(value):
    raise ValueError(f"非法 JSON 常量：{value}")


class _TraceReader:
    """Bound memory by one JSON value, rather than the complete event array."""

    def __init__(self, source):
        self.source = source
        self.buffer = ""
        self.offset = 0
        self.eof = False
        self.decoder = json.JSONDecoder(parse_constant=_reject_constant)

    def fill(self):
        self.buffer = self.buffer[self.offset :]
        self.offset = 0
        chunk = self.source.read(READ_CHUNK_CHARS)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self):
        while True:
            while self.offset < len(self.buffer) and self.buffer[self.offset] in " \n\r\t":
                self.offset += 1
            if self.offset < len(self.buffer):
                return self.buffer[self.offset]
            if self.eof:
                return ""
            self.fill()

    def take(self, expected):
        if self.peek() != expected:
            raise ValueError(f"时间线 JSON 不完整或格式错误：预期 {expected!r}")
        self.offset += 1

    def value(self):
        if not self.peek():
            raise ValueError("时间线未完整导出，缺少 JSON 值")
        while True:
            try:
                item, end = self.decoder.raw_decode(self.buffer, self.offset)
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("时间线未完整导出或 JSON 值格式错误") from None
            else:
                if end - self.offset > MAX_VALUE_CHARS:
                    raise ValueError("单个时间线 JSON 值超过大小上限")
                # A scalar number may span a chunk boundary (e.g. 1e+3).
                # Require a delimiter or EOF before accepting a decoded value.
                if end < len(self.buffer) and self.buffer[end] in " \n\r\t,:]}":
                    self.offset = end
                    return item
                if self.eof:
                    if end != len(self.buffer):
                        raise ValueError("时间线 JSON 值后存在非法字符")
                    self.offset = end
                    return item
            if len(self.buffer) - self.offset > MAX_VALUE_CHARS:
                raise ValueError("单个时间线 JSON 值超过大小上限")
            self.fill()

    def event_array(self):
        self.take("[")
        if self.peek() == "]":
            self.take("]")
            return
        while True:
            item = self.value()
            if not isinstance(item, dict):
                raise ValueError("时间线事件必须是 JSON 对象")
            yield item
            if self.peek() == "]":
                self.take("]")
                return
            self.take(",")
            if self.peek() == "]":
                raise ValueError("时间线数组不能包含尾随逗号")


def events(path):
    """流式读取根数组或 traceEvents 对象；遍历结束时验证整个 JSON 已闭合。"""
    with path.open(encoding="utf-8-sig") as source:
        reader = _TraceReader(source)
        first = reader.peek()
        if first == "[":
            yield from reader.event_array()
        elif first == "{":
            reader.take("{")
            found = False
            if reader.peek() != "}":
                while True:
                    key = reader.value()
                    if not isinstance(key, str):
                        raise ValueError("时间线对象的键必须为字符串")
                    reader.take(":")
                    if key == "traceEvents":
                        if found:
                            raise ValueError("存在重复 traceEvents，无法确定唯一时间线")
                        found = True
                        yield from reader.event_array()
                    else:
                        reader.value()
                    if reader.peek() == "}":
                        break
                    reader.take(",")
                    if reader.peek() == "}":
                        raise ValueError("时间线对象不能包含尾随逗号")
            reader.take("}")
            if not found:
                raise ValueError("找不到顶层 traceEvents 数组")
        else:
            raise ValueError("时间线为空或顶层不是 JSON 数组/对象")
        if reader.peek():
            raise ValueError("时间线 JSON 结束后仍有多余内容")


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
        relevant = name.startswith(("moe_offload/update_weights|", "moe_offload/host_callback|")) or (
            "MEMCPY_ASYNC" in name.upper()
        )
        if not relevant:
            continue
        # CANN may serialize timestamps as numbers or numeric strings.
        for key in ("ts", "dur"):
            value = event[key]
            try:
                if isinstance(value, bool):
                    raise ValueError("boolean time")
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid trace {key}: {value!r}") from None
            if not math.isfinite(number):
                raise ValueError(f"Invalid trace {key}: {value!r}")
            event[key] = number
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
