# SPDX-License-Identifier: Apache-2.0
"""离线核查加载流长空泡中的短回调，并按实际加载准备次数拆分Host间隔。"""

import argparse
import bisect
import collections
import json
import math
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from inspect_moe_profile import _interval, _task_kind, copy_gap_boundaries
from moe_outlier_stats import outlier_summary
from offload_bubble_stats import pair_direction
from verify_memcpy_gaps import CASES, read_offload_header, select_host_callbacks, select_track

STAGES = frozenset(("update_weights", "load_submit", "synchronize", "copy_submit", "load_prepare", "scale_refresh"))
SINGLE_STAGES = ("update_weights", "load_submit", "synchronize", "copy_submit")
MIN_GAP_US = 10.0
LONG_GAP_US = 5000.0
SHORT_LOAD_US = 10.0
TARGET_GAP_US = 2500.0
MAX_EXAMPLES = 2
MAX_EXAMPLE_CALLBACKS = 8
MAX_TRACES = 64


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class Coverage:
    """Union coverage on the exported time axis, never a sum across devices."""

    def __init__(self, intervals):
        self.intervals = union_intervals(intervals)
        self.starts = [start for start, _ in self.intervals]
        self.ends = [end for _, end in self.intervals]
        self.prefix = [0.0]
        for start, end in self.intervals:
            self.prefix.append(self.prefix[-1] + end - start)

    def before(self, time):
        index = bisect.bisect_right(self.ends, time)
        value = self.prefix[index]
        if index < len(self.starts):
            value += max(0.0, time - self.starts[index])
        return value

    def overlap(self, start, end):
        return max(0.0, self.before(end) - self.before(start))


def read_details(path, key, callbacks):
    """Second full JSON pass: retain stages, target-stream work and other-stream compute."""
    host_pid = callbacks[0]["pid"]
    window = min(cb["start"] for cb in callbacks), max(cb["end"] for cb in callbacks)
    stages, tasks, compute = collections.defaultdict(lambda: collections.defaultdict(list)), [], []
    retained = len(callbacks)
    for number, event in enumerate(events(path), 1):
        if event.get("ph") != "X":
            continue
        pid, tid = str(event.get("pid")), str(event.get("tid"))
        name = str(event.get("name", ""))
        stage = name.split("|", 1)[0].removeprefix("moe_offload/")
        if pid == host_pid and name.startswith("moe_offload/") and stage in STAGES:
            meta = fields(event)
            call = meta.get("call", "")
            if not call.isascii() or not call.isdecimal():
                raise ValueError("加载分项缺少有效call ID，不能猜测与回调的对应")
            start, end = _interval(event)
            stages[int(call)][stage].append(dict(start=start, end=end, tid=tid, meta=meta))
            retained += 1
        elif pid == key[0]:
            start, end = _interval(event)
            if (pid, tid) == key:
                tasks.append(dict(start=start, end=end, name=name, event_index=number))
                retained += 1
            elif end > window[0] and start < window[1] and _task_kind(name) == "compute_candidate":
                compute.append(dict(start=start, end=end, tid=tid, name=name))
                retained += 1
        if retained > MAX_INTERVALS:
            raise ValueError("诊断保留的区间超过上限；请按单个采集窗口处理")
    return stages, tasks, compute


def annotate_callbacks(callbacks, stages):
    annotated, seen, visits = [], set(), collections.Counter()
    for cb in sorted(callbacks, key=lambda x: (x["start"], x["end"], x["event_index"])):
        call = cb["call"]
        if call is None or call in seen or cb["layer"] is None or cb["kind"] == "unknown":
            raise ValueError("回调身份无效或重复，不能按call/层配对")
        if annotated and cb["start"] < annotated[-1]["end"]:
            raise ValueError("Host回调重叠，不能将它们解释成串行空载链")
        seen.add(call)
        parts = stages.get(call, {})
        issues = [f"{name}_count={len(parts.get(name, []))}" for name in SINGLE_STAGES if len(parts.get(name, [])) != 1]
        for name, records in parts.items():
            for part in records:
                if (
                    part["tid"] != cb["tid"]
                    or not cb["start"] <= part["start"] <= part["end"] <= cb["end"]
                    or part["meta"].get("kind") != cb["kind"]
                    or part["meta"].get("layer") != str(cb["layer"])
                ):
                    issues.append(f"{name}_identity_or_nesting")
        if not issues:
            update, submit = parts["update_weights"][0], parts["load_submit"][0]
            for name in ("load_submit", "synchronize"):
                part = parts[name][0]
                if not update["start"] <= part["start"] <= part["end"] <= update["end"]:
                    issues.append(f"{name}_outside_update")
            for name in ("copy_submit", "load_prepare", "scale_refresh"):
                if any(not submit["start"] <= p["start"] <= p["end"] <= submit["end"] for p in parts.get(name, [])):
                    issues.append(f"{name}_outside_submit")
            if len(parts.get("load_prepare", [])) != len(parts.get("scale_refresh", [])):
                issues.append("prepare_refresh_count_mismatch")
        durations = {name: sum(p["end"] - p["start"] for p in parts.get(name, [])) for name in SINGLE_STAGES}
        prepared = len(parts.get("load_prepare", []))
        load_class = "unknown" if issues else ("with_prepare" if prepared else "zero_prepare")
        visits[cb["kind"], cb["layer"]] += 1
        annotated.append(
            dict(
                cb,
                load_class=load_class,
                prepared=prepared,
                issues=issues,
                visit=visits[cb["kind"], cb["layer"]],
                durations_us=durations,
                load_wait_us=None if issues else durations["load_submit"] + durations["synchronize"],
            )
        )
    orphan = set(stages) - seen
    if orphan:
        raise ValueError(f"存在没有外层回调的加载分项：{len(orphan)}个call")
    return annotated


def validate_report(report, callbacks):
    """Aggregate planned loads independently check prepare marker coverage, per kind/rows group."""
    issues = []
    if report is None:
        return ["缺少profile_report.json，zero_prepare仅为标记证据"]
    for name in ("errors", "dropped", "mstx_errors"):
        if report.get(name) != 0:
            issues.append(f"report.{name}={report.get(name)!r}")
    if report.get("count") != len(callbacks) or report.get("mstx_enabled") is not True:
        issues.append("report回调总数或MSTX启用状态不符")
    if report.get("load_backend") != "python" or report.get("h2d_backend") not in (None, "torch"):
        issues.append("未验证原Python/Torch加载路径")
    groups = report.get("groups", [])
    if not groups or sum(group["count"] for group in groups) != len(callbacks):
        issues.append("report分组计数不完整")
    # All report groups use a separate row count; rows are recovered from stage metadata below.
    for group in groups:
        selected = [cb for cb in callbacks if cb["kind"] == group["kind"] and cb["rows"] == str(group["token_rows"])]
        expected = group.get("loads", {}).get("mean")
        if len(selected) != group["count"] or not isinstance(expected, (float, int)):
            issues.append(f"{group['kind']}/{group['token_rows']}分组不匹配")
        elif not math.isclose(sum(cb["prepared"] for cb in selected), expected * group["count"], abs_tol=0.01):
            issues.append(f"{group['kind']}/{group['token_rows']}准备次数与实际加载数不符")
    if any(cb["issues"] for cb in callbacks):
        issues.append("部分回调缺失/冲突的分项标记")
    return issues


def find_episodes(tasks, callbacks):
    """Use the union of ALL tasks on the selected stream, including Transpose and events."""
    ordered = sorted(tasks, key=lambda t: (t["start"], t["end"], t["event_index"]))
    intervals = [(t["start"], t["end"]) for t in ordered]
    starts = [cb["start"] for cb in callbacks]
    boundaries, _ = copy_gap_boundaries(intervals)
    episodes = []
    for left, right, duration in boundaries:
        start, end = intervals[left][1], intervals[right][0]
        if duration < MIN_GAP_US:
            continue
        first, last = bisect.bisect_left(starts, start), bisect.bisect_right(starts, end)
        indices = [i for i in range(first, last) if callbacks[i]["end"] <= end]
        if len(indices) < 2:
            continue
        kinds = collections.Counter(callbacks[i]["load_class"] for i in indices)
        episodes.append(
            dict(
                start=start,
                end=end,
                gap_us=duration,
                indices=indices,
                classes=dict(kinds),
                in_bubble_band=duration < LONG_GAP_US,
                same_layer_visit=len({callbacks[i]["visit"] for i in indices}) == 1,
                left_task=ordered[left],
                right_task=ordered[right],
            )
        )
    return episodes


def host_pairs(callbacks, coverage, episodes):
    episode_by_index = {i: e for e, episode in enumerate(episodes) for i in episode["indices"]}
    pairs = []
    for i, (left, right) in enumerate(zip(callbacks, callbacks[1:])):
        direction = pair_direction(left, right)
        if direction not in ("prefetch->reactive", "reactive->prefetch"):
            continue
        duration = right["start"] - left["end"]
        both_in_episode = i in episode_by_index and episode_by_index.get(i + 1) == episode_by_index[i]
        pairs.append(
            dict(
                direction=direction,
                layer_pair=[left["layer"], right["layer"]],
                class_pair=f"{left['load_class']}->{right['load_class']}",
                visit_pair=[left["visit"], right["visit"]],
                prepared_pair=[left["prepared"], right["prepared"]],
                left_call=left["call"],
                right_call=right["call"],
                gap_us=duration,
                compute_overlap_us=coverage.overlap(left["end"], right["start"]),
                in_multi_callback_gap=both_in_episode,
            )
        )
    return pairs


def analyze(path, stream_name, report=None, progress=print):
    before = path.stat()
    progress(f"READ 1/2: {path}")
    counts, names, callbacks, _ = read_offload_header(path)
    key = select_track(counts, names, stream_name)
    callbacks, _ = select_host_callbacks(callbacks)
    progress("READ 2/2: 按call核对加载分项、设备空泡及其他流计算候选")
    stages, tasks, compute = read_details(path, key, callbacks)
    callbacks = annotate_callbacks(callbacks, stages)
    for cb in callbacks:
        updates = stages[cb["call"]].get("update_weights", [])
        cb["rows"] = updates[0]["meta"].get("rows") if len(updates) == 1 else None
    integrity = validate_report(report, callbacks)
    coverage = Coverage((t["start"], t["end"]) for t in compute)
    episodes = find_episodes(tasks, callbacks)
    pairs = host_pairs(callbacks, coverage, episodes)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("读取中trace发生变化，本次结果不使用")
    return dict(
        trace=str(path),
        track=dict(pid=key[0], tid=key[1], name=names[key]),
        integrity_issues=integrity,
        callbacks=callbacks,
        pairs=pairs,
        episodes=episodes,
        compute=compute,
    )


def stats_text(values):
    summary = outlier_summary(values)
    raw, clean = summary["raw"], summary["filtered"]
    if raw is None:
        return "n=0"
    return (
        f"n={raw['n']} mean/p50/p95={raw['mean']:.3f}/{raw['p50']:.3f}/{raw['p95']:.3f} "
        f"IQR_mean={clean['mean']:.3f} removed={summary['removed_count']}"
    )


def render(result):
    callbacks, pairs, episodes = result["callbacks"], result["pairs"], result["episodes"]
    yield f"TRACK {result['track']}; callbacks={len(callbacks)}"
    yield f"MARKER_INTEGRITY {'PASS' if not result['integrity_issues'] else result['integrity_issues']}"
    yield "zero_prepare=无专家准备标记；完整性PASS时按原Python加载代码对应0个计划加载，仍不等于已证明全部命中。"
    for kind in ("prefetch", "reactive"):
        selected = [cb for cb in callbacks if cb["kind"] == kind]
        counts = collections.Counter(cb["load_class"] for cb in selected)
        short = sum(cb["load_wait_us"] is not None and cb["load_wait_us"] < SHORT_LOAD_US for cb in selected)
        layers = collections.Counter(cb["layer"] for cb in selected if cb["load_class"] == "zero_prepare")
        yield (
            f"CALLBACKS {kind}: {dict(counts)}; load+wait<10us={short}; "
            f"zero_prepare_layers={dict(sorted(layers.items()))}"
        )
    yield (
        "HOST_GAPS_US direction class_pair | 原始n/mean/p50/p95；IQR仅辅助；other_compute为间隔内其他流计算候选并集均值"
    )
    for direction in ("prefetch->reactive", "reactive->prefetch"):
        selected = [p for p in pairs if p["direction"] == direction]
        yield f"{direction} ALL | {stats_text([p['gap_us'] for p in selected])}"
        for load_class in sorted({p["class_pair"] for p in selected}):
            part = [p for p in selected if p["class_pair"] == load_class]
            overlap = sum(p["compute_overlap_us"] for p in part) / len(part)
            yield f"{direction} {load_class} | {stats_text([p['gap_us'] for p in part])}; other_compute={overlap:.3f}"
        for inside in (True, False):
            part = [p["gap_us"] for p in selected if p["in_multi_callback_gap"] == inside]
            yield f"{direction} {'INSIDE' if inside else 'OUTSIDE'}_MULTI_CALLBACK_GAP | {stats_text(part)}"
    regular = [e for e in episodes if e["in_bubble_band"]]
    same_visit = [e for e in regular if e["same_layer_visit"]]
    yield (
        f"MULTI_CALLBACK_GAPS count_10_to_5000us={len(regular)}; same_layer_visit={len(same_visit)}; "
        f"count_ge_5000us={len(episodes) - len(regular)}; "
        f"all_zero_prepare={sum(set(e['classes']) == {'zero_prepare'} for e in regular)}"
    )
    origin = callbacks[0]["start"]
    for index, episode in enumerate(sorted(regular, key=lambda e: abs(e["gap_us"] - TARGET_GAP_US))[:MAX_EXAMPLES], 1):
        start, end = episode["start"], episode["end"]
        overlapping = [t for t in result["compute"] if t["start"] < end and t["end"] > start]
        coverage = Coverage((t["start"], t["end"]) for t in overlapping).overlap(start, end)
        names = collections.Counter(t["name"] for t in overlapping).most_common(4)
        yield (
            f"EPISODE {index}: ts_us={start:.3f}..{end:.3f}; relative_to_first_callback_us="
            f"{start - origin:.3f}..{end - origin:.3f}; gap={episode['gap_us']:.3f}; "
            f"callbacks={len(episode['indices'])}; classes={episode['classes']}; other_compute_union={coverage:.3f}"
        )
        yield f"  bounds={episode['left_task']['name']} -> {episode['right_task']['name']}; compute_names={names}"
        for cb_index in episode["indices"][:MAX_EXAMPLE_CALLBACKS]:
            cb = callbacks[cb_index]
            previous = callbacks[cb_index - 1] if cb_index else None
            before = cb["start"] - previous["end"] if previous else None
            yield (
                f"  call={cb['call']} {cb['kind']} layer={cb['layer']} visit={cb['visit']} "
                f"host_us={cb['end'] - cb['start']:.3f} prepare={cb['prepared']} "
                f"submit_us={cb['durations_us']['load_submit']:.3f} sync_us={cb['durations_us']['synchronize']:.3f} "
                f"load+wait_us={cb['load_wait_us']} preceding_host_gap_us={before} issues={cb['issues']}"
            )
        if len(episode["indices"]) > MAX_EXAMPLE_CALLBACKS:
            yield f"  其余{len(episode['indices']) - MAX_EXAMPLE_CALLBACKS}个回调未逐行打印"


def discover(root):
    if root.is_file():
        return [("single_trace", root, None)]
    roots = [root] if (root / "profile").is_dir() else [root / "tokens5", root / "tokens128"]
    records = []
    for run in roots:
        for case, policy in CASES:
            traces = sorted((run / "profile" / case / policy).rglob("trace_view.json"))
            if not traces:
                records.append((f"{run.name}/{case}", None, None))
            for trace in traces:
                report = run / case / "profile_report.json"
                records.append((f"{run.name}/{case}", trace, report if report.is_file() and len(traces) == 1 else None))
                if len(records) > MAX_TRACES:
                    raise ValueError("超过文件上限；请指定单个四组实验目录")
    return records


def compare_pairs(baseline, candidate):
    """Align layer visits with equal load counts; this does not prove equal expert IDs."""
    if baseline["integrity_issues"] or candidate["integrity_issues"]:
        yield "MATCHED_COMPARE skipped: 分项完整性未通过"
        return
    for name in ("request_specs", "digests", "usage"):
        if not baseline["identity"].get(name) or baseline["identity"][name] != candidate["identity"].get(name):
            yield f"MATCHED_COMPARE skipped: {name}缺失或不同"
            return

    def keyed(pairs):
        return {(p["direction"], *p["layer_pair"], *p["visit_pair"]): p for p in pairs}

    left, right = keyed(baseline["pairs"]), keyed(candidate["pairs"])
    if len(left) != len(baseline["pairs"]) or len(right) != len(candidate["pairs"]):
        yield "MATCHED_COMPARE skipped: 层访问键不唯一"
        return
    matched = collections.defaultdict(list)
    excluded = len(set(left) ^ set(right))
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        if a["prepared_pair"] != b["prepared_pair"] or a["class_pair"] != b["class_pair"]:
            excluded += 1
            continue
        matched[a["direction"], a["class_pair"]].append((a["gap_us"], b["gap_us"]))
    yield (
        f"MATCHED_COMPARE excluded_pairs={excluded}; 同层访问序号、前后加载数量和请求输出一致；"
        "不证明专家ID或缓存初态相同，不作因果检验"
    )
    for (direction, kind), pairs in sorted(matched.items()):
        a_mean = sum(a for a, _ in pairs) / len(pairs)
        b_mean = sum(b for _, b in pairs) / len(pairs)
        yield (
            f"MATCHED_US {direction} {kind}: n={len(pairs)} baseline={a_mean:.3f} "
            f"candidate={b_mean:.3f} delta={b_mean - a_mean:+.3f}; raw，不剔除"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="四组根目录、同时含tokens5/tokens128的父目录，或单个trace")
    parser.add_argument("--stream-name", default="Stream 56")
    parser.add_argument("--output", type=Path, help="统一摘要路径，默认实验目录下empty_load_gap_summary.txt")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    output = args.output or (root.parent if root.is_file() else root) / "empty_load_gap_summary.txt"
    records = discover(root)
    if output.resolve() in {p.resolve() for _, p, _ in records if p} or output.exists():
        raise ValueError(f"摘要路径已存在或指向输入；用--output指定新文件：{output}")
    failures = 0
    comparisons = collections.defaultdict(list)
    with output.open("x", encoding="utf-8") as sink:

        def emit(line):
            print(line, flush=True)
            sink.write(line + "\n")
            sink.flush()

        emit("只读取已有采集；不启动模型，不改统计阈值或原报告。仅写本摘要；单位us。")
        emit("Host空载分组不使用时长阈值；小于10us仅辅助。加载流包含Transpose等全部占用任务。")
        emit("其他流计算候选重叠是时间线证据，不证明设备因果依赖；层visit序号不是输出token序号。")
        for label, path, report_path in records:
            emit(f"\nCASE {label}")
            try:
                if path is None:
                    raise ValueError("缺少trace，未伪造零样本")
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else None
                result = analyze(path, args.stream_name, report, progress=lambda line: print(line, flush=True))
                emit(f"TRACE {path}")
                for line in render(result):
                    emit(line)
                failures += bool(result["integrity_issues"])
                # Retain only the small Host-pair records across traces, not device tasks.
                comparisons[label].append(
                    dict(
                        pairs=result["pairs"],
                        integrity_issues=result["integrity_issues"],
                        identity={name: (report or {}).get(name) for name in ("request_specs", "digests", "usage")},
                    )
                )
                del result
            except (OSError, ValueError, TypeError, KeyError) as exc:
                emit(f"FAILED {type(exc).__name__}: {exc}")
                failures += 1
        for label, candidates in comparisons.items():
            if not label.endswith("/mte_combined") or len(candidates) != 1:
                continue
            for baseline in ("callback_legacy", "callback", "mte_default"):
                baseline_label = label.rsplit("/", 1)[0] + "/" + baseline
                if len(comparisons.get(baseline_label, [])) == 1:
                    emit(f"\nCOMPARE {label} vs {baseline_label}")
                    for line in compare_pairs(comparisons[baseline_label][0], candidates[0]):
                        emit(line)
        emit(f"SUMMARY traces={len(records)} incomplete={failures}; file={output}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
