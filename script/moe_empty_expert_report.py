"""Verify zero-load reasons from expert IDs, independently of profiler clocks."""

import json
from collections import Counter, defaultdict

from moe_expert_trace_report import layer_rounds


def classify(record):
    if record.get("status") != "completed" or not record.get("h2d_wait_returned"):
        return "unknown_incomplete"
    required = ("needed_ids", "hit_ids", "missing_ids", "resident_before", "loads", "copies")
    if any(key not in record for key in required):
        return "unknown_missing_fields"
    needed = set(record["needed_ids"])
    resident = {int(e) for e in record["resident_before"]}
    if set(record["hit_ids"]) != needed & resident or set(record["missing_ids"]) != needed - resident:
        return "unknown_inconsistent_sets"
    if record["loads"]:
        return "loaded"
    if record["copies"]:
        return "unknown_copies_without_loads"
    if not needed:
        return "empty_candidates"
    if needed <= resident:
        return "all_needed_resident"
    # Missing IDs alone cannot distinguish a policy cap from victim rejection.
    return "missing_but_not_loaded"


def report(trace):
    yield "EMPTY_EXPERT_CHECK：按专家ID集合核验，不根据回调时长或Host/NPU重叠判因。"
    yield "本次诊断不采性能；新请求不能自动视作旧profile同一轮。层号从0开始。"
    if trace.get("complete") is not True or trace.get("errors") != 0 or trace.get("dropped") != 0:
        yield "EMPTY_CHECK=INCOMPLETE；记录不完整，不确认缓存原因。"
        return
    metadata = trace["metadata"]
    if metadata.get("offload_config", {}).get("expert_substitution_enabled", False):
        yield "EMPTY_CHECK=UNSUPPORTED；专家替换开启，入口路由不代表最终所需集合。"
        return
    records = [r for r in trace["records"] if r["kind"] in ("prefetch", "reactive")]
    classes = {r["event_id"]: classify(r) for r in records}
    if len(classes) != len(records) or any(c.startswith("unknown") for c in classes.values()):
        yield "EMPTY_CHECK=INCOMPLETE；事件重复、字段缺失或集合核验失败。"
        return
    sweeps = layer_rounds(trace)
    yield "SWEEPS " + json.dumps(sweeps, ensure_ascii=False)
    yield "REASONS " + json.dumps(dict(Counter(classes.values())), ensure_ascii=False)
    yield (
        "all_needed_resident=非空所需集合全部已驻留；empty_candidates=所需集合为空；"
        "missing_but_not_loaded=仍有缺失，不能算全命中。"
    )
    grouped = defaultdict(list)
    for r in records:
        grouped[r["kind"], r["layer"]].append(r)
    yield "| kind | layer | 每次访问 visit:加载数:原因 | 相邻访问所需ID完全相同 |"
    yield "| --- | ---: | --- | --- |"
    for (kind, layer), rows in sorted(grouped.items()):
        if all(classes[r["event_id"]] == "loaded" for r in rows):
            continue
        values = "; ".join(f"{r['layer_visit']}:{len(r['loads'])}:{classes[r['event_id']]}" for r in rows)
        same = [set(a["needed_ids"]) == set(b["needed_ids"]) for a, b in zip(rows, rows[1:])]
        yield f"| {kind} | {layer} | {values} | {same or '仅一次访问'} |"
    # Keep original callback adjacency; do not join across loaded work or sweeps.
    chains, chain = [], []
    for r in records:
        empty = classes[r["event_id"]] != "loaded"
        if chain and (not empty or r["layer_visit"] != chain[-1]["layer_visit"]):
            chains.append(chain)
            chain = []
        if empty:
            chain.append(r)
    if chain:
        chains.append(chain)
    for chain in chains:
        if len(chain) < 2:
            continue
        yield "EMPTY_CHAIN " + json.dumps(
            [
                {"visit": r["layer_visit"], "layer": r["layer"], "kind": r["kind"], "reason": classes[r["event_id"]]}
                for r in chain
            ],
            ensure_ascii=False,
        )
    # Exact IDs for the previously observed cluster and its immediate neighbors.
    for r in records:
        if 14 <= r["layer"] <= 19:
            fields = (
                "event_id",
                "kind",
                "layer_visit",
                "layer",
                "needed_ids",
                "hit_ids",
                "missing_ids",
                "resident_before",
                "loads",
                "still_missing_ids",
            )
            yield "CLUSTER_IDS " + json.dumps({k: r.get(k) for k in fields}, ensure_ascii=False)
    complete = bool(sweeps) and all(s["complete_layer_sweep"] for s in sweeps)
    yield (
        f"EMPTY_CHECK={'PASS' if complete else 'PARTIAL_SWEEPS'}；"
        "PASS仅表示诊断记录可解释，具体原因以REASONS和逐层记录为准。"
    )


def read_trace_text(path):
    trace = {"records": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        tag, _, body = line.partition(" ")
        if tag in ("COMPLETE", "ERRORS", "DROPPED", "METADATA", "REQUEST"):
            trace[tag.lower()] = json.loads(body)
        elif tag == "EVENT":
            trace["records"].append(json.loads(body))
    return trace
