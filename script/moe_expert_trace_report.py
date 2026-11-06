"""Readable, lossless expert-ID reports; no performance interpretation."""

import json
from collections import Counter, defaultdict


def layer_rounds(trace):
    """Verify complete layer sweeps before presenting a model-round view."""
    expected = list(range(trace["metadata"]["moe_layers"]))
    grouped = defaultdict(list)
    for record in trace["records"]:
        if record["kind"] == "reactive":
            grouped[record["layer_visit"]].append(record)
    return [
        dict(
            visit=visit,
            complete_layer_sweep=(
                [r["layer"] for r in rows] == expected and all(r["status"] == "completed" for r in rows)
            ),
            layers=[r["layer"] for r in rows],
            routed_rows_by_layer=[len(r.get("route_rows", [])) for r in rows],
            reactive_loads=sum(len(r["loads"]) for r in rows if r["status"] == "completed"),
        )
        for visit, rows in sorted(grouped.items())
    ]


def write_trace(trace, path):
    totals = defaultdict(lambda: dict(calls=0, loaded_experts=0, h2d_bytes=0, hits=0, missing=0))
    with path.open("w", encoding="utf-8") as out:
        print("专家加载诊断：不是性能测试。下列各项均来自本次请求。", file=out)
        print("层号从0开始；layer_visit=该类回调在该层的第几次访问，不是输出token序号。", file=out)
        print("路由每行是一次路由位置，可能包含MTP位置或padding；不能按行数推算接受token。", file=out)
        print("prefetch为预测专家，reactive为实际路由；loads每项=[目标槽位,专家ID]。", file=out)
        print("missing是进入回调时缺失；预取受数量上限约束，不一定全加载。", file=out)
        print("copies是Torch传输任务及逻辑源字节数，不是硬件MEMCPY数量/总线实际字节数。", file=out)
        print("completed须原始H2D等待正常返回；初始驻留ID见每层首次resident_before。", file=out)
        print("Prefill池全量加载单独列出，不混入Decode缓存命中率。", file=out)
        for key in ("complete", "errors", "dropped", "metadata", "request"):
            print(key.upper() + " " + json.dumps(trace.get(key), ensure_ascii=False), file=out)
        print(
            "完整层遍历核查：complete_layer_sweep=true时可把该layer_visit读作一次完整MoE遍历；不是一个输出token。",
            file=out,
        )
        for row in layer_rounds(trace):
            print("ROUND " + json.dumps(row), file=out)
        records = trace["records"]
        for record in records:
            print("EVENT " + json.dumps(record, ensure_ascii=False), file=out)
            if record["status"] != "completed":
                continue
            key = (record["kind"], record["layer_visit"], record["layer"])
            row = totals[key]
            row["calls"] += 1
            row["loaded_experts"] += (
                record.get("full_pool_experts", 0) if record["kind"] == "prefill_pool" else len(record["loads"])
            )
            row["h2d_bytes"] += record["logical_h2d_bytes"]
            row["hits"] += len(record.get("hit_ids", []))
            row["missing"] += len(record.get("missing_ids", []))
        grouped = defaultdict(Counter)
        print("\n逐层计数（只计completed；同一专家重复加载会重复计数）：", file=out)
        for (kind, visit, layer), values in sorted(totals.items()):
            print(f"LAYER kind={kind} layer_visit={visit} layer={layer} " + json.dumps(values), file=out)
            grouped[kind].update(values)
        for kind, values in sorted(grouped.items()):
            print(f"TOTAL kind={kind} " + json.dumps(values), file=out)
    print(f"EXPERT TRACE: {path}; complete={trace['complete']}; events={len(trace['records'])}", flush=True)
