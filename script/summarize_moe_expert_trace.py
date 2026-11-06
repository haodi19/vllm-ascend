#!/usr/bin/env python3
"""只读精简专家trace：配置、内存、逐轮逐层计数及少量ID样例。"""

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from read_moe_run_facts import parse_command

MEMORY = re.compile(r"(?i)Actual usage|Model loading took|KV cache (memory|size)|peak.*activation|PREFILL_POOL")
CONFIG_KEYS = (
    "cache_policy_enabled",
    "cache_policy_backend",
    "cache_recent_window",
    "cache_ema_beta",
    "cache_recent_weight",
    "cache_ema_weight",
    "cache_router_weight",
    "cache_age_weight",
    "expert_prefetch_enabled",
    "expert_prefetch_num",
    "expert_prefetch_tokens",
    "hot_expert_preload",
    "hot_experts_file",
    "expert_substitution_enabled",
    "host_sync_backend",
    "host_sync_load_stream",
    "host_sync_scan_mode",
    "host_sync_idle_probe",
    "host_sync_idle_mode",
    "host_sync_diagnose_prefetch_wait_at_join",
)


def emit(name, value):
    print(name + " " + json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def read_summary(path, selected=1):
    data = dict(
        headers={},
        command={},
        memory=deque(maxlen=10),
        model={},
        totals={},
        rounds=[],
        counts=defaultdict(Counter),
        examples={},
        failed=[],
        parse_errors=[],
        blocks=0,
        event_count=0,
        failed_count=0,
        run_status=None,
    )
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            try:
                if line.startswith("RUN="):
                    data["run_status"] = line
                elif "Effective serve command:" in line:
                    command = parse_command(line)
                    data["command"] = {
                        k: v for k, v in command.items() if k not in ("expert_offload_config", "profile_path")
                    }
                elif line.startswith("MODEL_CONFIG unavailable:"):
                    data["model"] = {"unavailable": line.split(":", 1)[1].strip()}
                elif line.startswith("MODEL_CONFIG "):
                    model = json.loads(line.split(" ", 1)[1])
                    keys = (
                        "model_type",
                        "num_hidden_layers",
                        "n_routed_experts",
                        "num_experts_per_tok",
                        "n_shared_experts",
                        "quantization_config",
                    )
                    data["model"] = {k: model[k] for k in keys if k in model}
                elif line.startswith("COMPLETE "):
                    data["blocks"] += 1
                    if data["blocks"] == selected:
                        data["headers"]["COMPLETE"] = json.loads(line.split(" ", 1)[1])
                elif data["blocks"] == 0 and MEMORY.search(line):
                    data["memory"].append(line[:700])
                elif data["blocks"] == selected:
                    tag, _, body = line.partition(" ")
                    if tag in ("ERRORS", "DROPPED", "METADATA", "REQUEST"):
                        data["headers"][tag] = json.loads(body)
                    elif tag == "ROUND":
                        row = json.loads(body)
                        data["rounds"].append({k: row[k] for k in ("visit", "complete_layer_sweep", "reactive_loads")})
                    elif tag == "TOTAL":
                        kind, _, payload = body.partition(" ")
                        data["totals"][kind.removeprefix("kind=")] = json.loads(payload)
                    elif tag == "EVENT":
                        record = json.loads(body)
                        data["event_count"] += 1
                        if record["status"] != "completed":
                            data["failed_count"] += 1
                            if len(data["failed"]) < 3:
                                data["failed"].append(
                                    {k: record.get(k) for k in ("event_id", "kind", "layer", "status", "error")}
                                )
                            continue
                        kind = record["kind"]
                        key = (kind, record["layer_visit"], record["layer"])
                        loads = record.get("full_pool_experts", 0) if kind == "prefill_pool" else len(record["loads"])
                        data["counts"][key].update(
                            dict(
                                calls=1,
                                loaded_experts=loads,
                                h2d_bytes=record["logical_h2d_bytes"],
                                hits=len(record.get("hit_ids", [])),
                                missing=len(record.get("missing_ids", [])),
                            )
                        )
                        if kind in ("prefetch", "reactive"):
                            example = {
                                k: record.get(k)
                                for k in (
                                    "event_id",
                                    "kind",
                                    "layer_visit",
                                    "layer",
                                    "hit_ids",
                                    "missing_ids",
                                    "loaded_ids",
                                    "evicted_ids",
                                    "prefetched_ids_used",
                                    "preceding_prefetch_event",
                                )
                            }
                            if loads and (kind, "first_load") not in data["examples"]:
                                data["examples"][kind, "first_load"] = example
                            previous = data["examples"].get((kind, "most_loads"))
                            if previous is None or loads > len(previous.get("loaded_ids") or []):
                                data["examples"][kind, "most_loads"] = example
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                data["parse_errors"].append(f"line={number}: {type(exc).__name__}")
    return data


def report(data, selected):
    headers = data["headers"]
    totals = defaultdict(Counter)
    rounds = defaultdict(Counter)
    for (kind, visit, _layer), values in data["counts"].items():
        totals[kind].update(values)
        rounds[kind, visit].update(values)
    matched = dict(totals) == data["totals"] and bool(totals)
    complete = (
        headers.get("COMPLETE") is True
        and headers.get("ERRORS") == 0
        and headers.get("DROPPED") == 0
        and not data["parse_errors"]
        and not data["failed_count"]
        and matched
    )
    emit(
        "CHECK",
        dict(
            request_record=selected,
            available_records=data["blocks"],
            complete=complete,
            events=data["event_count"],
            totals_match=matched,
            errors=headers.get("ERRORS"),
            dropped=headers.get("DROPPED"),
            failed_events=data["failed_count"],
        ),
    )
    if data["run_status"]:
        print(data["run_status"])
    if data["command"]:
        emit("SERVE", data["command"])
    meta = headers.get("METADATA", {})
    for label, keys in (
        (
            "CONFIG",
            (
                "runtime",
                "moe_layers",
                "experts_per_layer",
                "top_k",
                "decode_cache_slots",
                "prefill_pool_layers",
                "prefetch_load_cap",
            ),
        ),
        (
            "HOST_MEMORY",
            ("cpu_weight_storage_bytes", "cpu_quant_storage_bytes", "host_memory_fields", "host_memory_snapshot"),
        ),
    ):
        emit(label, {k: meta[k] for k in keys if k in meta})
    offload = meta.get("offload_config", {})
    emit("CACHE", {k: offload[k] for k in CONFIG_KEYS if k in offload})
    emit("MODEL", data["model"])
    request = headers.get("REQUEST", {})
    emit("REQUEST", {k: request.get(k) for k in ("index", "usage", "failed") if k in request})
    print("MEMORY_LOG（原日志最多10行；缺项不推算）：")
    print("\n".join(data["memory"]) or "未找到；请查原服务启动日志。")
    print("以下只汇总completed事件；专家重复加载重复计数，字节是逻辑H2D字节。")
    for kind, values in sorted(totals.items()):
        emit("TOTAL " + kind, dict(values, h2d_GiB=round(values["h2d_bytes"] / 1024**3, 4)))
    print("visit是同类操作的层访问轮次，不是输出token；完整按需层遍历核查：")
    for row in data["rounds"]:
        emit("ROUND", row)
    for (kind, visit), values in sorted(rounds.items()):
        emit(f"VISIT {kind}/{visit}", dict(values, h2d_MiB=round(values["h2d_bytes"] / 1024**2, 3)))
    print("逐层Decode表：每格=命中/缺失/加载；列名P为预取、R为按需；-表示未记录。")
    columns = sorted({(visit, kind) for kind, visit, _ in data["counts"] if kind in ("prefetch", "reactive")})
    layers = sorted({layer for kind, _, layer in data["counts"] if kind in ("prefetch", "reactive")})
    print("layer " + " ".join(f"{'P' if kind == 'prefetch' else 'R'}{visit}" for visit, kind in columns))
    for layer in layers:
        cells = []
        for visit, kind in columns:
            values = data["counts"].get((kind, visit, layer))
            cells.append(f"{values['hits']}/{values['missing']}/{values['loaded_experts']}" if values else "-")
        print(f"{layer:02d} " + " ".join(cells))
    print("ID样例（每类首次加载及加载最多事件，去重；完整ID保留在原trace）：")
    seen = set()
    for example in data["examples"].values():
        if example["event_id"] in seen:
            continue
        seen.add(example["event_id"])
        for key, value in list(example.items()):
            if isinstance(value, list) and len(value) > 32:
                example[key] = dict(first32=value[:32], count=len(value))
        emit("EXAMPLE", example)
    if not complete:
        emit("INCOMPLETE", dict(parse_errors=data["parse_errors"][:5], failed=data["failed"]))
        print("只能作为部分记录，不能据此报告完整请求加载量。")
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="实验目录或expert_trace.txt")
    parser.add_argument("--request", type=int, default=1, help="选择文件中第几份请求记录，默认1")
    args = parser.parse_args(argv)
    if args.request < 1:
        parser.error("--request必须大于0")
    path = args.path / "expert_trace.txt" if args.path.is_dir() else args.path
    print(f"只读已有trace，不导入Torch、不启动模型、不写文件。FILE={path}")
    try:
        return report(read_summary(path, args.request), args.request)
    except (OSError, UnicodeError) as exc:
        print(f"读取失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
