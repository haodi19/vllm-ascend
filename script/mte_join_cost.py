# SPDX-License-Identifier: Apache-2.0
"""Prepare an unprofiled comparison and compare measured Host callback costs."""

import hashlib
import json
import math
import re
import shlex

METRICS = (
    "total",
    "observe",
    "choose_victims",
    "load_prepare",
    "copy_submit",
    "scale_refresh",
    "load_submit",
    "synchronize",
)
LOAD_STATS = ("mean", "p50", "p95", "p99", "min", "max")
EMPTY_TEXT_SHA256 = hashlib.sha256(b"").hexdigest()


def prepare_cost_source(source: str) -> str:
    """Keep model settings, remove diagnostic interventions, override debug exports."""
    starts = list(re.finditer(r"(?m)^vllm\s+serve\s+", source))
    if len(starts) != 1:
        raise ValueError("成本对照要求脚本只有一条字面的 vllm serve 命令")
    start = starts[0].start()
    logical = re.sub(r"\\\r?\n", " ", source[start:] + "\n")
    lines = [line for line in logical.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise ValueError("vllm serve 后包含其他命令或未续行内容")
    lexer = shlex.shlex(lines[0], posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    words = list(lexer)
    if any(word and set(word) <= set(";&|<>") for word in words):
        raise ValueError("成本对照不接受 serve 命令后的管道、重定向或其他命令")
    if words[:2] != ["vllm", "serve"] or len(words) < 3 or words[2].startswith("--"):
        raise ValueError("缺少字面的模型路径")
    if words.count("--additional-config") != 1:
        raise ValueError("需要唯一的 --additional-config")
    index = words.index("--additional-config") + 1
    try:
        config = json.loads(words[index])
        offload = config["expert_offload_config"]
        if not isinstance(offload, dict):
            raise TypeError("expert_offload_config must be an object")
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("--additional-config 缺失或不是有效的专家卸载配置") from exc
    for key in list(offload):
        if key.startswith("host_sync_diagnose_"):
            del offload[key]
    offload["moe_offload_debug"] = False
    words[index] = json.dumps(config, ensure_ascii=False)
    # launch_spec retains export/unset lines. Put these LAST so inherited
    # DEBUG or launch-blocking exports cannot override the comparison settings.
    overrides = (
        "unset ASCEND_LAUNCH_BLOCKING\nexport ASCEND_MODULE_LOG_LEVEL='RUNTIME=3'\nexport ASCEND_GLOBAL_LOG_LEVEL=3\n"
    )
    return source[:start].rstrip() + "\n" + overrides + shlex.join(words) + "\n"


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} 不是有效的非负测量值")
    return value


def _validate_outputs(window, label):
    if any(
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        or digest.lower() == EMPTY_TEXT_SHA256
        for digest in window["digests"]
    ):
        raise ValueError(f"{label} digests 必须是非空输出文本的有效 SHA256")
    for usage in window["usage"]:
        if (
            not isinstance(usage, dict)
            or type(usage.get("completion_tokens")) is not int
            or usage["completion_tokens"] <= 0
        ):
            raise ValueError(f"{label} usage 必须包含正数 completion_tokens")


def _groups(report, backend):
    if not isinstance(report, dict):
        raise ValueError(f"{backend} 缺少报告")
    if report.get("mstx_enabled") is not False or report.get("profile_status") != "skipped":
        raise ValueError(f"{backend} 必须关闭 MSTX 与 profiler")
    count = report.get("count")
    if type(count) is not int or count <= 0 or report.get("errors") != 0 or report.get("dropped") != 0:
        raise ValueError(f"{backend} 回调计时不完整或包含错误/丢弃样本")
    if report.get("mstx_errors", 0) != 0 or report.get("units") != "us":
        raise ValueError(f"{backend} 计时单位或标记错误不符合成本对照")
    groups = report.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{backend} 缺少分组测量")
    indexed = {}
    for group in groups:
        try:
            key = (group["kind"], group["token_rows"])
            if key[0] not in ("prefetch", "reactive") or type(key[1]) is not int or key[1] <= 0:
                raise ValueError(f"{backend} 无效分组 {key}")
            if key in indexed:
                raise ValueError(f"{backend} 重复分组 {key}")
            if type(group["count"]) is not int or group["count"] <= 0 or not group["layers"]:
                raise ValueError(f"{backend} 分组样本数量或层列表为空")
            for stat in LOAD_STATS:
                _number(group["loads"][stat], f"{backend}/{key}/loads/{stat}")
            for metric in METRICS:
                for stat in ("mean", "p95"):
                    _number(group["metrics"][metric][stat], f"{backend}/{key}/{metric}/{stat}")
            indexed[key] = group
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{backend} 分组测量字段缺失或格式错误") from exc
    if sum(group["count"] for group in indexed.values()) != count:
        raise ValueError(f"{backend} 分组数量与总回调数不一致")
    for field in ("request_specs", "digests", "usage"):
        if not isinstance(report.get(field), list) or not report[field]:
            raise ValueError(f"{backend} 缺少 {field}，不能验证请求与输出一致性")
    if len({len(report[field]) for field in ("request_specs", "digests", "usage")}) != 1:
        raise ValueError(f"{backend} 请求、输出摘要与 token 用量数量不一致")
    _validate_outputs(report, f"{backend}/计时开启")
    _number(report.get("request_us"), f"{backend}/request_us")
    try:
        _number(report["request_stats_us"]["p95"], f"{backend}/request_p95")
        untimed = report["untimed"]
        if type(untimed["count"]) is not int or untimed["count"] <= 0:
            raise ValueError(f"{backend} 未开启计时的请求窗口为空")
        if untimed["count"] != len(report["request_specs"]):
            raise ValueError(f"{backend} 计时关闭窗口请求数与原请求计划不一致")
        if len(untimed["digests"]) != untimed["count"] or len(untimed["usage"]) != untimed["count"]:
            raise ValueError(f"{backend} 未开启计时的请求窗口不完整")
        _validate_outputs(untimed, f"{backend}/计时关闭")
        _number(untimed["request_us"], f"{backend}/untimed_request_mean")
        _number(untimed["stats_us"]["p95"], f"{backend}/untimed_request_p95")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{backend} 缺少请求计时开启/关闭窗口的完整统计") from exc
    positions = report.get("engine_error_log_positions")
    if positions is not None and (
        positions.get("cleanup_marker_unique") is not True or positions.get("before_cleanup") != 0
    ):
        raise ValueError(f"{backend} 清理前有引擎错误或无法区分清理边界")
    return indexed


def print_cost_comparison(reports: dict) -> dict:
    """Print descriptive costs; never treat unequal work or sum-of-p95 as a gain."""
    if set(reports) != {"callback", "mte"}:
        raise ValueError("需要 callback 和 mte 两份完整报告")
    indexed = {backend: _groups(report, backend) for backend, report in reports.items()}
    for field in ("request_specs", "digests", "usage"):
        if reports["callback"][field] != reports["mte"][field]:
            raise ValueError(f"两组 {field} 不一致，拒绝给出性能收益比较")
    for field in ("count", "digests", "usage"):
        if reports["callback"]["untimed"][field] != reports["mte"]["untimed"][field]:
            raise ValueError(f"两组计时关闭窗口的 {field} 不一致，拒绝给出性能收益比较")
    same_keys = indexed["callback"].keys() == indexed["mte"].keys()
    result = {"units": "us", "same_output": True, "group_keys_match": same_keys, "groups": []}
    print("OUTPUT_MATCH=PASS：计时开启、关闭两窗口的两组请求计划/输出摘要/token用量均一致。")
    for backend, report in reports.items():
        positions = report.get("engine_error_log_positions")
        if positions is None:
            print(f"ENGINE_ERRORS {backend}：未提供清理边界，需由运行器另行检查。")
        else:
            print(
                f"ENGINE_ERRORS {backend} before_cleanup={positions['before_cleanup']} "
                f"after_cleanup={positions.get('after_cleanup', 'unknown')}；"
                "仅合并日志顺序，不据此归因硬件故障。"
            )
    print("搬权重相关成本：单位 us；delta=MTE-callback（负数表示测得更低）。")
    print("以下均为内部计时开启时的 Host 测量；不表示设备空泡或两个 host_callback 的间隔。")
    for key in sorted(indexed["callback"].keys() | indexed["mte"].keys()):
        first, second = (indexed[backend].get(key) for backend in ("callback", "mte"))
        if first is None or second is None:
            print(f"kind={key[0]} rows={key[1]}：一组缺失，不可作同工作量比较")
            result["groups"].append({"kind": key[0], "token_rows": key[1], "comparable": False, "missing": True})
            continue
        samples_match = first["count"] == second["count"]
        load_stats_match = all(
            math.isclose(first["loads"][stat], second["loads"][stat], rel_tol=1e-9, abs_tol=1e-9) for stat in LOAD_STATS
        )
        layers_match = sorted(first["layers"]) == sorted(second["layers"])
        comparable = same_keys and samples_match and load_stats_match and layers_match
        experts = [group["loads"]["mean"] * group["count"] for group in (first, second)]
        group_result = {
            "kind": key[0],
            "token_rows": key[1],
            "comparable": comparable,
            "samples_match": samples_match,
            "load_stats_match": load_stats_match,
            "layers_match": layers_match,
            "counts": [first["count"], second["count"]],
            "estimated_experts": experts,
            "metrics": {},
        }
        label = "样本及工作量摘要一致" if comparable else "样本或工作量不一致；仅描述，不能归因为方案收益"
        print(
            f"kind={key[0]} rows={key[1]} callbacks={first['count']}/{second['count']} "
            f"估计加载专家数={experts[0]:.3f}/{experts[1]:.3f}；{label}"
        )
        print("metric                callback_mean    mte_mean   mean_delta   callback_p95     mte_p95")
        for metric in METRICS:
            a, b = first["metrics"][metric], second["metrics"][metric]
            row = {
                "callback_mean": a["mean"],
                "mte_mean": b["mean"],
                "mean_delta": b["mean"] - a["mean"],
                "callback_p95": a["p95"],
                "mte_p95": b["p95"],
            }
            group_result["metrics"][metric] = row
            print(
                f"{metric:21} {a['mean']:13.3f} {b['mean']:11.3f} {row['mean_delta']:+12.3f} "
                f"{a['p95']:14.3f} {b['p95']:11.3f}"
            )
        a, b = (
            sum(group["metrics"][metric]["mean"] for metric in ("load_submit", "synchronize"))
            for group in (first, second)
        )
        group_result["metrics"]["load_and_wait"] = {"callback_mean": a, "mte_mean": b, "mean_delta": b - a}
        print(f"{'load_and_wait':21} {a:13.3f} {b:11.3f} {b - a:+12.3f}  p95不可由分项相加")
        result["groups"].append(group_result)
    result["request_windows"] = {}
    for enabled in (False, True):
        a, b = (reports[backend] if enabled else reports[backend]["untimed"] for backend in ("callback", "mte"))
        stats_key = "request_stats_us" if enabled else "stats_us"
        row = {
            "callback_mean": a["request_us"],
            "mte_mean": b["request_us"],
            "mean_delta": b["request_us"] - a["request_us"],
            "callback_p95": a[stats_key]["p95"],
            "mte_p95": b[stats_key]["p95"],
        }
        label = "内部计时开启" if enabled else "内部计时关闭"
        result["request_windows"]["timed" if enabled else "untimed"] = row
        print(
            f"请求 {label}（辅助，ms）：均值={row['callback_mean'] / 1000:.3f}/{row['mte_mean'] / 1000:.3f}；"
            f"delta={row['mean_delta'] / 1000:+.3f}；p95={row['callback_p95'] / 1000:.3f}/{row['mte_p95'] / 1000:.3f}"
        )
    result["comparable"] = same_keys and all(group["comparable"] for group in result["groups"])
    print("计时关闭窗口仍安装计时包装器，但收集器未激活；不是完全移除计时代码的进程。")
    print("工作量一致仅指已有分组摘要，不证明逐次命中专家相同；未采集设备间隔。服务退出错误另行检查。")
    return result
