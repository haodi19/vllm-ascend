# SPDX-License-Identifier: Apache-2.0
"""离线时延统计：保留原值，统一提供3倍IQR敏感性结果。"""

import math
import statistics

MIN_FILTER_SAMPLES = 8
MAX_REMOVED_DISPLAY = 8


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def stats(values):
    if not values:
        return None
    return dict(
        n=len(values),
        mean=statistics.mean(values),
        p50=percentile(values, 0.5),
        p95=percentile(values, 0.95),
        max=max(values),
        sd=statistics.pstdev(values),
        total=sum(values),
    )


def filtered_stats(values):
    """统一事后3×IQR规则；小样本/零IQR不自动剔除。"""
    values = list(values)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("样本必须为有限非负数")
    raw = stats(values)
    if len(values) < MIN_FILTER_SAMPLES:
        return raw, raw, [], None, "small_n"
    q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
    if q3 == q1:
        return raw, raw, [], None, "zero_iqr"
    bounds = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    kept = [x for x in values if bounds[0] <= x <= bounds[1]]
    removed = [(i + 1, x) for i, x in enumerate(values) if not bounds[0] <= x <= bounds[1]]
    return raw, stats(kept), removed, bounds, "ok"


def outlier_summary(values):
    raw, clean, removed, bounds, reason = filtered_stats(values)
    return dict(
        rule="iqr3_linear_v1",
        reason=reason,
        raw=raw,
        filtered=clean,
        removed_count=len(removed),
        bounds=bounds,
        removed_largest=sorted(removed, key=lambda x: x[1], reverse=True)[:MAX_REMOVED_DISPLAY],
    )


def group_outliers(group):
    """仅消费真实逐回调样本；先逐样本求和，再独立筛选派生指标。"""
    metrics = group.get("metric_samples_us", {})
    gaps = group.get("gap_samples_us", {})
    if not metrics:
        return {}
    if set(metrics) & set(gaps):
        raise ValueError("Duplicate timing sample metric")
    columns = {**metrics, **gaps}
    if any(not isinstance(v, list) or len(v) != group["count"] for v in columns.values()):
        raise ValueError("Timing sample count mismatch")
    if "load_submit" in metrics and "synchronize" in metrics:
        columns["load_and_wait"] = [a + b for a, b in zip(metrics["load_submit"], metrics["synchronize"])]
    return {name: outlier_summary(values) for name, values in columns.items()}


def enrich_report(report):
    window = report.get("request_window")
    if window and "samples_us" in window:
        if len(window["samples_us"]) != window["count"]:
            raise ValueError("Request sample count mismatch")
        window["outliers_us"] = outlier_summary(window["samples_us"])
    if "request_samples_us" in report:
        report["request_outliers_us"] = outlier_summary(report["request_samples_us"])
    for group in report.get("groups", []):
        group["outliers_us"] = group_outliers(group)
