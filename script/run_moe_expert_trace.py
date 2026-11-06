#!/usr/bin/env python3
"""一次运行最终MTE配置，导出逐请求、逐层专家与H2D诊断。"""

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

from run_moe_host_sync import REPOSITORY_ROOT, SummaryTee, failure_excerpt, parse_args, run
from run_verified_cpp_tests import launch_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-script", type=Path, default=REPOSITORY_ROOT / "script/test_graph2.sh")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--port", type=int, default=8155)
    options = parser.parse_args()
    args = parse_args(
        [
            "--npu",
            options.npu,
            "--backend",
            "mte",
            "--cache-policy-backend",
            "cpp",
            "--worker-scan",
            "allocated",
            "--worker-probe",
            "atomic",
            "--worker-idle",
            "yield",
            "--diagnose-prefetch-wait-at-join",
            "--no-load-diagnostics",
            "--skip-profile",
            "--expert-trace",
            "--requests",
            str(options.requests),
            "--warmup-requests",
            str(options.warmup_requests),
            "--test-script",
            str(options.test_script),
            "--port",
            str(options.port),
        ]
    )
    root = options.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = 1
    with (
        (root / "summary.txt").open("w", encoding="utf-8") as summary,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, summary)),
    ):
        print("MTE普通流＋cpp LRC＋allocated/atomic/yield；消费者处严格等待。")
        print("诊断请求：预热后开启专家记录；不采profile，不用于TPOT/时延比较；max_tokens=5。")
        try:
            result = run(args, root)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
        finally:
            output = root / "expert_trace.txt"
            with output.open("w", encoding="utf-8") as combined:
                print(f"RUN={root}; EXIT={result}", file=combined)
                # The original launch command is emitted into this runner's summary.
                summary.flush()
                for line in (root / "summary.txt").read_text(encoding="utf-8").splitlines():
                    if "Effective serve command:" in line:
                        print(line, file=combined)
                for service in root.glob("*/inference_*.log"):
                    print(f"SERVICE={service}", file=combined)
                    text = service.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        if re.search(
                            r"(?i)memory|activation|KV cache|Actual usage|Model loading took|"
                            r"PREFILL_POOL|DECODE-STATS.*topology|MOE-MTE WORKER",
                            line,
                        ):
                            print(line, file=combined)
                    if result:
                        print(failure_excerpt(text), file=combined)
                traces = sorted(root.glob("*/expert_trace_*.txt"))
                for trace in traces:
                    print(f"\nSOURCE={trace}", file=combined)
                    combined.write(trace.read_text(encoding="utf-8"))
                if len(traces) != options.requests:
                    print("INCOMPLETE：记录请求数不足；不能作为完整trace。", file=combined)
                # Preserve exact current model configuration alongside actual command.
                try:
                    _, _, _, model = launch_spec(
                        options.test_script.read_text(encoding="utf-8"),
                        "cpp",
                        int(options.npu),
                        options.port,
                        args.timing_port,
                        root / "unused_profile",
                        mstx=False,
                    )
                    config = json.loads(model.read_text(encoding="utf-8"))
                    print("MODEL_CONFIG " + json.dumps(config, ensure_ascii=False), file=combined)
                except (OSError, ValueError) as exc:
                    print(f"MODEL_CONFIG unavailable: {exc}", file=combined)
            print(f"完整诊断文本：{output}；请复制其内容返回。")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
