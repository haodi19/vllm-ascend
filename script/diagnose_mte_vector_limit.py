# SPDX-License-Identifier: Apache-2.0
"""一次执行加载流分核诊断及实际 kernel 核数汇总；不改原始拷贝或等待协议。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--cores", type=int, default=28, choices=range(1, 57))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "bench_results" / f"mte_vector_limit_{uuid4().hex[:8]}"
    env = os.environ.copy()
    env["ASCEND_MODULE_LOG_LEVEL"] = "RUNTIME=0"  # Existing CANN module logging control.
    print("只改变加载流 Vector Core 上限；纯 MTE 严格等待、原始 Tensor 拷贝。非性能测试。", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "script/run_moe_host_sync.py",
            "--npu",
            str(args.npu),
            "--backend",
            "mte",
            "--diagnose-load-vector-cores",
            str(args.cores),
            "--requests",
            "1",
            "--warmup-requests",
            "1",
            "--profile-timeout",
            "30",
            "--runtime-source",
            "/home/l00992607/vllm-ascend",
            "--test-script",
            "/home/l00992607/vllm-ascend/script/test_graph2.sh",
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=env,
        check=False,
    )
    if (output / "summary.txt").is_file():
        with (output / "summary.txt").open("a", encoding="utf-8") as summary:
            summary.write(f"\nVECTOR LIMIT RUNNER EXIT={result.returncode}\n")
            summary.flush()
            subprocess.run(
                [sys.executable, "-B", "script/inspect_mte_task_serial.py", str(output), "--kernel-summary"],
                cwd=root,
                stdout=summary,
                stderr=subprocess.STDOUT,
                check=False,
            )
    print(f"返回此文件内容：{output / 'summary.txt'}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
