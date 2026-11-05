#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""读取一次四组实验已有日志；不导入框架、不启动设备、不写文件。"""

import argparse
import json
import re
import shlex
from collections import deque
from pathlib import Path

CASES = ("callback_legacy", "callback", "mte_default", "mte_combined")
FLAGS = (
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--data-parallel-size",
    "--max-num-seqs",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--gpu-memory-utilization",
    "--kv-cache-memory",
    "--seed",
    "--compilation-config",
    "--speculative-config",
)
OFFLOAD_KEYS = (
    "expert_offload",
    "enable_multi_card",
    "num_device_experts",
    "num_device_layers",
    "cache_policy_enabled",
    "cache_policy_backend",
    "expert_prefetch_enabled",
    "expert_prefetch_num",
    "hot_expert_preload",
    "hot_experts_file",
    "expert_substitution_enabled",
    "moe_offload_debug",
    "host_sync_backend",
    "host_sync_scan_mode",
    "host_sync_idle_probe",
    "host_sync_idle_mode",
    "host_sync_diagnose_prefetch_wait_at_join",
    "h2d_backend",
)
MEMORY = re.compile(
    r"Actual usage:|Available KV cache memory:|Current KV cache memory:|"
    r"GPU KV cache size:|NPU KV cache size:|Model loading took|"
    r"\[PREFILL_POOL\]|\[DECODE-STATS\] topology|\[MOE-MTE WORKER\]"
)
DECODE = re.compile(
    r"\brun\s+: model=|\boffload\s+:|\bmeasured\s+: decode|"
    r"on-demand loads /|prefetch loads|\[DECODE-STATS\].*(disabled|enabled|first sample)"
)
EXECUTION = re.compile(r"SpecDecoding metrics:|preempt(?:ed|ion)|recomput(?:e|ation)", re.IGNORECASE)


def emit(label, value):
    print(label + "=" + json.dumps(value, ensure_ascii=False))


def parse_command(line):
    args = shlex.split(line.split("Effective serve command:", 1)[1].strip())

    def option(key):
        if key not in args:
            return None
        index = args.index(key)
        return args[index + 1] if index + 1 < len(args) else None

    result = {key[2:]: option(key) for key in FLAGS if option(key) is not None}
    for key in ("compilation-config", "speculative-config"):
        if key in result:
            result[key] = json.loads(result[key])
    additional = json.loads(option("--additional-config") or "{}")
    offload = additional.get("expert_offload_config", {})
    result["expert_offload_config"] = {k: offload[k] for k in OFFLOAD_KEYS if k in offload}
    result["model_path"] = args[args.index("serve") + 1]
    profile = json.loads(option("--profiler-config") or "{}")
    result["profile_path"] = profile.get("torch_profiler_dir")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    root = args.run.resolve()
    if not root.is_dir():
        parser.error("实验目录不存在：" + str(root))
    print("只读已有日志：不导入Torch、不查询NPU、不写文件。RUN=" + str(root))
    details = root / "run_details.log"
    models = set()
    seen = set()
    if details.is_file():
        with details.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if "Effective serve command:" not in line:
                    continue
                try:
                    result = parse_command(line)
                    identity = json.dumps(result, sort_keys=True)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    emit("SERVE line=" + str(number), result)
                    models.add(result["model_path"])
                except (ValueError, IndexError, TypeError) as error:
                    print(f"命令解析失败 line={number}: {error}")
    if not seen:
        print("MISSING：未找到实际启动命令；不以当前test_graph2.sh替代历史配置。")
    for case in CASES:
        backend = "python" if case == "callback_legacy" else "cpp"
        path = root / case / ("inference_" + backend + ".log")
        print("\nCASE=" + case + "; LOG=" + str(path))
        if not path.is_file():
            print("MISSING：服务日志不存在")
            continue
        memory, decode, execution = deque(maxlen=14), deque(maxlen=14), deque(maxlen=10)
        with path.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if MEMORY.search(line):
                    memory.append((number, line.strip()))
                if DECODE.search(line):
                    decode.append((number, line.strip()))
                if EXECUTION.search(line):
                    execution.append((number, line.strip()))
        for title, records in (
            ("MEMORY/TOPOLOGY", memory),
            ("DECODE统计（可能覆盖整个服务生命周期）", decode),
            ("推测解码/抢占线索（最后10条；未必对应profile窗口）", execution),
        ):
            print(title)
            if not records:
                print("MISSING：没有匹配记录，不推算缺失数值")
            for number, line in records:
                print(f"{number}: {line}")
    for model in sorted(models):
        path = Path(model) / "config.json"
        print("\nMODEL_CONFIG=" + str(path) + "（读取当前文件，不保证文件从实验后未被修改）")
        if not path.is_file():
            print("MISSING：模型config.json当前不可访问")
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            for label, section in (("root", config), ("text_config", config.get("text_config", {}))):
                keys = (
                    "model_type",
                    "architectures",
                    "num_hidden_layers",
                    "n_routed_experts",
                    "num_experts",
                    "num_experts_per_tok",
                    "n_shared_experts",
                    "first_k_dense_replace",
                    "num_nextn_predict_layers",
                    "quantization_config",
                )
                selected = {key: section[key] for key in keys if key in section}
                if selected:
                    emit(label, selected)
        except (OSError, ValueError, TypeError) as error:
            print("读取失败：" + str(error))
    print("\n结束。只需返回控制台输出；无须重跑模型或传输日志文件。")


if __name__ == "__main__":
    main()
