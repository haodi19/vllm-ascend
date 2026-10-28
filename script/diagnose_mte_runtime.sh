#!/usr/bin/env bash
# 单次纯 MTE 原始拷贝复现；仅对子进程开启 CANN Runtime DEBUG。
set -eu
cd "$(dirname "$0")/.."
if [ "$#" -ne 1 ]; then
    echo "用法：bash script/diagnose_mte_runtime.sh NPU编号"
    exit 2
fi
echo "任务级诊断：纯 MTE、原始拷贝、无有界退出、无报告线程替代。"
echo "Runtime DEBUG 会扰动时延；本轮不作为性能结果。"
ASCEND_MODULE_LOG_LEVEL=RUNTIME=0 python -B script/run_moe_host_sync.py \
    --npu "$1" --backend mte --diagnose-native-wait \
    --requests 1 --warmup-requests 1 \
    --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
    --runtime-source /home/l00992607/vllm-ascend
