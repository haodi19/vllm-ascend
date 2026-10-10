# 已验证 C++ LRC：单测和实际推理对照

本轮只比较原版 Python LRC 与此前已验证的 C++ LRC。
不继续尝试 load_submit 的全原生迁移，也不根据测试结果改变算法。

| 部分 | Python 组 | cpp 组 |
| --- | --- | --- |
| LRC 状态、observe、choose_victims | Python | C++ 扩展，通过 Python 包装调用 |
| 缺失专家判断、槽位安排、映射更新 | 原 Python | 原 Python |
| load_submit、任务构造、量化 scale 刷新 | 原 Python | 原 Python |
| 权重复制 | 原 TorchCopy | 原 TorchCopy |
| 融合规划、ACL 实验、C++ load_submit 实验 | 关闭 | 关闭 |

`script/run_lrc_graph.py` 的默认后端现在为 `cpp`，表示仅使用 C++ LRC。
显式选择 `--backend python` 仍可使用原实现。项目全局配置默认后端仍为 Python，
直接执行原 `script/test_graph2.sh` 的默认行为没有改变。

## 绿区执行

在已同步本轮代码的容器内、项目根目录执行：

```bash
python -B script/run_verified_cpp_tests.py
```

关键数据同时写入 `bench_results/verified_cpp_<时间>_<标识>/summary.txt`，
终端开始时打印完整路径。也可以指定一个尚不存在的目录：

```bash
python -B script/run_verified_cpp_tests.py --npu 4 --output-dir bench_results/lrc_compare_01
cat bench_results/lrc_compare_01/summary.txt
```

构建、单测原始输出及推理服务日志分别保存在同目录的 `build.log`、`parity.log`、
`micro_*.log`、`inference_python.log`、`inference_cpp.log`，不混进关键数据汇总。
每次使用新目录，避免覆盖上次结果。profile 仍按下文路径保存。

脚本自动构建 LRC 扩展、检查 LRC 一致性，执行四轮单测，然后依次启动 Python 和
C++ 两个推理服务。每组完成后关闭自己启动的服务，再进行下一组。
不会关闭其他已有服务；运行前应保证脚本指定的 NPU、服务端口和计时端口可用。

默认沿用 `script/test_graph2.sh` 的 NPU 可见范围和服务端口，
另使用 `127.0.0.1:18155` 控制 worker 内的计时窗口。
单卡配置使用可见设备中的逻辑卡 0。可明确指定物理卡或其他空闲端口：

```bash
python -B script/run_verified_cpp_tests.py --npu 4 --port 8156 --timing-port 18156
```

只运行一部分：

```bash
python -B script/run_verified_cpp_tests.py --only micro
python -B script/run_verified_cpp_tests.py --only inference
```

默认每组推理先预热 3 个请求，再采集 10 个串行请求；请求内容固定为
`The future of AI is`，`temperature=0`，`max_tokens=5`。
可以用 `--warmup-requests`、`--requests`、`--max-tokens` 修改，两组保持一致。

## 与 test_graph2.sh 对齐的范围

如果 Python 推理的计时已经写入旧汇总，但在停止 profiler 时中断，可以只补跑 C++：

```bash
python -B script/run_verified_cpp_tests.py --npu 5 \
    --python-baseline bench_results/lrc_compare_01/summary.txt \
    --output-dir bench_results/lrc_compare_cpp_01
```

`--python-baseline` 自动选择仅推理模式，仅启动 C++ 服务，不重复单测和 Python 服务。
仍会构建扩展并运行 LRC 一致性检查。它校验旧汇总的设备编号、启动脚本哈希、预热次数、
请求次数、max_tokens、回调错误/丢弃数及分项表完整性，不匹配就停止。
需要使用原来的脚本、设备和请求参数；示例对应 NPU 5、预热 3 次、采样 10 次、max_tokens=5。

新汇总包含 C++ 原始分项数据及与旧 Python 基线的对比，且在 `/stop_profile` 前写出。
之后照常停止、导出 profile；若导出再失败，已完成的对比仍保留，但整轮标记为未完成。
两组来自不同运行会话，设备负载等环境差异仍需考虑。旧汇总中的均值精度为 0.001 微秒；
旧版未保存生成文本哈希，补跑时只能比较 token usage，不能验证两组生成文本是否一致。

## 推理配置保留

脚本读取该文件实际的 `vllm serve` 参数和环境设置，不依据其中可能过时的注释。
模型路径、量化、MTP、图模式、预取、缓存容量、前缀缓存、debug、seed 等配置保留。
默认不改 host、端口和设备范围。控制台打印每组实际的完整启动参数。

必要差异只有：LRC 后端、显式指定原 TorchCopy、诊断计时端口、两组独立的 profile
输出子目录，以及用户显式传入的端口/NPU覆盖。当前仅支持 TP=DP=PP=1 的单卡卸载。
脚本优先导入当前仓库，且不执行原 shell 中删除日志、修改宿主机 sysctl 和 CPU
频率的操作；需要的系统调优应事先设置，两组共用同一宿主机状态。

## profile 保存位置

保留原 `--profiler-config`，仅将输出目录分组，避免覆盖历史采集：

```text
<test_graph2.sh 的 torch_profiler_dir>/
└── verified_cpp_<时间>_<随机标识>/
    ├── python/
    │   └── <worker 的 Ascend profile 目录>/...
    └── cpp/
        └── <worker 的 Ascend profile 目录>/...
```

以当前脚本配置为例，根目录是 `/home/l00992607/vllm-ascend/moe_profile`。
在预热后调用 `/start_profile`，对采样请求同时进行 profiler 采集和回调分项计时，
然后调用 `/stop_profile`，等待导出 `trace_view.json` 和 `analyse.done` 后关闭服务。
控制台出现 `PROFILE SAVED` 才表示脚本确认了设备时间线导出。
默认导出等待上限 600 秒，可用 `--profile-timeout` 调整。
profile 留在绿区，可分别导入 MindStudio；无需传回文件。

等待时每约 15 秒输出当前阶段及耗时。`POST .../stop_profile` 表示等待服务的停止采集
接口返回；`profile export` 表示接口已经返回，正在等时间线和 `analyse.done` 文件。
两步分别最多等待 `--profile-timeout` 秒。CANN 日志中的 `data parsed` 仅代表相关
解析阶段结束，不保证整个导出和 HTTP 请求完成。`Running: 0` 表示当时没有在跑的请求，
不能据此断定 profiler 卡死。超时会在汇总中标记 FAILED，并保留已有数据。
时延表在 profile 导出前就写入汇总，避免导出失败导致已采集数据不可见。

如果请求或导出失败，脚本停止后续对比并清理自己启动的服务，已有 profile 目录保留。
导出超时的目录可能不完整，应检查该目录中的 profiler 日志。

## 时延字段

两类测试使用相同的分项定义，单位为微秒，每项打印 mean、p50、p95、p99、min、max。
最后打印 Python/C++ 的均值、差值和变化百分比，负百分比表示 C++ 组该项更短。

| 字段 | 含义 |
| --- | --- |
| total | `_update_weights` 回调内部总耗时 |
| observe | 更新 LRC 访问记录的累计耗时 |
| choose_victims | 选择淘汰专家的累计耗时 |
| synchronize | `_synchronize_h2d()` 的累计等待耗时 |
| outer | `total - observe - choose_victims - synchronize`，包含 load_submit |
| host_without_sync | `total - synchronize` |
| load_submit | `_load_expert_weights_into_slots()` 的累计耗时 |
| load_prepare | `_build_expert_h2d_tasks()` 构造传输任务的耗时 |
| copy_submit | `TorchCopy.copy_batch()` 调用耗时，不是 NPU memcpy 的纯硬件执行时间 |
| scale_refresh | `_refresh_expert_fp32_scale()` 的累计耗时 |
| load_other | `load_submit - load_prepare - copy_submit - scale_refresh` |
| fused_plan | 本轮未开启，通常为 0 |

这些字段存在包含关系，不能全部相加。差值先逐回调计算，再统计分位数。
total 从进入回调函数开始，不包含 `_launch_host_func` 入队到回调开始的排队时间，
也不能直接等同于 MindStudio 中两个 MEMCPY_ASYNC 时间块之间的空白。

单测按 Python → C++ → C++ → Python 顺序执行，固定相同输入参数，
从模型 config.json 获取专家数和 top-k，从 test_graph2.sh 获取卸载配置。
仍然使用合成权重、默认 1 行 token 输入、1000 次预热和 10000 次计时；
不是完整模型推理，也不模拟完整量化 scale 数据。

实际推理在原回调内计时，按 reactive（当前层加载）/prefetch（预取）和输入行数
分别统计，同时打印回调数量、涉及层号和每次加载专家数。MTP 场景可能出现多种输入行数，
不能将其直接理解为请求数量。两组工作量不一致时，不能仅根据均值断言迁移收益。
请求整体耗时另以毫秒输出，并比较 token usage 和生成文本的哈希是否一致。

实际推理的所有采样均包含计时器、debug（若原脚本开启）和 profiler 的开销，
不能当作无采集的生产性能，也不能直接与无 profiler 的合成单测绝对耗时相比。
现有的前缀缓存设置会使重复提示词走热缓存，这是沿用原配置的结果。

## 返回结果与本地验证边界

脚本不执行 Git 操作。请检查并脱敏后复制 `summary.txt` 内容返回；其中包含分组表、
COMPARISON、请求均值、一致性结果和阶段状态。无需传回文件或完整服务日志。

蓝区只能验证 CPU 逻辑、启动参数保留、计时公式、采样窗口及服务清理等行为。
模型启动、实际 NPU 回调、profile 导出和性能结论都必须在绿区执行后确认。
