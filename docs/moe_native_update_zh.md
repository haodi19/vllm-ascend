# MoE 原生 C++ 权重更新（实验扩展）

新增 `host_update_backend="native"`，默认仍为 `"python"`。原回调版、MTE 默认版、MTE 组合版及原五组实验命令继续使用 Python 权重更新路径。

## 实现范围

本版面向单卡、C++ LRC、普通加载流、消费者处严格等待的 MTE 图重放。MTE 工作线程不再获取 GIL 调用 Python 回调，而是直接执行原生任务：读取已经完成 D2H 的专家 ID/分数/映射，计算命中与淘汰，提交 ACL H2D，等待加载流完成，提交映射更新，发布完成代数。设备 publish/wait 与延后预取等待的位置不变。

初始化、图构建、eager 预热和 prefill 仍使用原实现。本版不是整个推理引擎的 C++ 重写。原 Python 回调计时、专家 trace 和 debug 不支持新路径，配置时明确拒绝；不会输出假的 Python `host_callback` 时间。现有 profiling/五组统计脚本尚不用于原生版验收。

权重沿用原来已准备的 Storage 字节布局。scale/offset/scale_bias 在初始化时额外准备 pinned CPU 缓冲，逻辑值按目标 dtype、shape、stride 写入，运行时直接按字节搬运。目标必须是基础格式、每个专家槽位物理连续且无重叠；不支持的格式直接报错。派生 FP32 scale 也提前准备。原缓冲保留供 eager/prefill 使用；新增内存由启动日志 `scale_buffer_bytes` 报告，不预设为 8 GiB。取消额外副本不在本版内。

异步搬运后必须同步完成再发布成功；超时或错误不提交新映射、不发布成功，原 MTE watchdog 终止服务。注册任务及其源、目标、请求缓冲一直保留到设备同步和工作线程停止之后。使用的运行时接口为 [aclrtMemcpyAsync](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0106.html) 和 [aclrtSynchronizeStreamWithTimeout](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0077.html)。

原生规划采用专家 ID 升序分配本次 miss，淘汰优先级、热度更新与预取候选选择复用原 C++ LRC 语义。它不复刻 CPython set 的物理遍历顺序，因此专家到槽位的配对可以不同；检查驻留专家集合、热度状态和槽位内容，而不能要求物理映射逐字节相同。槽位不足会报错，不继续执行缺少专家的计算。

## 运行初次验证

在原来能够跑通实验的容器、已激活原环境、项目根目录执行：

```bash
git pull --ff-only origin dev/moeoffload-worker-poll
python -u -B script/run_native_moe_compare.py \
  --npu 1 --port 8156 \
  --test-script script/test_graph2.sh \
  --max-tokens 5 --warmup-requests 5 --requests 20 \
  --output-dir "/data/w00919836/moeoffload/bench_results/native_update_$(date +%Y%m%d_%H%M%S)"
```

脚本继承当前环境，只读取 `test_graph2.sh` 的 `vllm serve` 参数，不执行其清日志、sysctl 等前置命令，不修改原脚本。模型路径、MTP、图捕获尺寸、缓存容量等继续使用该文件的设置。输出目录必须是新目录。

依次执行：构建两个 CPU 扩展；读取当前逻辑卡 0 的实际型号并构建匹配的 MTE 核；运行 CPU 回归；用小张量验证连续/转置布局 scale 的字节拷贝；原 MTE 组合配置与原生更新配置各启动一个服务，5 次预热、20 次正式请求。两组只有 `host_update_backend` 不同。不会启动 profiler/MSTX。

正式请求比较输出文本、token 用量；未发现原生图任务实际完成标记时拒绝给出收益。任一步失败即停止，服务退出时只清理自己启动的进程组。`summary.txt` 汇总请求时延、输出一致性与失败位置，详细构建和服务日志在同目录。

这轮时延包含 prefill，是初次功能与粗略性能检查，不能代替正式 decode-only/TPOT、多次交错实验和完整精度验收。通过后再扩展原生阶段标记与正式统计。不要把本轮结果混入原五组历史表格。

## 验证状态

本地 Windows CPU 环境可以编译、加载两个 C++ 扩展，并验证原生线程在 Python 持有 GIL 时独立完成、错误不发布成功、重复请求读取新输入、预取上限和 LRC 状态一致性。随机序列测试比较 200 次更新的驻留集合与实际槽位字节。已有工作线程、生命周期、C++ LRC 和延后等待回归保留。

本地没有 Ascend NPU，尚未完成真实 ACL 拷贝、模型输出和性能验证；上述绿区脚本负责这些检查。仅在检查通过后才能报告收益。
