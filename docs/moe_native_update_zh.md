# MoE 原生 C++ 权重更新（实验扩展）

新增 `host_update_backend="native"`，默认仍为 `"python"`。原回调版、MTE 默认版、MTE 组合版及原五组实验命令继续使用 Python 权重更新路径。

## 实现范围

本版面向单卡、C++ LRC、普通加载流、消费者处严格等待的 MTE 图重放。MTE 工作线程不再获取 GIL 调用 Python 回调，而是直接执行原生任务：读取已经完成 D2H 的专家 ID/分数/映射，计算命中与淘汰，提交 ACL H2D，等待加载流完成，提交映射更新，发布完成代数。设备 publish/wait 与延后预取等待的位置不变。

初始化、图构建、eager 预热和 prefill 仍使用原实现。本版不是整个推理引擎的 C++ 重写。原 Python 回调计时、专家 trace 和 debug 不支持新路径，配置时明确拒绝；不会输出假的 Python `host_callback` 时间。现有 profiling/五组统计脚本尚不用于原生版验收。

权重沿用原来已准备的 Storage 字节布局。原生配置的 MXFP4 scale 首次加载时就按设备目标的 shape/stride 存储，逻辑值不变，原 eager/prefill 的 Tensor 拷贝继续可用。注册描述符时，dtype 和物理排列相同的 pinned CPU 缓冲直接复用；其余 scale/offset/scale_bias 才额外整理为目标布局。派生 FP32 scale 也提前准备，保留原路径先转换到目标 scale dtype 再转 FP32 的舍入语义。新增内存由 `scale_buffer_bytes` 报告，复用字节数由 `scale_reused_bytes` 报告，不预设为 8 GiB，也不承诺所有量化格式都零额外副本。

目标必须是基础格式、每个专家槽位物理连续且无重叠；不支持的格式直接报错。描述符计入目标 storage offset 并检查地址范围。注册后禁止通过权重加载入口重新加载或重排已注册层；创建图任务时检查目标 tensor 身份、地址和布局，防止复用失效的描述符。更新权重需重启并重新注册；不支持外部代码绕过加载入口修改源缓冲。

异步搬运后必须同步完成再发布成功；超时或错误不提交新映射、不发布成功，原 MTE watchdog 终止服务。注册任务及其源、目标、请求缓冲一直保留到设备同步和工作线程停止之后。使用的运行时接口为 [aclrtMemcpyAsync](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0106.html) 和 [aclrtSynchronizeStreamWithTimeout](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0077.html)。

原生规划采用专家 ID 升序分配本次 miss，淘汰优先级、热度更新与预取候选选择复用原 C++ LRC 语义。它不复刻 CPython set 的物理遍历顺序，因此专家到槽位的配对可以不同；检查驻留专家集合、热度状态和槽位内容，而不能要求物理映射逐字节相同。槽位不足会报错，不继续执行缺少专家的计算。

这次参考同事实现中的提前整理 scale、复用 pinned 缓冲、冻结搬运描述符三个做法。原来的 Torch 异步 H2D 和本版 ACL H2D 都使用设备搬运能力；改变的是主机提交路径。本版保留每批同步完成后发布成功的协议，没有迁入对方的独立 DMA 队列、设备完成通知写入或提前下一层预取调度。这些改动影响依赖关系，不能直接删除现有同步来替代。

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

握手核显式使用 `CMAKE_BUILD_TYPE=Release`，CMake 工程也为未指定构建类型的调用提供 Release 默认值。构建开启 verbose；失败时，本步骤最后至多 60 行/32 KiB 日志自动进入 `summary.txt`，保留失败命令，完整日志仍在 `build.log`。

依次执行：构建两个 CPU 扩展；读取当前逻辑卡 0 的实际型号并构建匹配的 MTE 核；运行 CPU 回归；用小张量验证连续/转置/MXFP4 布局 scale 的字节拷贝和 pinned 复用；运行真实 NPU 图的 D2H→publish→原生搬运→严格 wait→映射 H2D 测试；原 MTE 组合配置与原生更新配置各启动一个服务，5 次预热、20 次正式请求。两组只有 `host_update_backend` 不同。不会启动 profiler/MSTX。

小图使用禁止调用的 Python 回调作为兜底哨兵，验证重复输入、miss、淘汰后的权重/scale/映射内容；Python 持有 GIL 忙等期间，必须观察到至少一次尚未完成的原生任务完成。无法形成该观测则报验证不充分，不假报通过。该测试在独立子进程运行，严格等待失败后终止子进程，不继续模型测试。

正式请求通过流式响应比较输出文本摘要、token 用量；未发现原生图任务实际完成标记时拒绝给出收益。任一步失败即停止，服务退出时只清理自己启动的进程组。`summary.txt` 汇总全部逐请求数据、均值表、原生准备内存、输出一致性与失败位置，详细构建和服务日志在同目录。

主时延为首个非空输出至最后非空输出的 decode 时延；同时保留 TTFT、完整 HTTP 请求时延和 TPOT（decode 时延除以实际输出 token 数减一）。沿用既有流式计时函数，SSE 块数不当作 token 数。所有请求保留 raw 和同规则 IQR 统计，主表列 raw 均值。这仍是单对服务的初次对照，不能代替多次交错实验和完整精度验收，也不验证加载流空泡是否消除。通过后再扩展原生阶段标记与正式统计。不要把本轮结果混入原五组历史表格。

## 验证状态

本地 Windows CPU 环境可以编译、加载两个 C++ 扩展，并验证原生线程在 Python 持有 GIL 时独立完成、错误不发布成功、重复请求读取新输入、预取上限和 LRC 状态一致性。随机序列测试比较 200 次更新的驻留集合与实际槽位字节。已有工作线程、生命周期、C++ LRC 和延后等待回归保留。

2026-11-20：重编译 C++ LRC 扩展，8 组本地测试共 150 项通过、2 项平台相关跳过。新增验证覆盖真实 MXFP4 布局函数、单缓冲加载、旧 eager/prefill 数值一致性、dtype 转换回退、带 storage offset 的视图、重载保护及地址溢出。改动文件的定向 Ruff 检查通过；manager 原有 12 条 lint 问题未扩大。完整 `format.sh ci` 因本地缺少 pre-commit 未执行成功。

本地没有 Ascend NPU，尚未完成真实 ACL 拷贝、模型输出和性能验证；上述绿区脚本负责这些检查。仅在检查通过后才能报告收益。
