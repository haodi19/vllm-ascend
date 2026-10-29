# MoE 预取 AIV 等待与加载流转置阻塞：底层联合定位材料

## 当前问题及已知边界

将 MoE 卸载的 CPU 回调交接替换为共享内存轮询后，真实模型在首次预取交接中出现
加载流同步超时。当前尚未确认根因，也没有证明硬件资源死锁。
已确认原始 scale.copy_ 中的 H2D 完成，后续设备 Transpose 未完成。
最近两次只读日志查询没有进一步区分调度阻塞和 kernel 内部执行停滞。

## 环境与复现

- Ascend950PR_9579；torch 2.10.0+cpu；torch_npu 2.10.0.post4.dev20260715。
- DeepSeek-V4-Flash，单卡，MTP=2，expert_prefetch_num=1。
- FULL_DECODE_ONLY 图模式，capture sizes 为 1、2、3。
- 基于原始阻塞提交 942868dd 的 dev/moeoffload-worker-poll 分支。
- 原始 Tensor 拷贝，未启用后续 CPU 重排和 Storage 适配。
- 原生 Host worker 在 SyncLaunchStream 上提交权重拷贝并等待其完成，随后写完成代数。
- 设备预取流通过 publish 算子写请求代数，再由严格 AIV wait 等待完成代数。
- 故障发生在第一个预热请求，尚未开始请求阶段的 profiler 采集。

复现入口（会启动模型，联合定位需要时再执行，不要求为准备材料重复执行）：

```bash
bash script/diagnose_mte_runtime.sh 7
```

该入口为纯 MTE 严格等待、原始拷贝、Runtime DEBUG，以及前八个加载批次的诊断同步屏障。
运行时间不能用作性能结果。模型与请求参数沿用 script/test_graph2.sh。
已有实验日志足够用于下面第一轮任务映射查询，应优先使用已有日志。

## 已完成的干预与结果

| 干预 | 结果 | 可以得出的结论 |
| --- | --- | --- |
| 真实模型、预取严格 AIV 等待、原始拷贝 | 第二次交接预取 layer=1、slot=2，scale 拷贝后超时 | 可复现基线 |
| 预取改报告线程等待，仍由同一类原生 worker 做原始拷贝；按需仍 AIV 等待 | 移除逐项诊断屏障后，有限的三次请求通过 | 预取等待方式影响阻塞，不是单纯换掉拷贝实现才恢复 |
| 预取有界 AIV 等待，后续报告回调严格守卫真实完成 | 多次耗尽预算后，worker 随后完成 | 支持 AIV 驻留与推进存在关系，未定位具体资源或依赖 |
| 真实加载流 Vector Core 上限从 56 改为 28 | 实际下发 blockDim=28，仍在相同位置超时 | 减半核需求不足以解除问题，不能排除所有资源竞争 |
| 独立小图及简单 fork/join 中测试同布局 D2D 与 AIV 等待 | 通过 | 简单拓扑未复现，不能代替真实模型结论 |

## 可以直接核对的 Runtime 证据

原始 56 核实验目录：bench_results/moe_host_sync_8e2b8992。
PID=66595，加载流 stream=27。

- task_sn=74900、task_id=710：512 KiB H2D；Runtime 记录完成并回收。
- task_sn=74901、task_id=711：Transpose_int8_int64_high_performance_10001。
- 转置下发 coreDim=56，schemMode=0，随后同步超时。

28 核实验目录：bench_results/mte_vector_limit_8260e005。
PID=68108，设备日志标识 device_id=7，加载流 stream=27。

- task_sn=74860、task_id=698：scale 的 H2D 下发。
- task_sn=74861、task_id=699：同名 Transpose。
- 实际 blockDim=28、groupDim=2、groupBlockDim=14、schemMode=0。
- worker tid=68277；raw ACL 同步等待 10000 ms，返回 507046。
- 超时快照中的 AIVEC 任务为 (stream=21, task=2)、(23,64)、(27,699)。
- 前两组任务在现有单算子下发记录中未找到映射，不能指定其为 host_sync_wait 或模型算子。
- 捕获时 Python 流对应 Runtime ID 47（按需）、56（预取）；不能直接套用到图执行内部的 21、23。

原始 scale 形状为 [64,4096,2]，uint8；CPU 源 stride=[8192,2,1]，pinned；
NPU 目标 stride=[2,128,1]，storage_offset=10485760。Transpose 是原始 copy_ 内部
处理布局差异的设备算子，不是新增 CPU 重排。

task_id 是可循环复用的位置；定位必须结合 PID、stream、task_sn、时间或图实例，
不能把早期同 task_id 的 DMA 完成记录误认为后续 AIVEC 已完成。
DavidSendTask 的 drvRet=0 只证明下发成功，不证明 kernel 已开始执行。

## 需要 Runtime／设备侧协助回答的三个具体问题

1. 对 PID=68108 的图执行，(21,2) 与 (23,64) 分别对应哪个图实例、哪个节点、哪个 kernel？
   请提供图内任务到捕获节点/函数的映射，以及有关 Event/Notify 的生产者和等待者。
   当前应用日志没有这条映射，不应仅凭相邻行推断。
2. 对 (27,699,task_sn=74861) 的 Transpose，超时时是尚未调度、部分 block 已启动，
   还是所有 block 已启动后停滞？若尚未调度，请确认具体等待的调度条件或资源；
   若已启动，请确认阻塞指令/同步点及未满足条件，而不只返回 KERNEL_AIVEC 类型。
3. 根据上述实际状态，能否画出闭合等待链？若涉及核资源，需要实际核占用/调度证据；
   若涉及图内依赖，需要具体节点与边；若涉及完成标记可见性，需要设备读取值与 Host
   发布值的对应证据。不能单凭“换成报告线程后通过”直接认定是哪一种机制。

这些证据也可以通过自行编写的诊断程序尝试获取，不以有对接人员为前提。
首先只读检查绿区安装的头文件、库导出符号和工具路径，再依据实际接口设计下一次采集。
不要在不知道可获得何种证据时重复全模型运行。

```bash
python -B script/inspect_mte_diagnostic_capabilities.py
```

脚本不导入 torch/torch_npu、不初始化设备、不启动模型或设备诊断工具，只读元数据、
头文件与文件路径；若有 nm，则限时查询 libascendcl.so 的导出符号。全部输出到控制台。
声明或符号存在不等于目标硬件支持；候选库未导出也不等于其他库不存在。
此步骤用于选择可实现的诊断手段，不是对阻塞根因的验证。

## 后续工作的判定边界

现阶段暂停核数枚举、重复等待预算实验和重复单算子日志筛选。
有底层映射/执行状态后，再针对具体依赖设计一次对照；没有这些证据时，结论保持
“预取等待方式影响真实模型中的转置推进，底层机制未定位”。
报告线程替代版只能视为有限验证通过的对照路径，不能据此声称纯轮询方案已修复，
也不能把后续布局绕行当成已针对根因修复。

本材料可经人工检查、脱敏后提供给内部对接人员；没有自动发送任何日志或消息。

## 与 MTE 改造的关系及自主图映射诊断

当前证据支持的是“新增预取 AIV 等待方式参与触发阻塞”，不是“共享内存本身错误”。
纯 MTE 路径将回调机制换成设备 publish、Host worker 轮询、设备 AIV wait，并为 worker
使用 SyncLaunchStream。预取报告等待对照保留了共享内存请求发布、原生 worker、
SyncLaunchStream 和原始拷贝，仅替换预取完成等待，有限请求恢复正常。
因此，这个对照比笼统地归因于整个 MTE 改造更具体；还不能单独认定 flag 可见性、
模型图调度、AIV 资源或内部同步中的哪一项是根因。现有证据也不足以断言
SyncLaunchStream 与其他机制的组合完全无关。

绿区安装了 CANN 9.1.0 与 msdebug。工具文件存在不证明它支持当前型号的在线调试；
异常回调符号存在也不保证普通流超时会触发所需设备信息。
前一能力检测脚本遗漏 aclmdl 前缀，导致漏查图模型调试接口，现已修正并加回归测试。

CANN 9.1 官方的
[aclmdlRIDebugPrint](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1786.html)
支持950，要求捕获结束后使用有效图实例，打印 device/stream/task 信息。
本次采用 Torch NPU 的 NPUGraph.enable_debug_mode 与 debug_dump 入口，避免猜测
内部图句柄或在捕获过程中调用打印。正式默认配置关闭，只有诊断配置显式开启。
绿区脚本先检查 Python 方法存在，再运行原始纯 MTE 模型，捕获完成后打印图信息。
该接口不能保证直接给出核上的运行 PC；第一目标是补齐图内任务身份。

```bash
python -B script/diagnose_mte_graph_mapping.py --npu 7
```

本轮不分核、不替换等待、不重排权重。完整日志保留本地，只返回 graph_diagnosis.txt。
该文件报告图打印开始/结束次数及 Runtime 候选记录。接口调用失败、工具输出不包含
映射或日志被轮转均需明确区分，不能将“开启图打印”本身写成已定位根因。

### 图打印首轮启动失败及修正

PID=71190 的首轮图打印诊断在捕获完成后报错：
`NPUGraph.debug_dump() missing 1 required positional argument: 'debug_path'`。
这是诊断代码使用了不匹配当前 Torch NPU 的无参数调用所致，并非原来的加载流阻塞。
本轮服务没有启动完成，不能纳入原阻塞的正反实验结果。

已改为显式传入路径，在实验目录 mte/graph_debug 下为每次捕获分配独立路径。
启动脚本先用 inspect.signature 绑定路径参数，再启动模型。参数签名检查不能代替
设备接口功能验证；本地只验证调用结构与默认路径回归。摘要列出图产物和小段文本预览，
无需传输原始文件。

### 图文件已生成，转为解析本轮捕获结构

`mte_graph_mapping_44f5e96f` 的图打印开始/结束各一次，生成约 3.6 MB 的 JSON 图文件。
本轮 PID=74275，依旧在第二次交接的 scale 原始拷贝后超时。
图实例63的 stream=21、task=2 明确为 host_sync_wait_2，Numblocks=1，
前后为 publish 和映射回传等节点。此前仅有 AIVEC 类型，现在这个图节点身份可直接读取。
但仍须与本轮快照核对，不能自动套用其他进程的 stream/task 编号。

新增只读解析脚本按本轮 Runtime 快照提取目标，打印图节点、同流前后节点及同图内
Event 同编号名称候选。Event 复用或缺少完整语义时，不直接认定唯一依赖。
图文件的规则 ts/dur 不是实测时间，不用于判断算子已经开始、完成或卡住的时刻。

```bash
python -B script/inspect_mte_graph_dump.py bench_results/mte_graph_mapping_44f5e96f
```

该步骤不重新启动模型，只读取已有图文件和日志。尚未获得其他待查图任务的身份，
仍不能声称已建立闭合等待链或已定位根因。

### 图内路由初始化与驻留等待的定点对照

对 PID=74275 同轮图文件的解析确认了第二个图内任务身份：

| 位置 | 任务 | Numblocks | Schem Mode |
| --- | --- | --- | --- |
| stream21/task2 | host_sync_wait_2 | 1 | 0 |
| stream23/task64 | MoeInitRoutingV3 | 56 | 1 |
| stream27/task711 | 动态加载流的 Transpose（Runtime 日志确认，不在捕获图中） | 56 | 0 |

图中预取分支从主计算流 EVENT_RECORD_8 分出，经 publish、wait、映射回传后，
通过 EVENT_RECORD_9 对应主计算流 task118 的 EVENT_WAIT_9。路由初始化位于
主计算流 task64，在这个汇合位置之前。图文件表示依赖结构，不是实测时间线。

[Runtime 调度模式说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1814.html)
中模式1为 batch 调度，需要所需的核全部空闲后启动。新增待验证假设是：
预取等待算子持续驻留，与请求56个 block 的路由初始化任务发生资源调度相互影响。
目前没有直接证据说明它如何进一步阻止其他流的 Transpose；不能据此宣布已证明死锁。
Numblocks 也不能直接当作已经独占的物理核数量。

之前把加载流 Transpose 从56减到28仍阻塞，改的是加载流，不是这次确认的路由任务。
本次诊断只在调用路由初始化时设置当前流 Vector Core 上限28，提交后恢复原上限。
纯 MTE 的 publish/wait、原始 Tensor 拷贝、加载流和严格 completion 等待均保留。
不修改路由算子的调度模式，不绕过数据完成条件。默认上限配置0表示不启用此诊断。

```bash
python -B script/diagnose_mte_graph_mapping.py --npu 7 --routing-cores 28
```

脚本合并运行和图文件检查，在 graph_diagnosis.txt 中输出实际路由和等待算子的
Numblocks、Schem Mode，以及请求结果或失败位置。判断规则为：

- 路由仍为56：干预没有生效，不能当作假设的反证。
- 路由变为28、严格等待仍在且请求通过：支持路由资源配置参与阻塞；仍需区分
  资源需求与 tiling 变化，不能直接断言底层调度算法或宣称这是最终修复。
- 路由变为28仍阻塞：该限制不足以消除阻塞；检查同轮快照身份是否变化，
  不重复宣称原 scale 拷贝的位置是新发现。

此诊断包含额外 ACL 调用、Host 提交队列排空和 Runtime DEBUG，不用于性能收益比较。

### 路由降为5个 block 仍超时，以及同步下发对照的边界

`mte_graph_mapping_6180ccf9`（PID=77262）请求路由上限5。图文件确认
MoeInitRoutingV3 为5个 block、模式1，host_sync_wait 仍为1个 block、模式0。
第二次交接仍停在预取 layer1 的 w13_weight_scale 原始拷贝后的等待，
加载流27/task699 返回507046；w13 和 w2 已分别完成，slot2 的 request=1、done=0。
这次核数干预有效但未解除阻塞，不能再用“56个 block 必须占满核”作为充分解释。
本轮摘要没有主计算流超时快照，不能沿用 PID=74275 的 task64 身份判断仍是路由在等待。

下一步只读本轮 Runtime 快照并关联本轮图，区分主计算流是否推进到其他算子。
解析脚本增加精简模式；运行诊断脚本也自动附带该映射，避免下次再次单独补采。
多个快照时刻的目标并集不代表同一瞬间，输出保留原始快照时间以便核对。

```bash
python -B script/inspect_mte_graph_dump.py bench_results/mte_graph_mapping_6180ccf9 --compact
```

`ASCEND_LAUNCH_BLOCKING=1` 用于强制算子同步执行、改善异步错误的调用栈定位，
并关闭 Torch NPU task queue，见
[官方环境变量说明](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/ASCEND_LAUNCH_BLOCKING.md)。
本分支 platform.py 明确拒绝该设置与 ACL 图同时启用，因此不绕过校验直接测试原图。
eager 下，expert_offload_manager.py 的预取分支在非 capturing 时直接调用
_prefetch_host_cb，不经过 MteHostDispatcher.launch；按需分支同样直接调用 CPU 函数。
因此 eager 或 eager 加同步下发可作为原始算子路径的辅助对照，但会移除待查的图内
驻留等待组合；即使通过，也不能证明纯 MTE 图路径已修复。

### 减核后主计算流的快照位置变化

PID=77262 的同一时刻快照显示，主计算流23当前任务为82，名称
HcPre_c12a9d608e5abdbd1728e849208025ff_1000，类型 KERNEL_AICORE；
预取流21仍为 task2 host_sync_wait，加载流27仍为 task699。
因此减小路由 block 数后，主计算流快照位置已经越过此前的路由 task64。
这支持“干预影响了主计算流推进”的判断，但没有消除加载等待，不能将路由宣布为唯一根因。
快照本身也不区分 HcPre 尚未获调度与已开始执行后的内部等待。

旧解析器沿用了仅选 KERNEL_AIVEC 的过滤条件，导致打印了原始 HcPre 快照却未匹配
它的图节点。这是诊断工具遗漏，已改为图解析覆盖快照中的全部任务类型，包括
AICORE、EVENT_WAIT、NOTIFY_WAIT；不在捕获图中的任务仍明确标记缺失。

本地 deepseek_v4.py 的调用链为 DeepseekV2DecoderLayer.hc_pre → npu_hc_pre_v2 →
run_hc_pre_fusion → aclnnHcPre。csrc/moe/hc_pre/op_kernel/hc_pre.cpp 声明
KERNEL_TYPE_MIX_AIC_1_2；950 分支的 tiling key 1000 使用 MKSplitCore 实现，
其中有 CrossCoreWaitFlag 和 SyncAll。也就是说，不能把快照的 AICORE 标签解释成
不需要 Vector 核。绿区二进制是否与此源码完全一致、实际执行是否卡在其中某个等待，
仍未验证，不能直接把源码中的等待指令当作已经捕获的阻塞位置。

下一步先复用同一份图检查 HcPre 的 block 数、调度模式和 Event 依赖，不重跑模型，
也不直接修改其跨核同步或调度模式。此次仍沿着“预取 AIV 驻留等待与真实模型算子
组合导致加载流无法推进”的方向排查，待查对象从路由扩展到快照中实际出现的混合算子。

### HcPre 配置确认及主计算流资源对照

同轮图中 task82 HcPre 的 Numblocks=28、Schem Mode=0，前一任务为 HcPost，
后一任务为 RmsNorm。stream22/task11 等待的 EVENT_WAIT_13 对应主流 task84 的
EVENT_RECORD_13，符合主流未完成 HcPre 后，下游事件也无法推进的现象。
预取完成汇合 EVENT_WAIT_9 位于主流 task118，晚于当前 task82。已导出的 Event
依赖没有显示主流提前在 task82 等待这次预取 completion；不能据此排除未导出的
资源依赖、算子内部同步或 Runtime 调度问题。

下一次实验保持路由上限5、原始 Tensor 拷贝和纯 MTE 严格等待，在图捕获期间给
主计算流设置 Cube 上限14、Vector 上限28，捕获结束后恢复原值。配置默认0关闭。
这是主计算流资源需求的对照，会影响其他主流算子，不能称为只修改 HcPre。
不修改 kernel 的 CrossCoreWaitFlag、SyncAll 或调度模式，不更换权重布局。
设置资源上限不等于独占或预留物理核，也不能保证所有缓存 tiling 都响应这个设置。
接口含义见 [aclrtSetStreamResLimit](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1947.html)。

```bash
python -B script/diagnose_mte_graph_mapping.py --npu 5 --routing-cores 5 --compute-cube-cores 14
```

脚本同时汇总 HcPre、路由与 host_sync_wait 的实际图配置、请求结果、失败快照映射。
若 HcPre 的实际 block 数未变，则此目标的资源干预没有得到验证，不能用失败反驳假设。
若 HcPre 确实减核且加载完成，支持主计算图资源需求参与阻塞，但不能唯一归因到
HcPre，也不作为推理正确性或性能验收。若减核后仍失败，先看任务位置是否再次变化；
不能仅凭失败继续任意尝试更小核数。当前改动只在蓝区通过模拟接口与回归检查，
实际限核效果和设备行为等待绿区验证。

### 主流限制生效，阻塞推进到 DSA 辅助分支

`mte_graph_mapping_7bc223a9`（PID=83599，物理卡7）中，HcPre 实际14个 block，
路由5个 block，host_sync_wait 仍为1个 block。完成交接数为3，未完成项变为
slot4、预取 layer2；此前一次输出为完成1项、slot2、layer1。本次仍在原始
w13_weight_scale 的同步等待返回507046，不能称为已修复。
两轮物理卡不同，也不能把推进一层换算为性能收益或单凭一次实验认定确定因果。

本轮快照和图结构为：

| 流/任务 | 算子 | block / 模式 |
| --- | --- | --- |
| 21/7 | host_sync_wait | 1 / 0 |
| 22/39 | MatMulV3 BF16 | 28 / 1 |
| 23/195 | KvCompressEpilog | 1 / 0 |
| 27/801 | scale 拷贝后未完成的 AIVEC 任务 | 图外加载流 |

主流193记录 Event37，辅助流22等待后执行 MatMul、记录 Event38；主流194动态量化、
195写压缩KV，再在196等待 Event38。这与 dsa_v1.py 的 decode 索引器分支一致：
辅助流做 weights_proj，主流做 query 量化和 KV scatter，随后汇合。
本地 KvCompressEpilog 实现没有轮询 Host generation 或等待这个 MatMul 输出的循环；
只有有界数据处理和本地流水线同步。不能描述成“它在内部等辅助 MatMul”。

上一轮仅设置主计算流的上限；dsv4_dsa_overlap_stream 创建的独立辅助流没有被限核，
其 MatMul 仍为28个 block。因此下面先验证控核接口，再控制辅助流，不改变图的并行拓扑。

复核官方 Torch NPU 2.10 源码发现，set_stream_limit 除设置 ACL 资源外还启用框架的
core control；原始 ctypes 调用不会设置这一框架状态。异步任务队列的资源处理受该状态
控制，不能假定仅设置某个 ACL 流就覆盖 Torch 所有下发路径。
参考 [NPUFunctions.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUFunctions.cpp)、
[NPUStream.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUStream.cpp)。
绿区为开发构建，最终以实际图配置为准。新接口恢复各流原资源数值，但框架 core control
会保留在本次诊断进程内；每组使用独立服务进程，不能混入性能比较。

合并脚本先跑 native_main：使用 Torch get/set_stream_limit，主流14/28、路由5，
不设置辅助流上限。仅当这组重现已知 scale 等待且图配置有效时，再跑 native_main_aux：
在相同接口和主流设置下，增加辅助流14/28。MTE严格等待、原始copy和DSA并行均保留。
若第一组已通过，停止第二组，先分析框架控核接口的影响；启动失败或配置不生效也停止。

```bash
python -B script/diagnose_mte_resource_scope.py --npu 7
```

只返回输出目录的 summary.txt。每组图、完整日志和配置留在本机。图摘要包含流编号，
用于确认辅助 MatMul 的实际 block 数；不再只看命令参数。无诊断参数时沿用原逻辑，
旧 ctypes 主流实验也保留可重现。

### 原生控核主流对照仍阻塞，辅助流对照启动失败

PID85186 的 native_main 对照中，Torch 原生控核已打印主流14/28，图内 HcPre14、
路由5、辅助 MatMul28。仍在 slot4 等待 completion 时返回507046；图映射仍为
预取 host_sync_wait、辅助 MatMul、主流 KvCompressEpilog。换用原生控核接口没有
解除这次阻塞，不能据此排除其他资源调度问题。

native_main_aux 在服务启动阶段退出，未取得图导出或实际辅助流核数记录。
该组不能用于判断辅助流限核的效果，也不能直接归入原来的 scale 阻塞。
上一版精简摘要过滤了异常调用栈，现补充 EngineCore 调用栈、末尾错误及日志尾部；
优先读取已有日志，明确启动失败原因后再决定是否重跑。

```bash
python -B script/diagnose_mte_resource_scope.py --inspect-latest
```

此命令只在控制台输出最近一次实验的辅助组日志片段，不导入 Torch、不初始化设备、
不运行模型、不写文件。也可用 `--inspect 实验目录` 明确指定目标。调用栈片段本身
不自动视为根因证明；若进程没有留下异常，还需进一步区分外部终止与未捕获的退出。

补读该组完整上下文后，EngineCore PID86760 最后记录为 checkpoint 加载2/46，
尚无捕获或控核生效日志。checkpoint 大小148.66 GiB，读取时显示可用RAM192.60 GiB；
这不是峰值内存需求或容器限额，不能据此宣布内存足够，也不能认定发生OOM。
可见内核段错误发生在11:16，而本组退出在17:12，不能用该历史记录解释本轮退出。
辅助流上限设置位于图捕获内部，目前没有证据将这次加载阶段的失败归因到该设置。

主流对照已有有效结果，可以只补跑未完成的辅助流组，保留新目录和独立摘要：

```bash
python -B script/diagnose_mte_resource_scope.py --npu 7 --case native_main_aux
```

脚本不自动重试，不重跑主流对照，不修改加载策略。摘要增加启动前主机内存快照；
若再遇到无效启动，自动附带退出上下文与内核候选记录。单独补跑不能保证与前一轮
资源负载完全相同；需先检查图内辅助MatMul实际配置和请求结果，不用于性能比较。
