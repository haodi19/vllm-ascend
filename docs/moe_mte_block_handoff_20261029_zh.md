# MoE 卸载 MTE 阻塞排查交接（2026-10-29）

## 1. 新对话先读这里

本文用于在没有上一轮聊天上下文的情况下继续工作。记录截止代码提交
`9083f155b3872521df31201cccbf86de3da72236`，以及用户返回的第二组启动失败日志。
本文是当前状态快照；其他文档包含历史阶段的判断，应按实验时间区分，不能把旧的
“尚未映射任务”或“待执行”当作最新状态。

**当前没有定位出底层根因，也没有修复纯 MTE 的真实模型阻塞。**
已确认：真实模型的原始 scale 拷贝涉及 H2D 和设备布局转换，卡点落在后续
设备 Transpose 未完成；预取路径上的 AIV 严格等待与真实模型执行的组合参与触发。
尚未确认：Transpose 是未获得调度资源，还是已开始执行后内部停滞；没有建立
带实际资源/依赖证据的闭合死锁链。

最近一次有效主流控核对照仍然阻塞。后续辅助流控核组在 checkpoint 加载阶段
失去进展并退出，未形成有效对照。下一步已经准备好：**只补跑辅助流组**。
用户尚未返回这次单组补跑的结果。不要默认它已经跑过或已通过。

## 2. 用户目标、工作方式和必须遵守的边界

- 目标是定位 MTE 改造为何引入阻塞，以及最终能否减少 CPU–NPU 交接开销。
  当前优先定位，不要求立刻优化。约400 μs空泡、降低50%以上是早期目标，
  不是当前方案已经实现的效果。
- 用户不接受将 CPU 重排、额外 Storage 适配等绕行直接称为根因修复。
  要与原回调版比较，不能把自己后来引入的开销当作原始基线。
- 原回调版本必须保留；实验新增功能默认关闭，显式选项开启。
- 不得以有界等待结束代替权重真正完成，再让模型消费旧权重。
- 用户无法传输绿区文件，只能复制控制台或一个精简 `summary.txt` 的文本。
  完整日志、图文件和 profile 保留绿区，脚本应自动提取必要内容。
- 能合并的实验和日志提取应合并；不要为每段日志让用户重新跑一次模型。
- 每轮先给短结论，区分事实、假设和未得到有效结果的实验。
  反复确认“超时对应 scale.copy_”不算新的底层定位进展。
- 不要假定有外部对接人员；应优先利用绿区现有源码、接口和工具自主排查。
- 蓝区负责修改、测试、签名提交和推送。用户已长期授权修改后自行提交推送。
  绿区由用户拉取和执行，不在绿区创建正式提交或推送。
- 所有新文档用中文，遵守 AGENTS.md 和 BLUE_GREEN_DEVELOPMENT_GUIDE.md。
- 用户工作区中的未跟踪文件 `a.txt` 不属于本任务，不修改、不提交。

## 3. 仓库、分支和运行环境

| 项目 | 当前值 |
| --- | --- |
| 蓝区正在开发的工作树 | `D:\华为\nano2\vllm-ascend-worker-poll` |
| 活动分支 | `dev/moeoffload-worker-poll` |
| 用户新对话默认 cwd 可能是 | `D:\华为\nano2\vllm-ascend`，这是另一工作树，不要误改 |
| 原工作树分支 | `dev/moeoffload`，本次检查时提交为 `dc0f904c` |
| 个人远端 origin | `https://github.com/haodi19/vllm-ascend.git` |
| 上游 upstream | `https://github.com/LookAround0301/vllm-ascend.git`，不要推送这里 |
| 绿区当前诊断仓库 | `/home/l00992607/vllm-ascend-worker-poll` |
| 绿区已有运行产物/脚本来源 | `/home/l00992607/vllm-ascend` |
| 绿区 vLLM 核心代码 | `/home/l00992607/vllm` |
| 原始阻塞基线 | `942868dd8cec9019dee04bee038bfcd678b6d66a` |
| 当前代码提交 | `9083f155b3872521df31201cccbf86de3da72236`，已推送 |
| 设备 | `Ascend950PR_9579` |
| Python / Torch | Python 3.11.10；`torch=2.10.0+cpu` |
| Torch NPU | `2.10.0.post4.dev20260715`，不能直接等同于公开版本所有实现细节 |
| CANN | `/usr/local/Ascend/ascend-toolkit/latest` 指向 `/usr/local/Ascend/cann-9.1.0` |
| 模型 | `/mnt/data/models/DeepSeek-V4-Flash` |

典型模型配置：单卡，MTP=2，`expert_prefetch_num=1`，`FULL_DECODE_ONLY`，
图捕获尺寸1、2、3。测试脚本来源是 `script/test_graph2.sh`，不是猜测的
`scripts/test_graph2.sh`。诊断包装器明确使用绿区已有仓库的该脚本和运行产物。

请求沿用短提示词 `The future of AI is`、`max_tokens=5`、temperature=0，
服务模型别名 `glm-5`。实际参数以运行器打印的完整 serve 命令为准。
最近实验使用物理 NPU7；脚本映射到可见逻辑设备0。历史也用过其他物理卡，
不能把不同卡、不同负载的差异当成严格性能因果证据。

蓝区没有目标 NPU/CANN 运行环境，只能做 CPU 单测、静态检查和源码分析。
不能把蓝区单测通过写成 NPU 验证通过。

## 4. 当前 MTE 实现与正常交接流程

本工作树从旧阻塞基线发展，保留原权重搬运路径，没有移入另一分支的
CPU 重排、目标池布局调整等绕行方案。注意：原代码自身就可能对主权重用
Storage 拷贝，对量化参数用 Tensor 拷贝；“未加入 Storage 适配”不等于
整个旧版本从未使用 Storage。

图内交接主要流程：

1. 外层 `ExpertOffloadManager.update_weights()` 准备路由回传等操作。
   按需交接使用计算流，预取交接使用预取流。
2. `MteHostDispatcher.launch(stream, callback, args)` 为这次捕获的交接注册槽位，
   保存 Python 函数与参数，向对应流提交 `host_sync_publish` 和严格 `host_sync_wait`。
   捕获的是设备任务；重放时 generation 改变，触发同一槽位关联的 Host 工作。
3. AIV 发布算子递增代数，通过 MTE `DataCopyPad` 写注册的 Host 请求内存。
4. 原生 C++ worker 轮询发现请求，领取任务、获取 GIL，调用 Python 包装函数，
   进入 `_update_weights_guarded()` / `_update_weights()`。
5. Python 执行缓存处理并在加载流提交原始权重和 scale 拷贝。
   MTE 后端加载流是 `SyncLaunchStream`，并保留加载流同步等待。
6. Python 工作函数真正返回后，C++ worker 才写匹配的 done generation。
7. AIV 等待算子观察到完成后退出，交接流继续映射写回和后续计算。

普通 eager 路径仍有直接调用逻辑，不能把整个程序每次调用都描述为上述图内路径。
MTE 路径中的 `host_callback` 打点名称也不表示使用了 ACL 报告线程。

关键数据与线程：

- **交接槽位**是一次捕获的 Host 任务登记位置，不是专家权重的物理缓存槽位。
  原子请求/完成控制区在注册的 CPU 内存中，通过设备别名供 MTE 访问。
- 一个控制槽位的 request 与 completion 分开512字节，整个槽位1024字节。
  权重本体不存放在这些控制槽位中，仍按原有 H2D 路径传输。
- NPU `states` 是设备上的代数及诊断状态记录；等待结果包含预期代数、
  退出状态、观察到的完成代数、轮询次数。它与 Host 控制内存不是同一个数组。
- 请求代数已更新、done仍为0、worker正在同步，说明 CPU 已接到任务，
  尚未完成。不能直接归因为 CPU 没轮询到请求或完成标志不可见。
- 默认诊断固定 `scan=all / probe=locked / idle=yield`，避免混入 worker 优化。
  已实现的有效槽位扫描、原子预检查、spin开关暂不作为当前阻塞的解法。
- `SyncLaunchStream` 的引入意图是避免 worker 提交依赖普通流的 Host 队列排空。
  它不是“拷贝本身变为同步 API”；异步拷贝后仍显式等待加载完成。
  该变化已经生效但未消除真实模型阻塞，也不能断言与其他机制组合完全无关。

## 5. 最重要的已确认阻塞证据

早期真实模型的 batch2、layer1 预取任务中：

- `w13`：8 MiB，已记录完成。
- `w2`：4 MiB，已记录完成。
- `w13_weight_scale`：512 KiB，`copy_()` 返回，随后加载流同步超时。
- 后续 scale/refresh 尚未执行，worker停在加载流等待。

因此，HTTP失败是后果，不是根因；不是已经完成的 w13 主权重又“没搬完”。
`copy_()` 返回也不代表里面所有设备任务完成。

真实 scale 元数据的一次记录：uint8，shape `[64,4096,2]`；
CPU pinned源 stride `[8192,2,1]`；NPU目标 stride `[2,128,1]`，
storage_offset为10485760。源/目标布局不同，原始 Tensor 拷贝会涉及设备转换。

最明确的 Runtime 记录来自 `bench_results/moe_host_sync_8e2b8992`：

| 身份 | 任务 | 已有证据 |
| --- | --- | --- |
| PID66595，stream27，task710，task_sn74900 | 512 KiB H2D | Runtime记录完成并回收 |
| 同进程/流，task711，task_sn74901 | `Transpose_int8_int64_high_performance_10001` | 下发coreDim56、schemMode0，后续同步超时 |

这与拆分 H2D、设备布局转换的实验结果一致。这里的 Transpose 是原始
`w13_weight_scale.copy_()` 的设备内部操作，不是后来新增的 CPU 重排。

## 6. 已完成实验及其证据边界

### 6.1 拆分等待与拷贝

`--diagnose-native-wait` 将前八批加载的阶段拆为：

```text
取加载流句柄
原操作 BEGIN → SUBMITTED
直接 aclrtSynchronizeStreamWithTimeout → ACL_WAIT_RETURN
原 Torch stream.synchronize → COMPLETE
```

直接 ACL 等待也会超时，说明不只是 Python 正在获取句柄或 Torch 同步前排空
Host 队列的现象。进一步拆分及 Runtime 记录把问题落到 H2D 后的设备转换。
诊断有逐项同步、日志、Runtime DEBUG，会改变时序，不作为性能结果。

### 6.2 小图与并行能力隔离

独立图、简单 compute→prefetch→compute 分支汇合、AIV等待期间worker提交D2D等
实验通过。另一次故意让CPU函数等待约100ms的测试中，其他流计算可提前约95ms完成。
因此不能说“AIV wait只要存在，就阻止所有其他流计算”。

这些小实验没有真实模型算子与资源需求，不能代替真实模型安全性验证。
有界wait的小实验，轮询次数不等于微秒数。

### 6.3 替换预取等待方式的对照：重要触发条件证据

预取改用报告线程等待，但保留 MTE 发布、原生 worker、SyncLaunchStream、
原始权重拷贝；按需仍用AIV等待。移除逐项诊断屏障后，有限三次请求通过。
这不是整体退回原回调版，而是针对预取完成等待的对照。

另一个实验先让预取AIV有界等待退出，再用报告回调严格守卫实际完成，
多次看到AIV耗尽预算后worker随后完成。报告回调没有提前放行模型消费者。

这些结果支持：**预取AIV等待方式影响真实模型中加载流的推进。**
但不能唯一归因为共享内存一致性、核资源耗尽、调度策略或某个内部事件。
对照路径通过不等于纯MTE已修复，不应作为最终方案直接替换结论。

### 6.4 加载流限核

`mte_vector_limit_8260e005`，PID68108，物理卡7：
Transpose实际blockDim从56降为28，groupDim2、groupBlockDim14、schemMode0，
仍超时。worker tid68277，ACL等待参数10000ms，返回507046。

结论只是“此次减半不足以解除”，不是“所有资源竞争都被排除”。

### 6.5 捕获图映射

首次图诊断误用了无路径参数的 `NPUGraph.debug_dump()`，导致启动失败；
提交 `33d640b3` 修正为明确路径并做签名检查。这是诊断代码错误，不是原阻塞。

后续 `mte_graph_mapping_44f5e96f`，PID74275，图实例63，确认：
预取21/2为 `host_sync_wait`，另一个主流任务为 `MoeInitRouting`。
此前只能看到 KERNEL_AIVEC 类型，现在可以确定节点身份。
任务映射已经完成这一层工作，不要继续把它当尚未完成或新进展。

### 6.6 路由限核后仍阻塞

PID77262，路由实际5 blocks后，主流快照推进到23/82 `HcPre`，
预取21/2仍为wait，加载27/699仍为未完成AIVEC任务。
本轮图映射中 HcPre 为28 blocks。HcPre源码存在混合核与内部同步逻辑，
不能仅根据 KERNEL_AICORE 标签认定它完全不占用Vector资源。

主流在HcPre后还有RmsNorm等操作，汇合本次预取的Event更靠后。
图结构没有显示主流在这个位置已经显式等待本次预取completion；
不能据此排除未显式导出的资源或算子内部依赖。

### 6.7 主计算流限核后仍阻塞

`mte_graph_mapping_7bc223a9`，PID83599，物理卡7：
主流Cube14/Vector28，路由5；HcPre实际14 blocks，配置确实生效。
记录到完成交接3次，待完成变为slot4/layer2预取，仍在scale同步超时。
此前有一次完成1次、slot2/layer1。部分轮次物理卡不同，不能把推进位置变化
换算成收益或仅据此断言单变量因果。

超时位置映射：

| 流/任务 | 图内任务 | blocks / 模式 |
| --- | --- | --- |
| 21/7 | host_sync_wait | 1 / 0 |
| 22/39 | DSA辅助流BF16 MatMulV3 | 28 / 1 |
| 23/195 | KvCompressEpilog | 1 / 0 |
| 27/801 | 图外加载流的未完成AIVEC | 不在该捕获图中 |

源码与图中的分支关系：主流193记录Event37；辅助流38等待、39计算MatMul、
40记录Event38。主流194动态量化、195执行KvCompressEpilog，再在196等待Event38。
对应 `dsa_v1.py` decode中的 weights_proj 辅流和query量化/KV scatter主流。

已读 `KvCompressEpilog` 实现：有界数据循环与本地流水同步，未发现等待该MatMul
结果的跨kernel轮询。不能说“KvCompressEpilog内部就在等MatMul结束”。

### 6.8 原生控核主流对照：最新有效实验

需要区分原始ctypes ACL控核与Torch `get/set_stream_limit`：后者还启用框架
core control，影响队列侧资源处理。公开2.10源码支持这种区别，但实际绿区是dev构建，
以本地接口及图中实际核数为准。恢复流的数值不等于撤销进程内框架开关，
所以两组使用独立服务进程。

最新实验根目录：`bench_results/mte_resource_scope_98527a74`。
第一组 `native_main`，PID85186，主流14/28、路由5、辅助流不改：

- `RESOURCE_API=torch`，主流原值28/56，设置14/28。
- 图内HcPre14、路由5、辅助BF16 MatMul28；wait1。
- pending_slot4，request_generation1、done_generation0、error_generation0、status1。
- 加载流同步507046，首个预热请求HTTP500。
- 快照及映射仍为21/7 wait、22/39 MatMul、23/195 KvCompressEpilog；加载27/825 AIVEC。

**换用Torch原生控核接口未解除本次阻塞。** 不能再把“只是用了原始ACL接口”
当作足以解释当前问题的根因；也不能因此排除全部资源调度关系。

### 6.9 辅助流对照：启动失败，不是有效阻塞实验

同目录第二组 `native_main_aux`：请求主流14/28、路由5、辅助流14/28。
API PID86182，EngineCore PID86760。426.5秒时服务提前退出，未达到1800秒启动上限。
没有图dump开始/结束、控核生效打印、实际辅助流核数或同类scale超时证据。

补读日志后发现：

- 17:06:50开始加载模型，加载流配置为SyncLaunchStream。
- 17:11:02记录checkpoint148.66 GiB、Available RAM192.60 GiB，开启文件page-cache预取。
- 17:11:17开始32线程的并行权重加载。
- 最后进度为2/46（4%）；17:12:32 API报告引擎初始化失败。
- 没有EngineCore Python异常栈。
- dmesg只有11:16另一进程的libpython段错误，与17:12本组退出不匹配。

没有足够证据确定是OOM、外部终止还是原生崩溃。Available RAM与checkpoint大小
不能证明峰值内存足够，也不能证明OOM；容器限额与宿主可用量不是同一指标。
不能用上午那条segfault证明下午的退出。

辅助流控核调用位于图捕获内部。目前没有执行到它的证据；不要把该加载阶段退出
写成“辅助流限核仍无效”或“仍是原来scale卡住”。

## 7. 现在尚缺的底层证据

目前只有两条明确的等待：

```text
预取AIV wait 等待 worker 发布 done
worker 等待加载流（含Transpose）完成
```

要证明死锁，还缺“Transpose完成依赖这个wait先退出”的具体证据。
可能是资源调度，也可能是算子/图内部依赖，不能把候选补成事实。

下一层真正需要区分的是：

1. Transpose尚未开始、部分block开始，还是全部开始后停滞？
2. 若未调度，实际等待什么资源/调度条件，哪些任务占用？
3. 若已执行，卡在哪个内部同步/指令，缺少哪个生产者或条件？

日志中的 `drvRet=0` 只证明下发成功；超时快照指向任务不证明该任务已经执行。
图dump中的ts/dur是捕获结构信息，不能当成实际profile时刻。
不同进程的stream/task编号不可互套；task_id可能循环复用，需结合PID、时间、
task_sn或图实例。相邻线程日志也不能自动连成任务映射。

## 8. 已准备好的下一步与结果判定

用户最后收到的命令如下，**尚未返回结果**：

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_resource_scope.py --npu 7 --case native_main_aux
```

它只补跑辅助流组，新建 `bench_results/mte_resource_scope_<随机编号>/summary.txt`，
不重跑已完成的native_main，不改变模型加载策略、原始拷贝或严格等待。
摘要会附启动前主机内存、实际图配置、结果；若启动失败，自动附退出前上下文与
内核候选记录。让用户只复制summary，不传文件。

判定规则：

- **仍在checkpoint加载阶段退出**：这是启动问题，先查退出原因，不能算资源对照。
- **运行到图，但辅助MatMul实际配置未改变**：这次干预未验证生效，不能用失败反驳假设。
- **实际核数改变，有限请求通过**：支持辅助流资源需求参与阻塞，仍不是唯一根因或性能验收。
- **实际核数改变，仍在scale超时**：核对本轮新快照，不继续任意枚举更小核数。
  若只是重复旧现象，应明确该假设未得到进一步支持，转向执行状态直接证据。

默认不加 `--case` 会依次跑native_main和native_main_aux；第一组通过或未形成
有效同类阻塞时停止。不要在当前补跑阶段误用默认模式浪费一次模型加载。

只读已有失败日志：

```bash
python -B script/diagnose_mte_resource_scope.py --inspect bench_results/mte_resource_scope_98527a74
```

该模式不导入Torch、不初始化设备、不启动模型、不写文件，打印日志片段和只读
dmesg候选。无权限或无匹配记录不等于系统从未发生相关错误。

## 9. 关键代码和脚本导航

路径相对当前worker-poll工作树，行号以后可能变化，应搜索函数名。

| 文件 | 阅读重点 |
| --- | --- |
| `vllm_ascend/expert_offload/expert_offload_manager.py` | 外层update_weights、内层_update_weights、加载提交与_synchronize_h2d；MTE加载流创建 |
| `vllm_ascend/expert_offload/host_sync.py` | MteHostDispatcher.launch、diagnose_load、报告等待对照、watchdog、原生/ACL限核辅助函数 |
| `csrc/host_sync/host_sync.cpp` | worker领取任务、GIL、Python调用、完成发布、扫描与idle策略 |
| `csrc/host_sync/host_memory.h`、`protocol.h` | Host注册内存、槽位布局、代数和状态协议 |
| `csrc/host_sync/vllm_ascend_host_sync.py` | 独立扩展Python包装与资源生命周期 |
| `csrc/host_sync_device/handshake.cpp` | publish/wait/strict wait；consume主要用于独立测试 |
| `vllm_ascend/compilation/acl_graph.py` | 图捕获内控核，debug_dump与路径；主/辅助流上限只在诊断开启 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | 路由Vector上限的诊断包装 |
| `vllm_ascend/attention/dsa_v1.py` | dsv4_dsa_overlap_stream；decode weights_proj与KV scatter的分支汇合 |
| `csrc/attention/kv_compress_epilog/op_kernel/kv_compress_epilog.h` | KV算子内部本地流水同步，没有发现跨kernel完成轮询 |
| `script/diagnose_mte_resource_scope.py` | 当前首选合并入口，支持单组补跑和只读恢复摘要 |
| `script/diagnose_mte_graph_mapping.py` | 每组启动运行器、图导出与超时映射；可独立指定控核参数 |
| `script/run_moe_host_sync.py` | 构建/运行产物、配置注入、服务/请求/profile/清理；不要从摘要丢掉真实错误 |
| `script/inspect_mte_graph_dump.py` | 只读本轮图和Runtime快照，匹配节点及Event候选 |
| `script/inspect_mte_task_serial.py`、`inspect_mte_runtime_task.py` | 已有Runtime日志的task_sn与单算子任务关联 |
| `script/inspect_mte_outstanding.py` | 超时其他任务的下发候选；未找到映射不能猜身份 |
| `script/probe_scale_wait_dependency.py` | 独立D2D与fork/join对照，已经测过；勿重复当模型验证 |
| `script/inspect_mte_diagnostic_capabilities.py` | 不初始化设备的CANN头文件、工具与导出符号检查 |

绿区已发现msdebug、msdebug-mi、msprof、msopprof、msobjdump等工具。
已发现异常信息查询/回调API的声明及部分库导出，包括
`aclrtSetExceptionInfoCallback`、`aclrtGetFuncHandleFromExceptionInfo`、
`aclrtGetArgsFromExceptionInfo` 等。仅发现接口不证明硬件支持所需调试，
也不证明普通stream超时会生成期望的设备执行状态。尚无设备PC/核占用的实证采集。

## 10. 关键提交索引

以下提交用于理解变化，不要求把工作树reset到这些位置。若需回退实验，用独立
worktree保留现有代码；不要破坏用户可随时切回的状态。

| 提交 | 内容 |
| --- | --- |
| `942868dd` | 旧阻塞基线，后续worker-poll工作基于它 |
| `7be7f811` | worker计时与yield/spin独立比较 |
| `10319e93` | 有效槽位集合扫描 |
| `61f886fe` | mutex前原子预检查 |
| `5a7d1740` | 可选轮询策略接入MoE；阻塞诊断保持默认all/locked/yield |
| `f7e536a3`、`c0b40078` | 拆ACL/Torch同步、拆scale H2D与设备布局拷贝 |
| `54dac14e`、`6e660f83` | 预取报告等待对照及取消诊断屏障后的有限通过记录 |
| `1a26e4a8`、`5c771543` | 有界AIV退出后由报告回调守卫真实完成；记录退出状态 |
| `dd4ed113` | 加载流核数干预 |
| `6204aeec`、`33d640b3` | 捕获图导出及修正debug_path参数 |
| `be0b891e` | 本轮超时任务到捕获图映射 |
| `de785a85` | 路由核数干预 |
| `01633328`、`7974815d` | 自动汇总本轮快照、包括混合核任务 |
| `baf628ae` | 捕获期间主计算流资源限制 |
| `8a0fa3da` | Torch原生主流/辅助流控核与两组自动对照 |
| `c81f7ef9` | 启动失败栈与只读摘要恢复 |
| `208ab1ec` | 没有EngineCore调用栈时保留API报错前日志、内核候选 |
| `9083f155` | 仅补跑指定组；内存快照、失败上下文自动汇总 |

## 11. 已有文档与哪些历史内容暂不继续

- [旧版阻塞定位](moe_mte_block_diagnosis_zh.md)：逐项同步、scale分解及早期候选解释。
- [Runtime联合定位材料](moe_mte_runtime_investigation_zh.md)：完整实验追加记录与图映射。
  前部有早期未映射/暂停实验措辞，最新状态以本文和后面的追加段落为准。
- [MTE接入流程](moe_host_sync_integration_zh.md)：CPU/NPU线程、内存和调用链。
- [worker轮询改造](worker_poll_optimization_zh.md)：yield/spin、扫描集合、原子预检查、绑核等。
- `host_sync_stage1_zh.md` 至 `host_sync_stage5_zh.md`：早期独立协议/图/真实拷贝验证，
  不应反复回到这些阶段宣称又验证了模型阻塞原因。

更早已做LRC C++迁移、MSTX打点和gap_debug开销排查；另一个主分支尝试过
CPU重排、目标池布局、Storage适配及性能对照。当前不是继续这些优化。
曾经的预取→按需host_callback间隔约400 μs、回调版几十μs，也是独立的性能问题，
不能把当前同步超时的原因未明，改写成已经解释了整个400 μs差值。

## 12. 蓝区检查和交付习惯

截至9083f155，最近改动对应的10项CPU测试和Ruff检查通过：

```powershell
Set-Location 'D:\华为\nano2\vllm-ascend-worker-poll'
python -B tests/ut/expert_offload/test_mte_resource_scope.py
& "$env:TEMP/vllm-bench-ruff/bin/ruff.exe" check script/diagnose_mte_resource_scope.py tests/ut/expert_offload/test_mte_resource_scope.py
git diff --check
```

核心交接逻辑变更还应运行 `test_moe_host_sync.py` 等相关测试；此次仅补跑入口
与摘要变更没有在蓝区做NPU验证。格式总入口 `E:/Git/bin/bash.exe format.sh ci`
最近因缺少pre-commit退出，须如实报告，不写成全套检查通过。

提交使用 `git commit -s`，按文件明确git add，排除a.txt。推送只到origin的
`dev/moeoffload-worker-poll`。最近正常代理推送曾挂起，中断该次push后，
临时使用 `git -c http.proxy= -c http.lowSpeedLimit=1 -c http.lowSpeedTime=20 push origin dev/moeoffload-worker-poll`
成功；不要擅自修改持久代理设置或把凭据打印出来。

每轮交付必须说明实际改了什么、本地验证了什么、绿区下一步执行什么，
不能停留在“建议继续观察”。当前用户准备开启新对话，暂不在没有新结果时
自行开始新一轮核数/等待策略修改。

## 13. 可以直接给新对话的开场文字

> 请先阅读 `D:\华为\nano2\vllm-ascend-worker-poll\docs\moe_mte_block_handoff_20261029_zh.md`，
> 当前工作树是worker-poll，分支dev/moeoffload-worker-poll，不是默认cwd下的另一个仓库。
> 继续定位纯MTE预取等待与真实模型scale转换阻塞。主流对照仍失败；辅助流组上次在
> checkpoint加载阶段退出，已准备只补跑native_main_aux，尚未返回新结果。
> 请区分已确认事实与假设，不重复已做的小实验，不把资源死锁当作已证明结论。
> 我会继续返回绿区summary.txt文本，你完成必要修改后提交推送并给我下一步命令。
