# MoE MTE 完整复核与等待位置实验（2026-10-29）

本文复核 `dev/moeoffload-worker-poll` 在 `1906b2c3` 的实现与历史记录，并说明本次新增的
默认关闭诊断。参考仓库 `D:\华为\nano2\vllm-ascend\vLLM-ascend-DSA` 仅只读审阅，
HEAD 为 `ce6900977e6a25377cd11205e3613ec2ec0208a4`。最初编写时未收到辅助流单组补跑结果；
另一对话中该组已通过短请求，现已合并到第14节，不能继续把它当作未执行实验。

**最新：第16节的callback/晚wait MTE采集均完成请求和stop，原加载阻塞未复现，用户已能查看profile。
load_prepare接近原版；提交＋等待均值仍有小幅差异。第17节准备两组同时启用C++ LRC的新对照，
包括大间隔阈值统计，尚待绿区结果；不是再次修改权重搬运或等待协议。**
根因仍未证实，停止继续枚举核数。等待位置实验检验“等待与消费者之前的真实模型计算重叠”
是否参与触发，不能直接区分资源调度、模型算子内部停滞和轮询访存影响。
第1–15节保留实验设计、运行命令与取证记录；最新实测见第16节，待执行实验见第17节。

## 1. 原项目与当前改造的完整顺序

单卡 Decode 中，路由结果进入外层 `update_weights`；它先汇合该层预取，再回传本轮路由与
`log2phy`。捕获时登记 Host 工作，重放时由设备依赖触发，CPU 不重跑外层 Python。
普通 eager 则直接调用 guarded 函数。MTP 和 Prefill 不应笼统视作同一张 Decode 图。

原 callback 路径使用 `_subscribe_report` / `_launch_host_func`；公开 Torch NPU 2.10
源码将其转为阻塞型 `aclrtLaunchCallback`，报告线程取得 GIL 后执行业务回调。
Runtime 的回调完成机制约束后续设备任务。这不等于新增 AIV kernel 忙等。
公开源码只能解释机制，不能代替绿区 dev 构建的每个内部实现。

当前 MTE 捕获路径为每个交接永久登记独立槽位和 callback/args，在交接流提交
`host_sync_publish` 与 `host_sync_wait`。publish 每次执行递增设备 state，经 MTE
写注册 Host request；C++ worker 领取任务、获取 GIL、进入原 `_update_weights`。
加载流上下文中的缓存规划、w13/w2、scale 拷贝与 refresh 执行后，仍需同步加载流。
callback 返回后 C++ 才发布该 generation 的 done。随后设备 wait 退出，交接流写回映射。

所以 **done 覆盖 callback 中的加载与 refresh 完成，但不包含 wait 后面的 log2phy H2D**。
消费者必须同时受 wait 和映射写回顺序保护。request/completion 在注册 Host 内存中，
每槽间距1024字节、两端分隔512字节；设备32字节 state 是另一块内存，不保存权重。

按需交接位于计算流。预取则是：

```text
计算流：上一交接完成 → ready → 中间模型计算 → 本层预取join → 本层路由D2H → 按需交接
预取流：                 wait ready → publish → strict wait → 映射H2D → done event
CPU worker：                         接请求 → 缓存规划 → 加载提交 → 等加载完成 → 写done generation
加载流：                                               H2D / 设备布局转换 / refresh
```

本分支从 `942868dd` 发展，仍保留原始拷贝。原实现主权重已可能采用 Storage 拷贝；
另一工作树后加的量化 Storage、CPU 重排、预打包与池布局调整没有移入。
`SyncLaunchStream` 改的是 Host 提交方式，不是取消异步设备工作或加载完成检查。
报告等待对照保留了该流与 native worker 后仍有有限通过，削弱它们单独必然失败的解释，
但没有排除与图调度组合的影响。

源码入口：`expert_offload_manager.py` 的 `update_weights`、`_launch_offload_callback`、
`_update_weights`、`_load_expert_weights_into_slots`、`_dispatch_prefetch`；
`host_sync.py` 的 `launch` / `diagnose_load`；`host_sync.cpp` 的 `poll`；
`host_sync_device/handshake.cpp` 的 publish/wait。

## 2. DSA 迁移中需要补充的三个差异

| 项目 | DSA 选定 raw-Hcomm / remote KV 路径 | 本地 MoE MTE |
| --- | --- | --- |
| 发布与代数 | Host launcher 的 `connection.nextGeneration++` 传给 AICPU entry，Hcomm WRITE payload 后发布 commit | AIV publish 对每个捕获槽的设备 state 自增，只发布 generation；Python任务已登记 |
| 数据完成 | CPU gather、URMA WRITE，等待数据 CQ 完成后再 WRITE observedGeneration 到 NPU completion | worker 向同一 NPU 提交含设备转换的加载任务，同步后在 Host completion 写 done |
| 默认/可选等待 | 默认在 AICPU entry 等；特定 token-gather 分支可延后到 Attention 消费 KV 前的 AIV WaitRemoteReady | 独立的1-block AIV wait 在预取 publish 后立即启动，持续经 MTE 读注册 Host 地址 |

对应源码（相对 DSA 仓库）：

- `torch_extension/csrc/npu_hcomm_kvcache_scatter_copy.cpp`：AICPU kernel 加载、提交。
- `torch_extension/csrc/npu_hcomm_kvcache_pool.cpp`：launcher 中 `connection.nextGeneration++` 的代数分配。
- `hcomm_kernel/kvcache_scatter_copy_hcomm.cpp`：初始化完成槽、payload/commit发布、是否延后等待的分支。
- `test/urma_mock_pool/urma_mock_pool.cpp`：`ProcessCommand`、数据 `PollJfc` 后发布完成、`PollCommands`。
- `csrc/src/a5_sparse_and_tail_attention_and_remote_scatter_copy/op_kernel/a5_sfa/arch35/sparse_flash_attention_service_vector_mla.h`：`WaitRemoteReady`。

DSA 的数据完成链不要求 Host 在等待期间再提交一个 NPU Transpose 才能完成。
当前迁移改变了生产者推进条件、等待执行单元、访存位置和图重放代数生命周期。
“两者都使用 generation”不能证明调度等价；也不能反推参考实现有问题。

## 3. 历史证据重新分类

| 证据 | 支持什么 | 尚不能支持什么 |
| --- | --- | --- |
| 原始 scale H2D task710完成；Transpose task711已下发后超时（PID66595、task_sn74901） | 拷贝包含设备工作，问题越过了H2D | kernel已开始执行、实际占核数、内部停在哪条指令 |
| `a6a6f2c4` 中 H2D完成，独立 `D2D_CONTIGUOUS` 也超时 | 原目标地址和非连续stride不是触发的必要条件 | 连续拷贝等于纯DMA、所有拷贝kernel相同 |
| request1/done0/status Running，worker在同步 | 已发现请求；业务完成尚未发生 | CPU没轮询到、CPU已写done但设备不可见 |
| 独立D2D、简单fork/join、有意100ms CPU等待时另一流计算通过 | 一个AIV wait并不必然阻挡所有其他计算 | 真实模型同样安全、设备调度条件完全相同 |
| 预取报告等待保留MTE发布/native worker/原始拷贝，去诊断屏障后有限通过 | 预取等待实现影响推进 | 唯一原因是核数、共享内存、GIL或某个算子bug |
| 有界AIV退出后报告严格守卫，worker随后完成 | 等待存续窗口参与触发 | AIV驻留和Host轮询访存两者谁起作用；已闭合死锁链 |
| 加载、路由、主流减核，实际图/下发配置改变后仍失败 | 这些已测干预不足以解除 | 全部资源调度关系已排除，或应该继续任意减核 |
| Torch原生主流限核仍失败 | 原始ctypes接口差异不足以单独解释 | 辅助流限核已经测过 |
| 辅助流组checkpoint加载2/46时退出 | 此次启动未完成 | 辅助流干预失败、scale再次阻塞、OOM或本次segfault已证明 |

本轮也检查了摘要分类：旧 `classify_report` 未验证辅助 MatMul 与目标流的归属。
现在将 `RESULT` 与 `RESOURCE_EVIDENCE` 分开。仅观察某个 MatMul14，或打印过设置请求，
均不足以证明它属于目标辅助流；缺少 capture pointer 到 replay stream 关联时输出
`dsa_aux=UNVERIFIED` 并保留实际观测。主流配置已验证也只指图配置，不是物理核占用。

`dsa_v1.py` 的辅助 weights_proj 与主流 query量化/KV scatter 是分支；原失败快照中
主流在 KvCompressEpilog 后才等辅助流 Event。该 kernel 源码未发现跨kernel轮询
MatMul结果，因此不能把“两个未完成任务”解释为内部互等。

## 4. 候选原因与可区分的新方向

| 优先级 | 假设与缺口 | 可执行方向、判据与停止条件 |
| --- | --- | --- |
| 1，下一次模型实验 | 预取wait过早驻留/访存，与消费者之前的真实主辅流计算组合阻碍推进 | 保留早publish，将同槽严格wait推迟至原join；通过支持该重叠窗口参与，仍失败则核对新快照是否已到gate。只做一轮，不追加核数扫描 |
| 2，解释结果所需 | 加载kernel未获执行机会，或已执行后停在流水/内部同步 | 对失败实例取实际kernel/core/block/PC证据。若现有日志只含下发/队列完成信息，明确不足；必要时在原scale前后加独立Host可读进度标记、无额外同步。前标记可见仅证明它自身启动并完成，不证明Transpose开始 |
| 3，有执行证据指向访存后 | tight loop每次MTE2读32B注册Host completion，可能影响共享通路；现有报告/有界对照同时撤掉驻留和访存 | 保持1-block与同generation严格门禁，仅改变读取节奏；不写假done、不用预算放行。通过也有时序混杂，未见相关执行证据时不先做多档调参 |
| 降级保留 | 未限辅助流的多算子资源关系尚未覆盖 | 已在跑的单组保留结果；尚未跑则让位于等待位置实验。有效一次后停止，不根据启动失败继续减核 |

线程TLS、默认流或隐含Event依赖仍需保留为候选。直接ACL同步超时、具体Transpose已下发，
使“仅仅卡在Python句柄获取或Host队列排空”不再是首选解释；并不证明所有子操作上下文正确。
共享内存完成可见性当前优先级较低，因为故障时Host尚未发布done。

设备PC证据很有价值，但不能把“发现工具路径”写成“已能附加全模型取所有核PC”。
公开 [msdebug指南](https://github.com/Ascend/msdebug/blob/master/docs/zh/user_guide/msdebug_user_guide.md)
要求相应驱动调试通道，并限制多算子程序只调指定单算子；绿区9.1工具与公开master的支持范围还需匹配。
因此本次不无依据地自动附加调试器或改设备调试配置。下发 `drvRet=0`、SQ完成位置、
图dump时间均不能补充为“kernel正在运行”的证据。

## 5. 本次实现：只延后严格预取等待及其后的映射写回

新增 `host_sync_diagnose_prefetch_wait_at_join=false`，仅允许 MTE、单卡、fate预取、
FULL/FULL_DECODE_ONLY完整捕获并启用图dump。禁止同时开报告等待、预算退出、控核、
scale分解或探测目标，避免混合干预。

```text
计算流：ready → 原中间模型计算 → record gate → wait done → 原路由D2H/按需交接/消费者
预取流：wait ready → publish → wait gate → 原strict wait → 原映射H2D → record done
CPU/加载流：          请求发现 → 原规划/原拷贝/refresh/同步 → 写同generation done
```

这会改变等待和映射写回的时序，不能描述成完全不改图拓扑或只测孤立同步原语。
通过后仍需分离驻留资源与轮询访存，也不能直接称为根因修复或性能优化。

实现保护：

- 捕获时分离publish与wait，token绑定捕获对象、slot和原预取流；拒绝重复、跨流、跨捕获使用。
- gate先在主流record，预取流再wait，避免等待从未record的Event；主流的done等待位于gate之后。
- 同槽strict wait完成后才能映射H2D和record done；原按需join仍在共享路由缓冲复用之前。
- 共享预取staging只允许一个待汇合生产者；下一次stage前检查上一个已汇合。
- 每次捕获首尾检查pending为空，部分捕获失败后禁止继续；事件与回调引用保持生命周期。
- eager、原callback、未开诊断的MTE保持原顺序。没有新增设备kernel或环境变量。

新脚本不重跑旧对照，仅运行一组，沿用真实模型、MTP=2、短提示词、max_tokens=5。
保留前八批逐项同步和Runtime DEBUG，便于与早期无限核诊断比较；核数不改。
执行1次预热和3次有限请求，跳过profiler，避免已知采集失败混淆请求结果。
诊断时序不能作为正式性能数据；与旧记录不是同时同负载对照，结果解释保留此边界。

## 6. 绿区命令与回传判据

如果辅助流补跑已启动，先让它结束并保留summary，不要并行启动本实验。
如果尚未启动，则直接运行下面的一次新实验：

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_wait_placement.py --npu 7
```

只返回新建 `bench_results/mte_wait_placement_*/summary.txt` 的检查脱敏文本。
构建、服务、图dump和Runtime明细留在绿区；脚本不执行Git操作。

- **启动/捕获失败**：返回上下文；不计为机制对照，不自动重跑。
- **没有非空publish/join配对或图结构无法识别**：先看本轮捕获证据；pending0本身不能证明干预发生。
- **`GATE_PAIR`**：检查同图publish→gate wait→strict wait→mapping→done，以及另一流gate record先于done wait。
  这是捕获结构证据，Event名称复用或schema不匹配时不强行关联，也不代表运行时PC。
- **请求通过且结构成立**：支持消费者之前的重叠窗口参与触发；保留digests/usage，尚无同轮回调精度基线。
- **仍scale超时且结构成立**：检查新快照是否位于gate之前、strict wait或后续任务；前段计算本应在gate前完成，
  不再照抄旧stream/task编号或继续限核。后续选择实际执行状态/前后进度证据。

蓝区只有CPU测试与静态检查，尚未做目标NPU验证。实验脚本在绿区构建现有Host/MTE库、
运行回归检查后才启动模型；捕获失败或不支持当前模型配置时明确退出。

本轮蓝区相关110项CPU测试通过，包括旧交接、延后等待、图摘要、资源证据和profiler跳过/异常清理。
修改的独立脚本、host_sync、acl_graph及新增测试通过Ruff；manager与ascend_config的14条
既有Ruff问题已逐条对照HEAD，未扩展本轮范围修复。`git diff --check`通过。
要求的 `E:/Git/bin/bash.exe format.sh ci` 已执行，但因缺少pre-commit未能启动全套检查。

## 7. 首次等待位置实验：捕获前退出，尚未形成机制对照

绿区返回 `mte_wait_placement_50028589`。上次交付版本为 `9370a1e7`；本次summary未独立记录Git HEAD。
API PID91361，EngineCore PID91939；服务在178秒时退出，早于1800秒启动上限。
这是新实验的启动失败，不是旧scale超时再次复现，也不是等待位置方案已被反证。

本轮实际进度：

- 20:35:40目标模型offload finalize结束，MTE worker策略打印与dispatcher启用均出现。
  这也说明初始化握手已完成；没有真实offload callback完成记录。
- MTP/draft checkpoint为46/46，20:35:44打印 `Loading weights took 3.75 seconds`。
- 最后一条EngineCore日志为同秒的topk_indices_buffer共享。
- `WAIT_PLACEMENT_RECORDS=0`、`CAPTURE_END_RECORDS=0`、`GRAPH_DUMP begin=0 end=0`，没有图文件。
- API随后报告引擎初始化失败，没有对应EngineCore异常栈或真实退出信号。

本轮与此前辅助流组的checkpoint 2/46退出处于不同阶段，不能合并成同一个加载故障。
`Service exited during startup: 1` 是运行器启动的服务/API退出码，不能当作EngineCore退出码。
现有dmesg候选仍是11:16的另一进程segfault，与20:35:44不匹配，不用于本次归因。
主机可用内存约616GiB既不能证明峰值/cgroup足够，也不是本次OOM证据。

源码复核提供下一层定位范围，但没有证明某条语句出错：

1. 最后日志位于 `llm_base_proposer._maybe_share_topk_indices` 内部，之后仍会遍历draft模块重绑buffer。
2. 调用者随后执行 `_maybe_share_lm_head`；MTP且DeepSeek MLA的分支会做 `torch.equal` 再共享head。
3. 返回父runner、退出设备内存统计范围后才打印 `Loading model weights took ... GB`；摘要未见该记录。
   该范围是按最后可见日志与源码顺序推断，日志缺失并非完整执行轨迹。

本轮 `enforce_eager=true`，draft的 `use_cuda_graph` 为False；其wrapper分支通常不会执行。
主模型wrapper以FULL构造，配置FULL_DECODE_ONLY也满足新校验。因此目前不能把退出归为新增的
wrapper模式检查。MTE worker初始化到退出约4秒，未见新槽发布或FATAL，也不支持120秒watchdog解释。
format-29转换警告之后，finalize和MTE握手继续完成；该历史回退警告不能单独作为退出根因。

下一步读取已有运行的服务日志、EngineCore/API分别对应的Runtime日志与本轮时间窗口内的内核记录。
新增入口不导入Torch、不初始化NPU、不启动模型，仅读输入并写一份新的取证summary：

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/inspect_mte_startup_exit.py bench_results/mte_wait_placement_50028589
```

仍只返回新建 `bench_results/mte_startup_inspect_*/summary.txt` 的检查脱敏文本。
脚本末尾打印 `SUMMARY=` 路径；输入日志保留在绿区。默认只扫描当前用户的 `ascend/log`、
`/var/log/npu/slog` 和 `/usr/slog` 中本轮PID命名的文本日志，并报告无法读取或未覆盖的范围。

优先找真实退出状态、原生fatal/异常及最后设备提交/同步线索；普通Runtime尾部仅表示最后可见操作，
不自动等于失败点。没有命中、读取范围外或日志轮转时保持 `NOT_ESTABLISHED`，不把API退出码、
旧OOM/segfault或缺少错误记录补成原因。本次不重跑等待位置、主流限核或辅助流组。
如果既有记录仍不能解释退出，下一次才在上述收尾范围加入阶段begin/end、原生崩溃栈和真实
EngineCore退出状态采集，维持模型、加载策略和等待实验变量不变。

本次仅增加离线取证脚本与测试，更新诊断文档；模型和MTE运行逻辑未修改。
14项CPU测试中13项通过，1项因本地Windows缺少符号链接创建权限跳过。
新增Python文件的Ruff检查与格式检查通过，`git diff --check`通过。
再次执行 `format.sh ci`，仍因缺少pre-commit而未能启动全套检查。

## 8. 首次离线取证：修正误筛，补齐同一运行的日志覆盖

本次返回仍来自 `mte_wait_placement_50028589` 的既有日志，没有新模型运行。
服务仍为EngineCore PID91939、API PID91361；日期窗口为2026-10-29 20:35:29–20:35:59。
找到11个PID命名日志并全部选入，但旧提取器有两项明确缺陷：

- `FAULT` 的裸 `timeout` 匹配把 `StreamSynchronize entry ... timeout=-1ms`、
  Event的 `timeout=0s` 当成故障。最新plog的1601条候选不是1601次超时。
  对用户实际返回的20条候选逐条重分类：19条是普通参数日志，1条是已知format转换回退的错误记录。
  这只重分类已经显示的内容，不能推断未显示的记录没有异常。
- 每文件仅首尾6条、全局20条的展示预算被上述噪声占用。另有4个约20MB的plog丢弃了
  约3.6–4.2MB前缀，部分已读尾区全部落在目标时间窗口，因此窗口内记录没有完整覆盖。

有效新增线索及边界：

1. 最新2.8MB的plog完整读到20:35:44.490401，stream61仍有
   `FinishedTaskReclaim ... task_type=66(KERNEL_AIVEC)` 记录。
   `curPos=1431`、`tarPos=1432`、`recycleHead=1432`是队列位置，不能称为“task1432完成”。
   该流有回收记录不等于所有设备流或模型加载后处理都已完成。
2. 内核窗口内出现20:35:40.519896的 `soc_get_dev_topology` 参数错误，设备参数为7和65。
   其host PID3970861尚未与容器PID91939建立映射；其后近4秒仍有上述Runtime回收记录。
   保留为待关联线索，目前不能据此判定退出根因，也不能由未见OOM/崩溃行排除这些情况。
3. 仍未取得EngineCore真实退出状态，也没有等待位置实验的捕获/执行证据。
   资源死锁等旧候选原因的证据等级不变。

修正版离线工具输出 `INSPECTOR_VERSION=2`，本轮只修改该工具、测试和本文档：

- 数值timeout配置与超时回报分开，ERROR级记录即使包含正常timeout参数仍保留。
  非零result/ret、signal和exit信息仅作候选，不能自动解释为最初故障或自然退出。
- 对选定文件从首字节流式扫描至打开时的大小，记录扫描字节、结束大小、原始行号和完整候选计数。
  短日期使用本次服务日志提供的年份，不取执行机器的当前年份。
- 在所有文件中先区分fatal、error和其他症状，再分配展示额度；各类保留有界的首尾样本。
  无日期fatal也保留优先级，并明确标为 `UNCORRELATED`，不强行归入本轮时间窗口。
- 仍最多选12个PID文本日志，所有未选文件、未展示候选、文本截断和读取失败均保留边界。
  无命中只说明所选日志与窗口中未得到线索，不等于进程没有原生崩溃或外部终止。

下一步只补读同一批日志，修复的是上一轮提取遗漏；不启动模型、NPU、限核或辅助流实验：

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/inspect_mte_startup_exit.py bench_results/mte_wait_placement_50028589
```

返回 `SUMMARY=` 指向的新summary脱敏文本；应看到 `INSPECTOR_VERSION=2`。
优先检查真实fatal/非零退出/原生错误，以及其前后是否仍有进度。
如果完整筛选后仍缺少退出原因，下一次才实施第7节所述阶段日志与真实子进程退出状态采集，
保持等待位置实验和模型参数不变，不继续重复离线筛选或盲目重试加载。

本轮23项CPU测试中22项通过，1项因本地Windows缺少符号链接创建权限跳过。
覆盖本次实际参数日志、3000条普通参数噪声、跨文件fatal优先、无日期fatal与清理错误共存、
文件首部错误、短日期和进程信号等边界。Ruff检查与格式检查通过，`git diff --check`通过。
`format.sh ci`已执行，仍因未安装pre-commit而无法启动全套检查。

## 9. 离线取证结束：一次带启动观测的模型运行

用户返回v2摘要 `mte_startup_inspect_c6260bcc1f8b4cd0abe3d69cb168fae2`。
11个日志的 `snapshot_bytes=scanned_bytes=end_size`，无未选文件、无文本截断、无日期缺失候选。
目标窗口内fatal为0；2条error仍是20:35:39的同一次format-29回退。
20条symptom主要是 `CtxGetSysParamOpt ret=0x7110014` 和 `GetFaultEvent` 查询记录。
查询行为本身不是故障发生记录；未把这些返回值解释为致命原因。保留的首次/末次记录之外仍有
2条symptom未展示，但完整扫描没有检出额外fatal/error。API退出后有其Runtime卸载记录，
仍不能代替EngineCore的实际退出状态。

本轮结论是：所选已有日志不足以解释启动退出。停止重复离线提取，也不再修改筛选器来制造进展。
此前两轮离线命令属于启动退出支线，没有增加原始scale阻塞机制的直接证据。
下一步仅运行一组：沿用早发布、消费者join处严格等待的配置，同时补齐启动观测。

本次实现与审阅边界：

- `--trace-startup` 仅允许配合MTE等待位置诊断。服务的模型、MTP、捕获尺寸、核数、worker策略、
  原始拷贝和已有逐项同步保持原实验设置；原命令的serve参数和shell设置完整保留。
- 诊断入口先核对PATH中的原 `vllm` console声明的Python与当前解释器真实路径一致。
  不一致、找不到入口或不能识别shebang时在导入vLLM前退出，不以另一Python环境继续实验。
- API进程只观察标准库multiprocessing已有的start、poll以及terminate/kill请求。
  fork/spawn/forkserver的实际poll实现均覆盖，包装函数保留原调用参数、次数、返回值和异常；
  不增加轮询线程或主动waitpid。观察保持到解释器退出清理，记录父进程取得的真实退出状态。
- 阶段helper由既有 `host_sync_diagnose_prefetch_wait_at_join=true` 启用；默认生产配置关闭。
  这意味着不带新CLI标志的等待位置诊断也会有阶段标记，但只有新CLI入口会安装进程观察。
  helper只用CPU，标记topk共享、lm_head共享、原有equal、drafter加载、内存统计退出及主wrapper构造。
  原 `torch.equal` 每层仍调用一次，没有新增tensor读取、设备同步或权重打印。
- API和EngineCore分别启用FD2/all_threads的faulthandler。它不能捕获SIGKILL，不能保证覆盖所有
  原生信号处理路径，因此仍需父进程退出状态。标记写失败不改变模型异常或阻止原进程清理。
- 汇总关联明确的EngineCore PID，分别列出父进程signal请求、runner/外层清理与退出观测顺序。
  退出观测时刻不是实际死亡时刻；清理后的负退出码不自动归为自然故障，没有请求记录也不排除外部kill。

阶段日志与进程观察均做了独立交叉审阅。模型runner只增加加载边界标记，不改变执行分支与顺序。
`runner.memory_profile_exit` 的begin只在加载体正常走到底时出现；缺少该begin不能反推异常清理时
Profiler没有执行退出函数。未闭合阶段只能缩小最后可见CPU代码范围，不能直接认定对应NPU算子有错。

绿区下一步命令（会启动一次模型，不是再次读取旧日志）：

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_wait_placement.py --npu 7 --trace-startup
```

入口先运行新增CPU观察回归检查（Linux覆盖真实fork/spawn/forkserver及SIGKILL状态），然后沿用
Host/MTE构建和既有回归流程。启动通过后继续1次预热、3次有限请求，仍跳过profiler，不自动追加第二组。
仅返回新建 `bench_results/mte_wait_placement_*/summary.txt` 的检查脱敏文本。

判读优先级：

1. **观测未启用/前置检查失败**：`STARTUP_TRACE_STATUS=UNVERIFIED`，检查明确FAILED记录与入口异常。
   没有EngineCore退出码时必须保持 `NOT_OBSERVED`，不得使用API返回码补齐。
2. **再次启动退出**：看EngineCore真实退出状态、清理请求顺序、最后未闭合阶段和崩溃栈。
   Python异常、原生信号、外部终止/父进程清理按证据分别处理，再做最小修复；不转去限核。
3. **启动通过并完成捕获/请求**：回到第6节的GATE_PAIR、严格等待和有限请求结果判据。
   本轮启动成功不代表此前启动问题根因已修复；启动支线暂时不挡住主线即可继续原阻塞实验。

本轮仍没有真实NPU结果，不作精度或性能结论。

蓝区相关138项CPU测试中136项通过，2项因Windows符号链接权限/POSIX信号支持跳过；
绿区入口会重跑新增的观察与阶段回归，覆盖Linux实际启动方式和SIGKILL退出状态。
13个修改或新增Python文件通过Ruff检查与格式检查；model_runner的8项既有Ruff问题已与HEAD
按规则和消息对照，没有新增，未扩大范围重排该文件。`git diff --check`通过。
`format.sh ci`已执行，因本地缺少pre-commit未能启动全套检查。

## 10. 等待位置有效通过：下一步去掉最初8批加载诊断

用户返回 `mte_wait_placement_c4cac7ed/summary.txt`，对应上一节命令及提交 `4f583c33`。
这次已经进入原阻塞问题的机制对照，不是又一次启动支线取证：

- `WRAPPER_EXIT=0`、`RUNNER_EXIT=0`、`RESULT=finite_requests_passed`；1次预热和3次有限请求完成。
  三次请求的输出摘要同为 `21e9f347a4cd40b1d4c2263761c4737c6543bdc4027b2c3ade06ce19f176621a`，
  每次prompt为5 tokens、completion为5 tokens。没有本次匹配的callback输出对照，不能宣称精度等价。
- 有效配置为MTE、早发布/消费者join等待、原始Torch拷贝、`all/locked/yield` worker策略；
  保留加载诊断与native诊断等待，没有限核。实际图仍有28核辅助流MatMul、56核routing与1核wait。
- `WAIT_PLACEMENT_RECORDS=86`、`CAPTURE_END_RECORDS=1`；capture1发布和汇合均为42个槽，pending为0。
  `capture_biruq2yk/graph_dump`检出42对结构。第一对为publish `[63,21,1]` →
  主流gate record `[63,23,118]` → 预取流gate wait `[63,21,2]` → strict wait `[63,21,3]` →
  mapping `[63,21,4]` → done record `[63,21,5]` → 主流done wait `[63,23,119]`。
  这是捕获依赖结构证据，不是各kernel的实际起止时间或全部generation的执行轨迹。
  摘要内capture_end重复展示来自首尾摘录重叠，实际计数只有1；本轮已去掉该展示重复。
- 已显示的batch8、layer4 `w2_weight_scale[L4,E37->S22]`及scale_refresh均完成。
  原scale布局仍存在：CPU连续、NPU目标stride为 `[2,64,1]`。这证明该次原始布局路径可以完成，
  不是把原Transpose拷贝换成连续目标后得到的通过。

启动观测完整：18条阶段标记无异常或未闭合阶段，EngineCore PID93533、API PID92955。
runner先记录 `cleanup_begin reason=completed`，其后父进程发出terminate请求，再观测到EngineCore
`exitcode=-15 (SIGTERM)`。不能把这次结束判为再次启动崩溃。原先PID91939为何提前退出仍未解释，
本次启动成功不表示已修复其根因。

`MOE_RESULT.engine_error_seen=true`需要单独保留边界：现有实现只是全文包含 `EngineDeadError` 的检查，
用户摘要未给出该字符串所在行。已有顺序证据支持结束时发生了主动清理，但不足以把该字符串也归因于清理。
新运行会保留原布尔值，并输出其在唯一cleanup标记前后的行数、首尾样例；没有唯一标记时位置为null。
这些是合并日志中的位置，不是精确事件时刻或因果归属。也补充已有callback计数及errors/dropped，
不增加设备查询或同步。

### 10.1 当前判断与优先级

1. **优先验证通过是否依赖初始加载诊断。** `diagnose_load()`只对最初8个有任务的加载批次执行
   prior-stream、逐项copy和scale_refresh的额外同步；之后本来就走普通批量加载。
   本次说明后续普通路径可以推进，仍未验证最初8批也不需要这些额外同步。
   下一组只关闭这条诊断执行路径，恢复从第一批开始的普通提交；不是取消协议完成条件。
2. **等待窗口仍是有依据的方向，具体机制未证实。** 历史report等待/有界等待对照与本次join等待通过，
   共同支持提前驻留的严格AIV等待与真实前段计算的重叠参与触发。设备资源调度、轮询访存干扰、
   内部依赖/时序仍没有被区分。本轮同时延后了映射写回，也不能只凭本轮把作用完全归给wait。
   当前主机可用内存等外部状态与早先失败运行不同，尚不是同环境下完整匹配的因果A/B。
3. **普通加载通过后，再做callback正确性对照与覆盖扩展。** 现有三次相同输出只说明该短请求重复一致。
   后续需要冷热缓存、不同token_rows/捕获尺寸与多次generation的正确性检查，再考虑性能。
   当前启动观测、graph dump、Runtime debug和计时标记仍在，不能拿request_mean_us作性能结论。

暂不重跑辅助流限核组或旧主流失败组，也不再离线提取旧启动日志；这些不是当前优先证据缺口。
若关闭初始加载诊断后失败，应先区分启动失败与已发布请求停滞，再根据worker快照/捕获依赖定位。
没有逐项记录时，不能仅凭失败就说再次卡在同一个scale/Transpose任务；同样不能直接称为资源死锁。

### 10.2 下一次只运行这一组

新增脚本选项 `--no-load-diagnostics`，仅用于MTE等待位置实验：明确把继承配置中的
`host_sync_diagnose_load`和`host_sync_diagnose_native_wait`覆盖为false，拒绝同时请求逐项诊断、
改写scale拷贝或加载限核。普通模式下若仍出现 `[MOE-MTE LOAD]`，运行器不会输出成功结果。
生产路径、默认选项和协议实现未改；保留最终 `_synchronize_h2d()`、callback返回后done发布、
严格generation wait、映射写回及主流done join。保持启动观测、图转储、核数、worker与请求设置。

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_wait_placement.py --npu 7 --trace-startup --no-load-diagnostics
```

仍只启动一次模型，1次预热+3次有限请求，不自动追加其他组。返回新建summary的检查脱敏文本。
预期 `LOAD_DIAGNOSTICS_REQUESTED=False`、有效配置上述两项为false、`LOAD_DIAGNOSTIC_RECORDS=0`，
并检查捕获pending为0、结构配对、请求结果、callback计数与实际退出顺序。零条诊断日志本身不是成功证据。

本轮142项相关CPU测试中140项通过，2项因Windows符号链接权限/POSIX信号支持跳过。
覆盖选项逐层传递、继承true值显式覆盖、冲突拒绝、原负载保留及错误日志位置边界。
修改的5个Python文件Ruff检查与格式检查通过，`git diff --check`通过。
`format.sh ci`仍因缺少pre-commit无法启动完整检查；新增无逐项诊断模式的NPU结果待上述命令返回。

## 11. 普通加载通过：转入有限输出对照与更长执行覆盖

用户返回 `mte_wait_placement_e5189f82/summary.txt`，对应提交 `e4236847`。
`LOAD_DIAGNOSTICS_REQUESTED=False`，有效配置的load/native_wait诊断均为false，
`LOAD_DIAGNOSTIC_RECORDS=0`。因此本轮确实从第一批开始走普通批量加载，没有借助前8批逐项同步。

结果与边界：

- wrapper与runner退出码均为0，1次预热+3次请求完成。三个文本摘要仍与上一轮相同，
  每个请求completion为5 tokens；这是两次MTE诊断配置间的短输入一致性，尚无callback对照。
- 计时窗口记录510次业务回调，errors=0、dropped=0。计数不能直接等同于MTE generation数，
  也不能证明全部捕获尺寸均被重放或全部数据依赖均已覆盖。
- 捕获开始/结束各一次，发布与汇合各42槽、pending=0；`capture_6ki985ja/graph_dump`的42对结构成立。
  主流与辅助流计算核数未改，仍保留严格generation等待和原始拷贝。
- `EngineDeadError`只出现1行，在唯一cleanup标记之后，之前为0行；EngineCore PID95109的
  SIGTERM退出观测也在runner `reason=completed`及父进程terminate请求之后。
  本轮结束符合主动清理顺序，没有证据把这一行计作请求失败。日志顺序本身仍不是完整因果证明。
- request_mean_us约3481752，不与上一轮约3201599比较性能。输入短且诊断日志、图转储、MSTX等仍开启。

这次证据排除了“该次通过必须依赖最初8批额外同步”的解释。等待位置方案已有两次有限通过，
其中一次是普通加载；这增强了等待/映射写回时序方向的可实施性，但没有区分底层调度、轮询访存、
算子内部依赖等机制。旧启动失败原因也没有因这两次成功而得到解释。

### 11.1 为什么下一步做输出对照

当前更紧迫的缺口是：延后等待与映射写回后，完成的请求是否仍与原callback路径输出一致，
以及更长生成和不同输入是否还能推进。因此不重复前8批诊断开关、不补跑限核组；下一组是功能验证，
不是继续证明资源死锁的实验。若失败，按启动、请求停滞、输出差异分别继续。

同一次命令顺序启动两个新服务：callback完成并清理后才启动MTE。两者使用同一次读取的
`test_graph2.sh`、相同模型/MTP/缓存配置、Python策略和Torch数据拷贝。callback保留原报告回调及加载流，
MTE保留当前SyncLaunchStream、早发布和join处严格等待；只有MTE启用启动观察与图转储。
两者均关闭逐项加载诊断、跳过profiler，并沿用既有Runtime debug设置；不作性能比较。

固定三种输入为英文解释蓝天、中文计算17×23、Python平方和函数。每个服务先用首个输入预热一次，
然后依次执行三种输入，temperature=0，每个请求最多生成32 tokens。默认旧实验仍是原提示词/5 tokens。
更长请求有机会暴露重复代数与缓存变化的问题，但不预先声称一定覆盖冷热专家、特定miss数或所有图尺寸。

运行器记录实际prompt的SHA256、max_tokens与temperature，逐请求比较返回text的SHA256及完整usage。
空输出、缺失结果、计时错误或请求计划不符都不能报告通过；不要求业务回调计数相等。
原始请求及响应保留在各服务目录的 `completion_records.jsonl`，含warmup/measured标记，方便出现差异后
继续读同一轮数据。summary仅展示摘要，不打印原始生成文本。完全一致也只构成这三种输入上的有限证据，
不是logits精度验证，更不是阻塞根因或性能结论。

### 11.2 绿区命令与判读

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_wait_placement.py --npu 7 --trace-startup --no-load-diagnostics --verify-callback
```

总计两次服务启动，每个服务1次预热+3次请求。保持各服务超时设置；外层总上限6000秒，
只清理本命令拥有的进程组。callback失败则不会继续启动MTE。

仍返回新建 `bench_results/mte_wait_placement_*/summary.txt` 的检查脱敏文本。检查：

1. 两份 `MOE_RESULT`、两份有效配置，`PAIR_RESULT.status=passed`，三个text_match和usage_match均为true。
   wrapper必须同时退出0，才标记 `RESULT=finite_output_match`。
2. MTE的逐项诊断仍关闭、捕获pending=0、图结构配对成立；请求输出与捕获结构分开判断，
   没有图证据时不得把输出一致升级为交接结构已验证。
3. 若只有callback结果、服务未就绪或请求中断，先按对应服务日志处理；若输出不一致，使用本轮原始响应
   判断具体差异。更长请求再次停滞时使用本轮generation/worker证据，不套用旧Transpose任务号。

本轮相关145项CPU测试中143项通过、2项因Windows平台能力限制跳过。覆盖不同输入的真实请求顺序、
原始响应落盘、异常清理、callback配置保持与失败后停止、结果缺失/空输出/不匹配拒绝、摘要失败路径。
7个修改的Python文件通过Ruff检查和格式检查，`git diff --check`通过；`format.sh ci`因缺少pre-commit
无法启动完整检查。新对照尚待绿区NPU结果。

## 12. 三种输入输出一致：验证是否依赖图debug（2026-10-30）

用户返回 `mte_wait_placement_6e30a074/summary.txt`，对应提交 `ea74b9d4`；日志事件发生于10月29日晚。
callback、MTE两次服务均完成1次预热和3次请求，wrapper退出0、`PAIR_RESULT.status=passed`。
三个不同输入的文本SHA256逐项一致，完整usage也逐项一致：prompt为13、11、17 tokens，
completion均为32 tokens。这构成32-token范围内的有限输出对照，不是完整回答语义、logits或通用精度验收。

MTE的加载诊断保持关闭，记录数0；capture1登记与汇合各42槽，pending=0。
`capture_mmw60xf8/graph_dump`确认42对结构，依然是早publish、主流gate、严格wait、mapping、done join。
callback保持原报告回调及加载流，MTE保持SyncLaunchStream；模型、MTP、缓存、策略、输入顺序相同。
两组约4.939秒与4.983秒的请求均值不作性能比较。

### 12.1 两项尚未解释的观测

- 业务计时记录分别为3825、3826，errors和dropped均为0。`CallbackTimings.wrap_update()`按活动计时窗口
  或外层上下文纳入记录，`stop()`停止新窗口并等待已纳入调用排空；它不是逐generation协议账本。
  当前没有相同输出必须对应相同业务调用数的已证不变量，也没有从本次分组数据定位相差1次的来源。
  不把差值直接判为漏调/重复执行，也不声称它已被解释。
- MTE EngineCore PID98738在runner记录 `cleanup_begin reason=completed`、父进程请求SIGTERM之后，
  被观测为 `exitcode=-9 (SIGKILL)`。相对terminate请求约128毫秒；本地 `stop_owned_service()` 的
  SIGKILL兜底在SIGTERM后30秒才发出，因此摘要不能用该兜底解释这次SIGKILL。观察器不覆盖所有直接
  `os.kill`、进程组信号或外部发送者，发送来源仍未知。请求已经完成不等于退出方式正常；此记录也不能
  反推请求阶段复现了旧阻塞。MTE唯一EngineDeadError行在清理标记之后；callback未启用该观察器，
  其同名错误行位置仍为null，不能借用MTE的顺序给callback归因。旧启动失败根因同样未明。

暂不为上述结束阶段信息重新启动离线取证循环；保留本轮日志，若后续出现清理前异常或服务无法退出，
再按真实发送者与时间线补证。启动阶段没有未闭合phase，不能把此轮退出称为启动崩溃。

### 12.2 下一步的代码依据与边界

当前 `validate_host_sync_config()` 把join等待与graph dump绑定；但 `ACLGraphWrapper` 中
`_moe_join_wait`独立控制capture上下文、pending检查，`_moe_graph_dump`只控制显式
`NPUGraph.enable_debug_mode()`和转储接口。已有三次MTE通过全部处于图debug开启状态，
尚未验证不调用这些接口时是否仍能推进。因此下一步解除这一条校验绑定，核验非图debug捕获。
这不是重新假设图debug已被证明影响调度，而是补齐从诊断路径走向实际使用的明确证据缺口。

本轮不改publish/wait/generation协议、映射顺序或加载实现。保留单卡MTE/fate预取、完整图捕获要求，
以及禁止限核、report等待、改写scale拷贝等混合配置的校验。默认仍关闭join等待选项。
新增脚本选项 `--no-graph-dump` 只允许用于现有callback/MTE验证套件，显式覆盖继承的graph dump=true；
如MTE仍出现转储标记或新图目录，运行器拒绝报告成功。对照运行保持Runtime debug日志、MSTX、
计时、启动观察、输入与生成长度，因而仍不能作为生产性能结果。

无图debug时，capture登记/汇合记录及pending检查继续保留，但它们只是CPU捕获路径证据。
摘要将标记 `GATE_GRAPH=NOT_REQUESTED`，不把旧运行的42对设备图配对当成本轮的实测结构，
也不把主动关闭转储解释为取证失败。

### 12.3 绿区命令

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_wait_placement.py --npu 7 --trace-startup --no-load-diagnostics --verify-callback --no-graph-dump
```

仍顺序启动两个服务，各1次预热+3次相同输入、最多32个生成token。比较新的callback/MTE结果，
避免把不同运行环境下的旧输出直接充作当前基线。返回新建summary的检查脱敏文本。
判据为有效配置graph dump=false、`GRAPH_DUMP_REQUESTED=False`/记录0，capture pending=0，
两组请求完整、三个text_match/usage_match均为true。没有图文件是预期行为。

若通过，说明此轮推进及有限输出一致不依赖显式图debug接口；后续再评估去除其余诊断、扩大覆盖。
若失败，先区分启动退出、请求停滞与输出差异，用本轮日志定位；不能直接证明图debug改变了某种硬件资源。

相关146项CPU测试中144项通过，2项因Windows平台能力限制跳过；覆盖无转储配置合法性、继承值覆盖、
完整套件参数传递、意外转储拒绝成功，以及摘要不推断设备图结构。6个修改的Python文件Ruff检查和
格式检查通过，`git diff --check`通过。`format.sh ci`仍因缺少pre-commit未能运行完整检查，
本轮新增配置的NPU结果待上述命令返回。

## 13. 回到阻塞主线：固定晚映射，只改变严格wait位置（2026-10-30）

用户返回关闭图debug的结果（EngineCore PID102344、API PID101766，10月30日10:23:49清理）。
两个服务均完成三种输入各32 tokens，三项text_match及usage_match全部为true。
有效配置graph dump=false、转储记录0、CPU capture登记/汇合各42槽、pending=0。
本轮没有设备图转储，不能复用旧运行的42对结构当作本轮设备证据。业务计时记录3825/3827、错误与丢失为0。
MTE清理后仍观测到SIGKILL（距terminate约129毫秒），发送来源未明；不将其作为请求阶段阻塞证据。

### 13.1 本轮阻塞排查进展

- **已知主线继续有效**：历史报告等待、有界等待以及延后严格等待的对照，共同支持预取AIV等待
  与真实模型执行的组合参与触发。不是重新从零怀疑“AIV等待有没有关系”。
- **本次新增证据**：本轮汇合处等待方案的推进和有限输出一致不依赖显式图debug接口。
  这是对已通过方案的条件核验；没有新增原始未完成设备任务、PC或资源依赖证据。
- **仍未解释**：提前严格wait与真实主辅流计算中的哪一段组合，如何阻碍加载；驻留资源、轮询访存、
  内部依赖等机制仍未区分。前面几轮正确性/可用性验证不能累计成底层根因已定位。
- **环境边界**：本轮启动前MemAvailable为1455545552 kB，上轮约931153268 kB，变化明显。
  这是主机快照，不是NPU资源或容器限额；既不能据此判为内存原因，也不能用历史失败充作严格同轮对照。

后续每轮按“新增阻塞证据—排除/削弱了什么—仍未知什么—下一实验区分什么”汇报。
功能验证另列，若没有新增根因证据就明确说没有。

### 13.2 下一实验具体定位什么

当前成功方案同时延后了严格wait和预取映射写回。为取得当前代码/环境下可判读的阻塞样本，
本轮固定映射在消费者gate之后，比较严格wait是否也必须等待gate。并不是重新检验AIV是否相关，
而是分离等待与映射写回的干预作用，给后续具体任务定位建立可复现对照。

| 本轮组别 | 预取流顺序 | 用途 |
| --- | --- | --- |
| late_wait | publish → wait gate → strict wait → mapping → done | 同轮推进对照 |
| early_wait | publish → strict wait → wait gate → mapping → done | 恢复提前等待，映射仍延后 |

主流两组都是原计算 → record gate → wait done。early_wait在原预取publish位置立即提交严格wait，
不是等到消费者Python代码执行时才提交一个无gate的wait。映射仍经过主流gate，不能越过它提前写回。
提前提交wait后仍保留共享staging所有权，下一次stage会检查未完成的映射汇合；孤立映射会使捕获失败。
同generation、原始拷贝、加载同步、worker策略、核数保持不变，不用假done或有界放行。

新增默认关闭的配置 `host_sync_diagnose_prefetch_wait_before_gate`，要求已有join捕获和graph dump。
不改变默认路径。CPU测试检查early组只提交一次strict wait、映射和done位于gate之后，并保持孤立捕获拒绝。

新入口顺序跑两个MTE服务，不再跑callback精度组。两组统一采用原阻塞排查的短负载：
1次预热+3次请求、原提示词、max_tokens=5；统一打开图debug以核验交换后的顺序，关闭逐项加载诊断。
相对上轮功能验证改变了负载和观测，因此结论只在新两组内部比较，不直接拿上轮结果归因。
只有late_wait完成请求且图结构与CPU捕获登记对应，才启动early_wait；对照不成立就停止。

### 13.3 判据及证据边界

- `CONTROL_NOT_ESTABLISHED`：late_wait未完成或结构缺失，不启动early组，不根据失效对照推理。
- `EARLY_WAIT_LATE_MAPPING_PENDING_AT_LOAD_SYNC`：early组顺序成立；同一watchdog快照中，活动worker
  对应已登记预取槽、request未完成且栈位于 `_synchronize_h2d`，同槽同generation随后达到watchdog时限。
  结合late组通过，支持在预取映射仍延后的条件下，恢复提前严格等待仍可触发加载推进问题。
  这可削弱“必须提前写回预取映射才会卡住”的解释，不能直接证明资源死锁。
- `EARLY_WAIT_LATE_MAPPING_PASSED`：本轮允许提前wait的图也通过，未得到新的阻塞样本。
  不能由允许重叠推断实际发生了与旧故障相同的重叠；映射时序、加载量和环境变化仍需区分。
- `INCONCLUSIVE`：提前组启动退出、结构缺失、或失败证据达不到上述门槛。保持未知，不自动归类为旧scale卡点。

图dump只证明捕获依赖，不证明kernel实际启动或核占用。单次Host快照也不是完整执行轨迹。
未得到具体设备任务/PC时，不能称为已再次定位到旧Transpose任务711，更不能照抄历史PID/stream/task。
已有业务计时分组会随摘要保留，供查看实际负载；没有对应快照时pending计数为null而非“观测到0”。

### 13.4 绿区命令

```bash
cd /home/l00992607/vllm-ascend-worker-poll
git pull --ff-only origin dev/moeoffload-worker-poll
python -B script/diagnose_mte_wait_order.py --npu 7
```

返回新建 `bench_results/mte_wait_order_*/summary.txt` 的检查脱敏文本。
重点是两个CASE的有效配置、设备图顺序、`WAIT_ORDER_FACTS`及末尾 `BLOCKING_PROGRESS`。
每组独立3000秒外层上限、只清理本组进程；记录各组启动前主机内存，不自动追加第三组或限核组。

本轮相关152项CPU测试中150项通过、2项因Windows能力限制跳过。覆盖早wait下的原槽/单次严格等待、
gate后的映射与done顺序、staging与孤立捕获保护、两种设备图顺序识别、活动worker与主线程栈区分、
同代watchdog关联，以及对照不成立时停止。7个新增/修改的独立Python文件通过Ruff检查与格式检查；
`ascend_config.py`两条既有Ruff问题已与HEAD按规则/消息核对，没有新增。`git diff --check`通过。
`format.sh ci`因本地缺少pre-commit未能启动全套检查。新干预尚无NPU结果，以上均不是硬件机制证明。

## 14. 合并两边证据：一次等待顺序对照后，转向候选方案的成本验收（2026-10-30）

本次回到原对话推进。已审阅1906b2c3到7bfefa35的提交、manager/dispatcher/捕获上下文、
等待顺序运行器及第10–13节验证记录。没有将另一对话的文档结果假装为本次重新执行的NPU实验。
本轮未修改运行逻辑，先避免在已有未执行的有效对照之外再增加新变量。

### 14.1 补入此前未同步的辅助流限核结果

原对话已经收到单组native_main_aux的成功结果，EngineCore PID90073：

- 主流14/28、路由5、辅助流14/28，使用Torch原生接口；纯MTE严格等待与原始拷贝。
- 图中stream22的同名BF16 MatMul为14 blocks，主流HcPre14、路由5、wait1。
  它与上一轮已识别的辅助MatMul签名/图位置相符，结合设置记录支持目标干预生效；
  仍不是运行时物理核占用证据，不把capture句柄直接等同于replay流编号。
- WRAPPER_EXIT=0、RUNNER_EXIT=0、请求完成5个生成token。request_mean_us约3784084，
  没有同轮callback输出基线，不作精度或性能验收。
- profile_status=invalid，错误为External init callback/registerClient线程不一致，
  engine_error_seen=true。请求通过不能升级成采集与整个服务均无错误。
- 对照native_main（PID85186）保持主流14/28、路由5而辅助MatMul28时仍在slot4失败。
  两轮系统负载未严格控制，不把它称为唯一因果证明。

所以目前有两条能够改变原失败结果的干预：缩小模型辅助计算需求；缩短预取AIV等待
与消费者之前模型计算的重叠窗口。这比“所有原始scale拷贝在MTE下都不能执行”的解释更符合已有结果。
它们也降低了单纯Host轮询未发现任务的优先级，但仍未区分设备资源调度、轮询访存、
算子内部依赖与时序影响；未证明闭合的硬件死锁链。

### 14.2 对另一对话修改的审阅结论

9370a1e7引入早publish、消费者join处严格wait和映射写回。4f583c33及其前后的提交
主要增加启动观测和日志筛选。e4236847去掉逐项拷贝诊断后验证推进，ea74b9d4验证
三种输入与callback输出一致，4f69fd4a补了关闭图debug的验证。最后7bfefa35提供
晚映射固定条件下的早/晚wait对照，当前还没有该对照的绿区结果。

代码中的关键依赖保护仍在：

1. publish在原预取位置提交，worker的规划和权重搬运没有被推迟到消费者。
2. 主计算流到原本的预取join位置才record gate，预取流先wait gate，再执行严格wait。
3. completion仍只在原CPU函数及加载同步完成后发布；没有假done或预算到期放行。
4. 严格wait之后才写回log2phy、record done；主流wait done后才回传/复用共享路由缓冲。
5. 捕获中保持原槽位、事件和Python参数引用，拒绝未汇合生产者、跨捕获token、
   下一生产者提前覆盖staging。捕获失败后禁止复用半成品状态。

这提供了可继续评估的绕行方案，不意味着所有输入、图尺寸、预取模式和生命周期均已验收。
当前只支持明确的单卡/fate/完整捕获条件，不能顺手扩大到其他模式。

### 14.3 为什么优先保留延后等待，而不是把限核当作最终方案

保持“早通知CPU，晚在消费点检查完成”符合预取目的，CPU与加载流仍有机会和中间模型计算重叠。
相比当前早wait，代码新增的是每次预取join的一对gate Event记录/等待；原done Event与消费者join仍在。
事件创建和任务登记发生在捕获阶段，重放仍需执行对应设备Event任务，不能把它的运行成本写成零。
映射写回也从原先可能提前完成变为必须在gate之后完成，可能增加消费者关键路径。

该方案不主动减少模型可用核数，不新增CPU scale重排、权重副本或额外权重数据传输，
因此是比直接固定限核更值得先测成本的候选。现有约4.9秒等诊断请求均值不能证明它开销小；
目前还不能保证相对callback无退化或已经减少400微秒空泡。

### 14.4 保留现成对照，设置明确的停止点

下一步使用7bfefa35已经写好的入口，不再新写一套重复测试：

```bash
python -B script/diagnose_mte_wait_order.py --npu 7
```

先late_wait，再early_wait；两者都保持晚映射、相同严格完成语义和原始拷贝。
若用户已经执行，应直接分析其summary，不要求重跑。结果按第13节门槛判断：

- late通过、early在同槽预取加载同步处阻塞：支持提前wait本身的重叠窗口参与，
  削弱“只有提前映射才会卡住”。此时已有候选绕行，不再无限等待设备PC证据才开始成本评估。
- 两组都通过：尚未复现提前wait的失败，不能说原问题已消失或映射已被证明为根因。
  保留晚wait候选，基于同轮证据决定是否还需要原拓扑对照，不自动追加第三组。
- 对照启动失败或结构不成立：先处理对应失败，不把它加入阻塞正反证据。

这一轮不是性能测试。之后对候选方案的成本验收应回到callback基线，保持模型、请求与缓存设置相同，
关闭Runtime DEBUG、图dump、启动观察、逐项加载同步和profiler等额外诊断；必须核对实际生效配置。
采用同等计时开关比较预取/按需的load_prepare、copy_submit、load_submit、synchronize与load_and_wait，
同时测设备侧预取join剩余等待与整体请求时延。CPU分项之和不能替代设备上MEMCPY间隔。
若逐项打点有明显扰动，分离不打点请求窗口与低频分项采样，而不是拿重诊断数据作收益结论。

若原始权重搬运成本恢复接近callback、且消费者join没有明显退化，可以继续有限覆盖后将该路径
作为可选绕行；若gate/晚映射关键路径开销较大，再针对实测项减开销。暂不预先承诺收益百分比，
不为追求时延移除严格完成检查，也不在未测成本前把诊断选项改成默认。

本轮重新运行test_moe_host_sync（55项）、test_mte_prefetch_join（14项）、
test_mte_wait_placement_runner（23项），共92项CPU测试通过。本轮没有新增NPU结果。

## 15. 固定晚映射的早/晚wait结果：开始评估候选方案的成本（2026-10-30）

### 15.1 本轮阻塞结论

实验目录为 `bench_results/mte_wait_order_17c1f323`，两个服务依次运行。
两组保持原始Tensor权重拷贝、全核模型计算、all/locked/yield工作线程策略，
映射写回均在消费者gate之后；关闭逐项加载诊断，开启图debug核对捕获结构。

| 对照 | 捕获结构 | 运行结果 |
| --- | --- | --- |
| late_wait，PID103916 | publish → gate_wait → strict_wait → mapping → done | 1次预热和3次5-token请求完成；510条回调计时，错误/丢弃均为0 |
| early_wait，PID105466 | publish → strict_wait → gate_wait → mapping → done | 首次预热中，预取slot2、layer1、generation1超过120秒仍未完成 |

两组都核对出42组结构，捕获结束published/joined槽位一致、pending=0。
第一组映射任务都为stream21/task4，早/晚对照只是task2/3的严格wait和gate wait交换。
这是捕获顺序证据，不将dump中的时间字段解释成设备实测时延。

早wait失败时request_generation=1、done_generation=0；工作线程已接到预取任务，栈位于
`_prefetch_host_cb → _update_weights → _synchronize_h2d → SyncLaunchStream.synchronize()`。
因此本轮支持：**在映射位置固定后，提前执行严格AIV等待仍会触发预取加载同步停滞；
延后等待能绕过当前有限请求下的阻塞。** 不能再仅归因为提前映射写回，也不是Host工作线程
没有发现任务。通用单组标签`RESULT=inconclusive`没有识别无逐项诊断的加载卡点；
本轮联合结构与watchdog的最终判据是`EARLY_WAIT_LATE_MAPPING_PENDING_AT_LOAD_SYNC`。

晚wait的3个输出摘要均为`21e9f347...`，请求均值约3219.330ms。这轮开启Runtime DEBUG、
MSTX与内部计时，而且没有callback组，不能拿这个均值宣称性能收益。
晚组唯一EngineDeadError出现在cleanup标记之后，并有SIGTERM清理记录；本轮不把它当作请求阶段阻塞。

仍未确定的部分是设备底层机制：本轮没有取得超时的具体设备任务/PC，不能直接声称再次
映射到旧Transpose；也未证明物理核资源死锁或轮询访存拥塞。此前辅助流限核通过与本轮
结果共同提高“提前等待与模型并行执行相互影响”的可信度，但不替代上述缺失证据。

### 15.2 下一轮：同一入口完成原callback与晚wait MTE的成本对照

新增可选入口，现有诊断模式与默认运行方式保留：

```bash
python -B script/run_moe_host_sync.py --npu 7 --compare-join-cost \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh
```

脚本仍只启动两个服务：先原callback，再晚wait MTE。每个服务按以下顺序完成：

1. 3次相同短请求预热；与test_graph2.sh对齐模型、MTP、预取与缓存设置。
2. 10次内部计时关闭的请求，记录客户端请求时延、输出摘要与token用量。
3. 同一服务开启分项计时，再运行10次相同请求，记录预取/按需的Host分项时延。
4. 完成后清理服务，再启动另一组；任一组请求失败立即停止，保留已完成组的数据。

两个请求窗口均为`The future of AI is`、max_tokens=5、temperature=0；可以通过
`--requests`、`--warmup-requests`修改次数。计时关闭窗口仍安装了包装器，只是收集器未激活，
不是完全移除计时代码的进程。先后窗口的差值也可能含缓存、温度与负载变化，不能全部归于打点成本。

此预设会清除继承的host_sync诊断设置，仅为MTE重新开启晚wait；强制关闭
moe_offload_debug、Runtime DEBUG、图dump、启动观测、MSTX、profiler和逐项加载诊断，
不限制核数、不更换权重布局、不改变all/locked/yield策略。原callback加载流与MTE的
SyncLaunchStream差异继续保留，属于当前方案成本；不会为了好看而偷偷换回回调交接。

`summary.txt`只保存关键结果，详细服务和构建输出在实验目录内的独立日志。
每组完成后先保存`callback/cost_report.json`、`mte/cost_report.json`，防止后续失败丢数据；
用户只需复制summary文本，不传文件。本轮不调用start/stop_profile，避免采集扰动和停止解析等待。

首要指标按prefetch/reactive及token_rows分别比较：`load_prepare`、`copy_submit`、
`scale_refresh`、`load_submit`、`synchronize`，以及`load_and_wait`的均值。
同时保留`total`、`observe`、`choose_victims`，每个实测项展示均值和p95；
load_and_wait仅相加均值，不把两个p95相加。先核对输出、token用量、样本数、层与加载数量摘要；
工作量不同则标为描述性比较，不写成同工作量收益。

成本验收尚待本轮绿区结果。通过不等于设备MEMCPY间隔或两个host_callback间距已经降低；
这些还需之后单独短profile核对。若搬权重成本接近原callback，优先保留晚wait候选并评估
汇合处剩余等待；若仍有明显回退，就针对本次实测项分析，不回到无约束的核数枚举。

蓝区验证：成本配置/统计校验17项、成本运行器9项、原MoE dispatch 55项、原推理运行器20项、
预取join 14项、等待位置运行器23项，共138项CPU测试通过。这里只验证脚本控制、计时窗口、
数据判据与已有依赖保护；新的成本对比尚未在NPU上执行。本次没有更改设备算子或MoE运行逻辑。

## 16. 短profile两组通过：分开判断加载阻塞、搬运成本与采集可用性（2026-10-30）

用户返回 `bench_results/mte_join_profile_01`。callback PID106943、MTE PID107993；
均为3次预热、随后采集1个5-token请求，开启MSTX/内部计时/profiler，moe_offload_debug=false。
模型、MTP=2、预取层数1、原始权重拷贝一致；MTE只启用消费者处严格等待，未限核。

本轮阻塞结论：两组请求通过，文本摘要和usage一致，各有170条有效回调计时、错误/丢弃为0。
预取84次、加载76个专家；按需86次、加载401个专家，分组加载数量摘要相同。
stop_profile分别10.3秒和7.9秒完成，均发现一个导出时间线及3164次MSTX标签字符串。
没有复现原预取加载同步超时；但这些事实不能替代时间线本体检查。

| Host均值（μs） | 预取callback | 预取MTE | 按需callback | 按需MTE |
| --- | ---: | ---: | ---: | ---: |
| load_prepare | 36.365 | 32.635 | 128.749 | 131.311 |
| copy_submit | 42.451 | 134.191 | 157.736 | 606.596 |
| load_submit | 91.059 | 183.276 | 333.363 | 803.908 |
| synchronize | 212.187 | 125.326 | 1025.700 | 614.627 |
| load_submit + synchronize | 303.246 | 308.602 | 1359.063 | 1418.535 |
| 内层update_weights总时长 | 426.704 | 418.224 | 1499.643 | 1553.702 |

当前load_prepare没有旧适配版本那样明显上升。主要差异落在copy_submit：MTE直接提交的Host时间
更长，后续synchronize更短；这符合提交路径与异步工作重叠不同的可能性，不能仅凭分项断言
拷贝实际执行更慢或全部增长都已被抵消。提交＋等待均值预取增加5.356μs（1.77%），
按需增加59.472μs（4.38%）；仍是有采集扰动的有限样本。

请求时延3185.172ms→3229.634ms，增加44.463ms（1.40%）。每组只有一个请求，
不能称为稳定退化或收益。Host分项之和不等于设备MEMCPY间隙；本轮还没有量化设备空泡。
gap_debug约0.25–0.34μs，与两组关闭调试输出相符，不能写成新C++迁移或MTE握手收益。

### 16.1 两个分析脚本问题及修正

第一，旧`analyze_callback_gaps.events()`只接受`{"traceEvents":[...]}`，不能读取此前Ascend
出现过的顶层数组`[...]`。本轮“traceEvents不在文件头1MB内”不是文件截断证据。
现补齐两种格式、完整外层与EOF校验，以及CANN数字字符串时间戳；不覆盖或修复原始采集。

第二，旧`assess_profile_report()`看到`External init callback must run in same thread as registerClient`
就标整组invalid，未检查NPU时间线。上游PyTorch2.10对应的
[Kineto检查](https://github.com/pytorch/kineto/blob/31f85df8fbd89c188f14ef10f1ec65379786b943/libkineto/src/libkineto_api.cpp#L18-L30)
是在初始化线程与注册线程不同时打印错误并跳过client初始化，不会直接宣告时间线导出失败。
厂商安装包差异未验证，仍保留这条兼容性报错，不能直接说完全无影响。

更新后的判据区分：JSON损坏或明确parser失败仍为invalid；目标轨道/标记未核实时为unverified；
JSON与目标轨道基本检查通过但保留兼容性报错时为complete_with_warnings。这只说明基本内容
可读、目标事件可见，不证明完整Host关联、跨时钟对齐或全部采集内容无损。

本轮MTE还有一条EngineDeadError，旧入口没有记录唯一cleanup边界，原因未知；
不能套用上一轮“确认在cleanup之后”的结论。后续采集入口补记cleanup边界，现有日志只读检查上下文。

### 16.2 先检查已有文件，不重跑模型

```bash
python -B script/inspect_moe_profile.py bench_results/mte_join_profile_01
```

脚本不导入Torch、不初始化NPU、不写文件，流式检查两组JSON、Ascend Hardware任务、
MTE publish/wait、实际MSTX区间与嵌套，打印同一Host进程上相邻host_callback的方向性间隔，
并保留服务日志兼容性报错/退出上下文。用户已能在MindStudio打开原始
`profile/{callback,mte}/python/*_ascend_pt`目录并查看时间线；脚本检查不是打开可视化的前置条件。
自动统计报错和MindStudio无法显示是不同的问题，不把前者写成后者。

### 16.3 当前正确性范围、请求计时与间隔统计口径

当前晚等待机制已在第12、13节的两轮对照中通过三种输入各32个生成token的文本摘要与usage一致性验证，
其中一轮关闭图调试转储但仍启用图捕获/重放。后续`7bfefa35`只新增默认关闭的提前wait对照分支，
晚wait路径保持gate等待→严格wait→映射写回→done；之后成本和profile脚本修改没有替换该运行机制。
因此已有验证适用于当前方案，但不等于所有输入/图尺寸、多卡、长时间运行或logits精度已验收。
本轮5-token请求也与callback输出一致；结束阶段EngineDeadError仍保留为未解释事项。

请求时延在客户端调用`/v1/completions`前后计时，包含请求编码、HTTP往返、服务处理和完整响应解析；
不包含服务启动、预热、start_profile/stop_profile、后续摘要计算。它不是纯NPU执行时间。
本轮记录上MTE比callback多44.463ms，但每组只有一个带profiler/计时的请求，尚无稳定端到端收益结论。

间隔按“后一个时间块的开始−前一个时间块的结束”计算，单位μs，输出样本数、均值、p50、p95、最大值。
以下命令直接读取本轮已保存的两个profile，不启动模型、不占用NPU、不写文件：

```bash
python -B script/inspect_moe_profile.py bench_results/mte_join_profile_01 --gaps-only
```

- `HOST_CALLBACK_GAP`：同一Host进程中按时间相邻的回调，分别统计预取→按需和按需→预取；
  范围是整个采集窗口，不自动认定属于同一层/同一次请求，也不是只统计Decode。
- `MEMCPY_GAP`：同一Ascend Hardware进程、同一Stream内所有MEMCPY_ASYNC时间块之间的间隔，
  不跨设备流拼接。重叠部分合并为已覆盖区间并单独计数，不作负的空闲时间；零间隔纳入统计。
  缺少两个拷贝区间时返回无样本，不将其均值写成0。
- 该流全部相邻拷贝包含同一批内w13/w2/scale之间的间隔和不同批之间的间隔，且可能包含路由/映射传输。
  不能把这个总平均值自动标成预取→按需的批次间空泡。若要精确统计那类边界，需要按实际权重加载流
  和可靠的任务/批次关联进一步筛选，不以时长阈值猜测归属。

Host回调间隔和设备拷贝间隔分别使用自己的时间轴；其中可能有其他计算或传输，均不直接解释为
整个NPU空闲或纯CPU–NPU同步开销。当前只扩展离线统计，不改MoE运行机制。

## 17. 开启cpp LRC后的回调与晚等待MTE对照（2026-10-30）

### 17.1 实验目的与固定条件

此前最新profile两组均使用Python LRC。本轮将两组同时切到已验证的`cpp`策略，
比较原callback交接与消费者处严格等待的MTE交接，观察分项时延及时间线间隔。
`cpp`包含observe、choose_victims及其配套状态管理；没有开启`cpp_fused`融合规划、
C++ load_submit或ACL搬运实验。原Python策略仍可选，原callback运行机制没有被替换。

当前晚等待修复的CPU/NPU八步流程、事件保护和copy_submit差异，已详细补入
[MTE交接流程第12节](moe_mte_handoff_flow_zh.md#12-当前阻塞规避提前发布预取在消费者处严格等待2026-10-30)。
这次仅扩展实验运行器与统计，不改变该修复的设备协议、权重布局或加载方法。

| 项目 | 两组设置 |
| --- | --- |
| 输入与生成 | 原短提示词，temperature=0，max_tokens=5 |
| 预热/测量 | 每组3次预热、1次采集请求；预热不进入profile |
| LRC | cpp，先构建独立扩展并强制通过Python/C++策略一致性检查 |
| 模型、MTP、预取层数和容量 | 读取指定test_graph2.sh的实际配置，不另设一套模型参数 |
| 权重准备与搬运 | 原Python任务准备＋Torch拷贝，无新增CPU重排/scale Storage适配 |
| MTE工作线程 | all/locked/yield，保持上一轮，不混入其它worker优化 |
| 诊断干预 | 关闭Runtime DEBUG、图转储、逐项加载诊断、限核和调试打印 |
| 采集 | 开启MSTX、内部计时和torch profiler；两组顺序运行，各保存独立profile |

默认只采集一个请求，是为了减少把多个HTTP请求之间的停顿纳入大间隔统计；
这也保持与上一轮短profile一致。可通过`--requests`增加数量，但整个采集窗口的间隔统计
会包含请求边界，不会自动只保留层内/层间时间。该轮端到端仍含采集扰动，不是无profiler的稳定性能验收。

### 17.2 一次执行

绿区同步提交后，在worker-poll仓库中执行；脚本内部没有Git操作：

```bash
python -B script/run_cpp_mte_profile_compare.py --npu 7 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --large-gap-us 100 \
  --output-dir bench_results/cpp_mte_profile_01
```

物理NPU编号可修改，输出目录须未存在。默认停止/导出预算120秒，超时保留已测时延和已产生文件，
清理自己启动的服务后继续另一组，不伪造缺失的profile统计。扩展构建或策略一致性检查失败则不启动模型。
完整构建/推理详情保留在绿区日志，关键数据在输出目录`summary.txt`，只需复制经检查的该文件文本。

两个profile分别位于：

```text
bench_results/cpp_mte_profile_01/profile/callback/cpp/<worker>_ascend_pt/
bench_results/cpp_mte_profile_01/profile/mte/cpp/<worker>_ascend_pt/
```

本轮保持原trace_view.json供MindStudio查看，不额外复制一份带人工overlay的完整时间线。
服务日志名为`inference_cpp.log`，其中cpp表示LRC策略；父目录callback或mte才表示交接方式。
通用运行器也新增`--cache-policy-backend cpp`，不指定时仍默认python。

### 17.3 大间隔筛选与数据判读

参数`--large-gap-us 100`的精确定义是：仅把**长度大于等于100μs**的间隔纳入大间隔子集。
两组使用相同阈值，继续保留全部间隔的样本数/均值/p50/p95/最大值；额外打印大间隔数量、
占全部间隔的比例、累计时长及其均值/分位数。某组无大间隔时数量为0、均值为空，不能写成0μs均值收益。
因为有些间隔优化后会落到阈值以下，不能只比较剩下的大间隔均值而忽略数量变化。

阈值可以滤掉多数批次内部的小间隔，但并不识别权重批次。较慢的批次内间隔仍可能被选中，
较快的真实批次边界也可能被排除；MEMCPY统计仍按实际设备进程和流分别输出。
Host的两个方向依然单独统计。它们是时间线区间统计，不能直接写成纯同步开销或整卡空闲时间。

同一脚本会给出请求时间、输出摘要/usage一致性、按预取/按需及token_rows分组的回调分项表，
包括observe、choose_victims、load_prepare、copy_submit、scale_refresh、load_submit、synchronize、
内层total与外围指标，另计算load_submit+synchronize的均值。p95不通过分项相加推导。
两组计时错误、样本丢失、输出不一致须明确失败；工作量摘要不同则不作为等工作量收益结论。
profile导出或解析失败与已经取得的CPU计时分别报告。

已有profile也可用不同阈值重新离线统计，无需重跑模型：

```bash
python -B script/inspect_moe_profile.py bench_results/cpp_mte_profile_01 \
  --cache-policy-backend cpp --gaps-only --large-gap-us 100
```

### 17.4 绿区结果：cpp_mte_join_profile_01

本轮两组均为cpp LRC、原Python/Torch权重路径，每组3次预热和1次5-token采集请求。
请求规格、文本摘要和usage一致；预取各84次、加载76个专家，按需各86次、加载401个专家，
工作量摘要一致。MTE没有复现原加载阻塞；这是本轮有限请求的结论，不扩展为完整精度或长期稳定性验收。

两份profile均为`complete_with_warnings`，时间线可用。External init线程警告仍在，
不等于Ascend时间线损坏；两组EngineDeadError均出现在清理边界之后，边界之前为0。
该分类依据合并日志顺序，不单凭此确定清理异常的根因。

| 指标（均值） | callback | MTE | MTE变化 |
| --- | ---: | ---: | ---: |
| 请求时间（ms） | 3196.486 | 3228.607 | +1.00% |
| 预取load_prepare（μs） | 37.590 | 32.196 | -5.394μs |
| 预取copy_submit（μs） | 38.583 | 120.744 | +82.161μs |
| 预取load_submit（μs） | 89.476 | 167.998 | +78.522μs |
| 预取synchronize（μs） | 212.806 | 136.544 | -76.262μs |
| 预取load_and_wait（μs） | 302.282 | 304.543 | +0.75% |
| 按需load_prepare（μs） | 147.336 | 124.894 | -22.442μs |
| 按需copy_submit（μs） | 155.712 | 534.693 | +378.981μs |
| 按需load_submit（μs） | 350.465 | 720.450 | +369.985μs |
| 按需synchronize（μs） | 1023.797 | 684.445 | -339.352μs |
| 按需load_and_wait（μs） | 1374.262 | 1404.894 | +2.23% |

表中差值按已显示的小数计算，原汇总用未舍入值计算时末位可能不同。
load_prepare已不高于基线；copy_submit仍显著增加，后续synchronize大幅减少，
需连同提交＋等待判断：预取多2.261μs，按需多30.632μs。提交上下文差异见流程文档12.7，
本轮不能把全部增量归因于队列机制，也不能把Host提交时间直接解释成设备传输时间。
内层total预取405.369→388.737μs，按需1484.220→1503.292μs。
两组均用cpp，故不是独立验证Python→C++收益的对照；按需observe为6.062/6.401μs，
choose_victims为4.807/4.711μs，预取choose_victims为5.875/5.598μs。

| Host相邻回调间隔 | callback均值/p95（μs） | MTE均值/p95（μs） | 均值变化 |
| --- | ---: | ---: | ---: |
| 预取→按需（各84个） | 115.506 / 339.525 | 93.866 / 354.350 | -18.73% |
| 按需→预取（各84个） | 105.744 / 229.062 | 65.774 / 73.775 | -37.80% |

100μs阈值下，预取→按需大间隔22→23个，均值243.841→196.120μs，累计5364.50→4510.75μs；
按需→预取大间隔20→1个，累计3330.25→148.75μs。
预取→按需均值改善但p95略升，大间隔数量没有下降，不能表述为尾部时延全面改善。
旧MTE曾观察到的约400μs平均间隔上升，本轮对应方向没有重现；不能将不同历史实验直接相减为修复收益。

**MEMCPY间隔：按可视化观察修正轨道对应。**

用户在MindStudio中关注的权重拷贝轨道为callback Stream57、MTE Stream24，
下表采用本轮summary中这两条轨道的数据。此前列举的Stream27/59各770个拷贝只是未经确认的候选，
不能用于代替用户所看的轨道，也不能据此得出目标拷贝间隔下降的结论。
这里的轨道对应来自用户可视化观察，脚本本身尚未自动验证每个拷贝的权重语义。

| 指标 | 回调版Stream57 | MTE版Stream24 |
| --- | ---: | ---: |
| 拷贝事件数 | 45940 | 45940 |
| 非重叠间隔数 | 42993 | 42851 |
| 全部非重叠间隔均值（μs） | 8.842 | 9.631 |
| ≥100μs间隔数量 | 198 | 199 |
| 大间隔均值（μs） | 1883.489 | 2023.941 |
| 大间隔p50（μs） | 347.875 | 293.750 |
| 大间隔p95（μs） | 7337.587 | 8678.750 |
| 大间隔最大值（μs） | 16469.250 | 17903.250 |
| 大间隔累计时长（ms） | 372.931 | 402.764 |

两条轨道有大量零或接近零的间隔，全部间隔均值不适合表示用户关注的大空泡。
按100μs阈值，大间隔均值增加约7.46%、累计时长增加约8.00%，p50下降但p95上升。
因此本轮目标MEMCPY轨道的大间隔没有体现整体改善，不能由Host回调间隔缩短推出MEMCPY空泡也缩短。
阈值仍只按长度筛选，不自动区分预取→按需、按需→预取或批次内部；间隔中也可能有模型计算，
不是纯同步开销或整卡空闲时间。其他稀疏轨道中的约100ms间隔不合并到该统计。

结论：当前路径完成了本轮阻塞规避和有限输出一致性验证；Host回调间隔改善，权重准备未新增明显负担，
但用户关注的MEMCPY轨道大间隔均值和累计时长增加，权重提交＋等待及端到端也略高。
仅1次采集请求且含profiler、MSTX与计时扰动，
不足以宣称稳定性能收益或认定约1%的差异具有统计显著性。

**copy_submit与SyncLaunchStream的关系。**

普通加载流在启用Host队列时，可以把提交工作先入队，由后台线程向Runtime下发；
SyncLaunchStream则由当前工作线程直接提交。因此原来可能在后台完成的部分工作，
会计入当前线程的copy_submit墙钟时间。它是本轮提交增大、随后等待减少的合理解释之一，
但当前两组同时改变了交接机制、执行线程和加载流，没有仅改变加载流的有效对照，不能定量认定全部增量由它造成。

该流由提交`be7b674d`引入，目的是绕过Torch Host提交队列，减少MTE工作线程加载任务对模型提交路径的依赖。
当时担心图执行等待影响队列推进，使CPU需要提交的加载任务无法及时下发；这是当时的排查假设，
并非已经证实的本次设备Transpose阻塞根因。改流后仍曾复现阻塞，当前能够完成请求的修复是延后预取严格等待。
SyncLaunchStream不是共享内存协议本身的必需条件，其在当前晚等待路径中的必要性尚未单独验证。
名字中的Sync指Host直接提交，不代表每次copy_都等待设备搬完；实际完成仍由_synchronize_h2d确认。

## 18. MTE加载流单变量对照（待绿区执行）

本轮直接比较同一晚等待MTE路径的两种加载流，不再比较callback与MTE。
第一组`sync_launch`使用现有`SyncLaunchStream`，第二组`ordinary`使用普通`Stream`。
新增配置`host_sync_load_stream`默认仍为`sync_launch`；callback后端始终沿用普通流。
两组都保留提前发布、消费者门控之后的严格等待、原始权重/scale拷贝和`_synchronize_h2d`，
都使用cpp LRC、all/locked/yield工作线程，不引入重排、限核、布局适配或额外逐项同步。

实验脚本只构建一次所需扩展，然后顺序启动两个独立服务，核验实际流类型日志。
沿用test_graph2.sh模型参数，每组3次预热、1次5-token采集请求，保存profile、MSTX与分项计时。
比较输出摘要/usage、实际加载工作量，以及预取/按需的load_prepare、copy_submit、load_submit、
synchronize、load_and_wait和Host/MEMCPY间隔。先看提交＋等待总时间，再看提交与等待各自的分布，
不能把计时从一个区间转移到另一个区间当成整体加速。实际结果尚未返回。

```bash
python -B script/run_mte_load_stream_compare.py --npu 7 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --request-timeout 120 --profile-timeout 120 \
  --output-dir bench_results/mte_load_stream_01
```

输出目录必须未存在。只需返回`bench_results/mte_load_stream_01/summary.txt`的内容。
profile分别位于该目录的`profile/sync_launch/cpp/`和`profile/ordinary/cpp/`，流编号由本轮分配，
不能继续硬套上一轮的57/24。普通流请求超时后停止该服务，保留已完成的第一组计时、日志摘录及profile；
不会让严格等待算子提前放行，也不把失败组记作性能样本。请求超时后仍有计时停止、profile停止及服务清理阶段，
所以120秒不是整个脚本的总退出预算；启动预算默认1800秒。
本轮仅用于比较加载流，短请求与profiler扰动限制保持不变。
