# MoE MTE 完整复核与等待位置实验（2026-10-29）

本文复核 `dev/moeoffload-worker-poll` 在 `1906b2c3` 的实现与历史记录，并说明本次新增的
默认关闭诊断。参考仓库 `D:\华为\nano2\vllm-ascend\vLLM-ascend-DSA` 仅只读审阅，
HEAD 为 `ce6900977e6a25377cd11205e3613ec2ec0208a4`。没有收到辅助流单组补跑的新结果。

**最新：消费者汇合处等待已取得一次有效有限通过；下一步关闭前8批额外加载诊断，见第10节。**
根因仍未证实，停止继续枚举核数。等待位置实验检验“等待与消费者之前的真实模型计算重叠”
是否参与触发，不能直接区分资源调度、模型算子内部停滞和轮询访存影响。
第1–9节保留实验设计、首次运行命令与启动取证记录；后续操作以第10节为准。

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
