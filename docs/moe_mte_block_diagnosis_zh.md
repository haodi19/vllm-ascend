# MTE 旧版权重搬运阻塞定位

## 已确认事实与目标

本次基于 worker-poll 分支保留的旧版权重搬运逻辑，先定位，不增加 CPU 重排或 Storage 适配。
诊断固定 `all + locked + yield`，关闭后续轮询优化，避免把两类问题混在一起。

此前逐项同步日志显示，batch 2、layer 1 的预取任务中：

- `w13` 的 8 MiB 拷贝完成。
- `w2` 的 4 MiB 拷贝完成。
- `w13_weight_scale` 的 512 KiB `copy_()` 返回，随后加载流同步没有完成。
- 请求代号为 1，完成代号仍为 0，worker Python 栈在加载流同步；后续 scale 和 refresh 尚未执行。

这排除了“CPU 一直没发现请求”作为该次阻塞位置，也说明不是已经记录 COMPLETE 的 w13 主权重。
HTTP 请求失败只是引擎 watchdog 退出后的结果。不能据此认定 scale 的 DMA 一定没搬完。

## 正常交接如何完成

1. 按需交接在计算流、预取交接在预取流：前置数据准备后，设备发布算子递增 generation，
   通过 MTE 把请求代号写入注册的 CPU 控制内存。
2. 同一交接流的 AIV 等待算子读取 CPU 完成区。它只有观察到对应 done generation 才正常退出。
   推理使用严格等待版本，不通过轮询次数超时放行，避免消费尚未加载的权重。
3. C++ worker 发现请求，领取任务并获取 GIL，调用 Python 包装函数，再进入 `_update_weights`。
4. `_update_weights` 切到 `self.load_stream`。MTE 版的加载流是 `SyncLaunchStream`，
   权重和量化参数通过原来的 `destination.copy_(source, non_blocking=True)` 提交。
5. worker 等加载流完成。只有 Python 函数返回，C++ worker 才发布 done generation。
6. 设备等待算子退出，交接流继续映射写回及后续任务。

注册共享内存中存放的是请求/完成控制字段，不是所有专家权重。专家权重沿原 H2D 路径搬运。
正常情况下，加载流能独立完成，步骤 5 与步骤 2 的等待不会构成死锁。

## 哪些改动可能形成循环等待

共同的已知等待关系是：

```text
交接流上的 AIV wait
    等 worker 写 done
worker
    等 load_stream 完成
```

只有再证明“load_stream 的完成依赖交接流越过 AIV wait”或等价资源依赖，才能闭合死锁环。
目前这一条反向依赖尚未实证，不能把上面两条正常握手关系直接称为已确认死锁。

### 方向一：实际提交到了错误的流，或者夹带了跨流依赖

代码意图是复制在 load_stream，AIV wait 在另一条交接流。但 worker 的线程上下文、
Torch 当前流、某个内部算子或事件路径可能影响实际提交位置。如果拷贝被排在同一条交接流的
wait 后面，关系就是“wait 等 CPU，CPU 等 copy，copy 等 wait”，必然无法正常推进。
当前仅根据 `with stream(load_stream)` 不能证明全部内部任务都没有跨流依赖。

诊断记录交接流原始句柄、加载流句柄和 worker 当前流句柄。如果 load 与 handoff 相同，
或者 current 与 load 不同，是直接需要追查的证据；句柄不同只能排除显式同流，不能排除隐式事件依赖。

### 方向二：量化参数 Tensor 拷贝包含额外设备操作

主权重走 Storage 拷贝，量化参数走 Tensor 拷贝。参考的 torch-npu 源码中，
非连续目标或 dtype/shape 条件不匹配时，Tensor H2D 可能先构造临时连续目标，
完成 H2D 后再通过设备侧 copy/cast 写入原目标。`copy_()` 返回只说明该调用返回，
不证明它内部只有一条 DMA，也不证明全部设备操作已经执行。

若额外操作等待交接流，或因运行中 AIV wait 与真实图调度存在资源依赖，就可能形成上述反向依赖。
之前独立图中计算可与 AIV wait 重叠、合成非连续 Tensor 拷贝也曾通过，
因此不能笼统认定“AIV wait 阻止所有计算”或“非连续拷贝必定死锁”。
需确认真实源/目标的 shape、stride、dtype、storage offset 和当前版本实际路径。
参考源码不等于容器安装的 dev 二进制，内部路径仍须结合运行结果确认。

### 方向三：Host 提交队列等待，而非已经提交的设备任务等待

参考的 `NPUStream::synchronize()` 会获取 stream 句柄，再调用 ACL 同步；
句柄获取在某些提交模式下可能先等待 Host 队列排空。SyncLaunchStream 旨在减少这种依赖，
但不能仅从 Python 栈断定内部具体停在哪一行。某些子操作若转入其他流或队列，仍需核实。

因此诊断把“拿句柄”“直接 ACL 等待”“原 Torch 同步”拆开。若直接 ACL 已返回成功，
但随后的 Torch 同步不返回，应优先追查 Host 提交、句柄获取或期间新增任务，
而不是继续认定刚才那段 DMA 未完成。

### 暂不优先的解释

请求 generation 已被领取，说明该轮 NPU→CPU 通知已经到达；done 没更新是回调尚未返回的预期结果。
所以 yield、扫描范围、GIL 获取前等待不是这次已进入同步后的直接卡点。
同样，不能把未写 done 归因为 CPU→NPU 标志可见性故障，必须先证明 CPU 确实写过匹配值。

## 本轮诊断：在同一次模型运行中拆分等待阶段

新增 `--diagnose-native-wait`，自动启用原有逐项拷贝诊断。只作用于前八个真实回调批次，
关闭时不增加这些同步。记录真实 Tensor/Storage 元数据，不读取权重数值、不增加权重布局转换。

每批次先获取加载流句柄，然后对 prior work、每次 copy 和 scale refresh 分别执行：

```text
BEGIN → 原操作 → SUBMITTED
      → ACL_WAIT_BEGIN → 直接 ACL 等待，最长参数 10000 ms → ACL_WAIT_RETURN
      → TORCH_WAIT_BEGIN → 原 stream.synchronize() → COMPLETE
```

直接调用 `aclrtSynchronizeStreamWithTimeout`，其超时参数单位为毫秒、成功返回 0，
参见 [Ascend 官方接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0077.html)。
API 错误立即打印诊断并退出引擎，不发布成功、不继续推理。外部请求超时和原 watchdog 仍保留，
不能假定所有 Runtime 故障都严格按 10 秒返回。
特殊 stream 类型可能不支持正常同步语义；直接 ACL 成功只能结合实际加载流类型和后续 Torch 结果解释。

| 最后阶段 | 可缩小到哪里 |
| --- | --- |
| STREAM_HANDLE_BEGIN，没有 READY | 句柄获取或当前流获取过程，尚未进行本批次拷贝 |
| copy BEGIN，没有 SUBMITTED | `copy_()` 调用内部，可能是准备、提交或隐式等待 |
| SUBMITTED 后 ACL 等待报错/不返回 | 绕开 Torch 句柄重新获取后，指定原始流仍不能正常同步；不能单凭此确定 DMA 或内部算子 |
| ACL 返回 0，TORCH_WAIT_BEGIN 后无 COMPLETE | 原始流检查已返回，重点追查 Torch 队列或同步路径，也需考虑原始流检查未覆盖尚未提交任务 |
| 全部通过 | 本次诊断改变时序后未复现；不能宣称原阻塞消失 |

这轮并非直接追踪每条 ACL 内部调用，而是在明确位置主动调用原始流同步作区分。
诊断会改变流水与时序，所有时延都不能用于性能比较。

## 执行命令与下一步边界

```bash
python -B script/run_moe_host_sync.py --backend mte --npu 7 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --worker-scan all --worker-probe locked --worker-idle yield \
  --diagnose-native-wait --requests 1 --warmup-requests 1 \
  --request-timeout 180 --profile-timeout 120
```

只返回 summary 文本。本轮不需要完整 profiler 文件，也不要求 gdb。
运行结果首先确定 Host/设备等待分界和真实布局；若确认原始加载流等不到，下一步才用真实布局
及已识别的流依赖构造针对性隔离，不重复泛化的“非连续拷贝是否能跑”实验。
若需证明具体事件或算子闭合循环，仍需要对应 Runtime/设备任务证据，不能只凭同步超时下结论。

## 原始 ACL 等待结果与 scale 分段诊断

`moe_host_sync_633868b4` 在第二批、layer 1 的预取中再次复现。此前主权重
`w13`（8 MiB）、`w2`（4 MiB）均完成原始 ACL 和 Torch 等待。
`w13_weight_scale`（512 KiB）提交返回后，原始 ACL 等待返回非零码 `507046`，
尚未进入该任务的 Torch 等待。随后诊断主动终止引擎，HTTP 500 是后果。
因此本轮不是仅在 Torch 同步入口卡住，但仍未证明具体设备任务间的循环依赖。

失败任务的源是 pinned CPU uint8 张量，shape 为 `[64,4096,2]`，stride 为
`[8192,2,1]`；目标 shape 相同，stride 为 `[2,128,1]`，非连续。
第一批相同 shape/stride 的 scale 拷贝成功，说明布局不一致本身不足以解释阻塞，
需要结合预取图与等待任务的执行条件分析。request=1、done=0 表示工作线程已接到任务，
正在完成拷贝，不能解释为初始轮询未发现请求。

下一轮使用 `--diagnose-split-scale`，自动包含原始 ACL 等待诊断，仅对前八批真实
scale 任务显式引入连续 NPU 临时张量，保持原目标张量、目标布局及逻辑值不变：

1. `TEMP_ALLOC_BEGIN/READY`：在加载流上下文中创建临时张量。
2. `H2D_CONTIGUOUS`：CPU 原始 scale 拷贝至临时张量，分别等待 ACL 和 Torch 完成。
3. `D2D_TARGET_LAYOUT`：临时张量写入原非连续目标，再分别等待完成。
4. 两段都成功后才允许执行原 scale refresh；失败立即终止，不提前发布 done。

若 H2D 完成而 D2D 不完成，阻塞就进一步缩小到目标布局写入所产生的设备工作及其依赖，
不能再归因于 CPU→NPU 的字节搬运没有结束。若 H2D 失败，则继续检查传输侧；
若全部成功，只能说明显式分段或新增同步改变了触发条件，不能当成根因已定位或修复完成。
这轮不做 CPU 重排，也不把临时张量方案接入正常路径；它改变分配与提交时序，不能比较性能。

执行上一节同一命令，将 `--diagnose-native-wait` 替换为 `--diagnose-split-scale`。
返回 summary 文本即可，不需要 gdb。

## 分段结果：H2D 完成，设备侧布局写入未完成

2026-10-28 返回的分段实验在 layer 1、专家 139、槽位 20 再次失败。
`TEMP_ALLOC_READY`、`H2D_CONTIGUOUS COMPLETE` 均出现；随后
`D2D_TARGET_LAYOUT SUBMITTED`，原始 ACL 等待返回 `507046`。
因此此次可明确区分：数据已进入连续 NPU 临时张量，未完成的是随后写入原非连续目标的设备工作。
尚不能仅据此认定某个具体算子占满 AIV，或证明加载流与预取流之间的完整循环依赖。

下一步隔离实验 `script/probe_scale_wait_dependency.py --npu 7` 不加载模型，复用已有算子库，
使用相同 shape、stride、目标元素偏移和同类 SyncLaunchStream，在原生 Host worker 中执行 D2D。
它一次运行三个独立进程：只发布请求，以及发布后等待 100000、1000000 次轮询。
三组都捕获、重放三次；重放前清零目标，验证写入值与相邻槽位不被覆盖。
等待次数不是时间，所有结果仅用于依赖诊断，不是性能数字。

有界等待仅存在于没有模型消费者的隔离图中，正式 MoE 的 strict wait 不改。
若只发布能完成，而 D2D 要等有界等待退出后才完成，且其等待随轮询预算增长，
将支持等待算子阻碍 D2D 推进的解释；仍需结合结果排除首次调用等因素。
若三组都正常，就说明独立图未复现，下一步应关注真实预取图依赖，而不是宣称问题不存在。
脚本不写文件，输出可直接复制。推理入口的 summary 同时改为只保留末尾关键任务、
最近布局、交接状态和首个错误附近信息；完整调用栈与逐项日志仍在 inference_python.log。

## 独立 D2D 与等待并行实验结果及图内依赖对照

2026-10-28 的三个独立进程均通过。无等待时 D2D 的 ACL 等待约 8.7～21.4 μs；
加入两种轮询预算时约 9.7～12.8 μs。两组等待的 state[1] 都为 1，实际轮询
204～879 次就收到完成信号，远小于 100000/1000000 的预算，没有靠等待超限解开阻塞。
这些是诊断时序，不作为性能验收。结果不支持“一个 AIV 等待算子就必然阻碍这类 D2D”
的解释，也没有证明真实模型存在死锁。死锁一直只是待验证假设。

此前实验未覆盖真实预取的图内分叉和汇合：
`trigger_next_layer_prefetch` 在计算流记录 ready 事件；`_dispatch_prefetch` 在预取流
等待 ready，提交交接，写回映射，再记录 load_done；后续 `update_weights` 在计算流等待
load_done。加载流仍是图外 worker 使用的 SyncLaunchStream。

脚本下一轮一次比较四组：独立图+100000 次预算、分叉汇合图不加等待、分叉汇合图
分别加 100000/1000000 次预算。新增图在计算流捕获，预取流等待 ready 后执行
publish/wait，再记录 done，计算流等待 done。所有事件在重放期间保留。
暂不加入模型计算、路由与映射拷贝，目的是单独判断这组事件关系是否足以触发问题。
该结构不是宣称与真实模型完全等价。若它也通过，就排除简单分叉汇合足以复现的解释，
继续需要真实图中的额外条件或设备任务证据；不能把通过当作原问题已解决。

命令仍为 `python -B script/probe_scale_wait_dependency.py --npu 0`，输出只在控制台。

## 分叉汇合实验通过，返回真实模型取证

后续四组全部通过，有界等待均以 state[1]=1 正常退出，实际轮询 226～1074 次，
没有耗尽预算。由此简单的分叉汇合关系也不足以复现真实阻塞；不能把事件等待直接认定为死锁。
不再继续扩展已通过的合成图，转回真实模型的第二批 scale 拷贝位置。

新增 `--diagnose-scale-context` 自动包含前述分段诊断，H2D 完成后按顺序检查：

1. `D2D_CONTIGUOUS`：写入独立连续 NPU 目标。
2. `D2D_INDEPENDENT_STRIDED`：写入独立、相同 stride 的 NPU 目标。
3. `D2D_TARGET_LAYOUT`：写入原模型的目标张量。

前两项只写临时目标，不改变模型权重；第三项仍执行原目标写入。每项失败立即停止，
不发布成功。独立非连续目标使用 preserve_format，并检查 stride 与原目标一致；
它的 storage_offset/底层分配不同，因此只用于区分布局操作与原目标关联条件，不能宣称完全等价。
这轮改变了分配和时序，若全部通过，应记录为触发条件变化，而不是问题修复。

同时记录临时张量、原目标、探测目标的 NPU format、stride 和 CPU 线程 grad/inference 状态。
ACL 等待失败后，立刻在同一 worker 线程调用 aclGetRecentErrMsg 获取 Runtime 错误说明，
再打印 Python 栈并终止。该接口默认读取线程级错误，并会清除所读消息，因此结果立即保存到
服务日志；不存在消息时明确打印空结果，不凭错误码推断具体任务。
接口依据：[AscendCL aclGetRecentErrMsg 文档](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0028.html)。

执行：

```bash
python -B script/run_moe_host_sync.py --backend mte --npu 0 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --worker-scan all --worker-probe locked --worker-idle yield \
  --diagnose-scale-context --requests 1 --warmup-requests 1 \
  --request-timeout 180 --profile-timeout 120
```

返回精简的 summary.txt 内容。完整错误信息保留在 inference_python.log；summary 单行最多
2000 字符，并标记截断，避免把大段 Runtime 信息再次变成难以复制的输出。

## 真实上下文中连续目标也失败：转向 Runtime 任务证据

实验 `moe_host_sync_a6a6f2c4` 中，第二批 layer 1 的 H2D 成功，随后新增的
`D2D_CONTIGUOUS` 已提交，但 ACL 等待返回 507046。因此尚未执行本批独立非连续目标，
更没有执行本批原目标写入。摘要前面的 `[32,4096,2] INDEPENDENT_STRIDED` 来自此前任务，
不能误读为第二批的非连续探测已通过。

这进一步排除了“必须写入原 scale 槽位或非连续 stride 才会触发”这一解释。
临时、原始和连续探测目标的 npu_format 都是 2；worker 的 grad_enabled=true、
inference_mode=false 只是上下文记录，不能据此认定是原因。

Runtime 明确报告 `Execution_Error_Stream_Synchronize_Timeout(EE1002)`，未完成任务类型
`KERNEL_AIVEC`，进程 56880，device_id=7，stream_id=27，task_id=773。
这属于实际失败点的任务类别证据，不等于已经知道 kernel 名称、具体依赖或死锁。
参考 Torch-NPU CopyKernelOpApi 源码的 D2D 基础格式路径会调用 aclnnInplaceCopy，
因此连续 Tensor.copy_ 也不能直接等同于纯 DMA；参考源码不保证与绿区开发版二进制完全一致，
本次以 Runtime 的 KERNEL_AIVEC 报告为准。

下一步先读取已有进程日志，不重启模型：

```bash
python -B script/inspect_mte_runtime_task.py bench_results/moe_host_sync_a6a6f2c4
```

脚本从服务日志提取失败 PID/stream/task，搜索当前用户 ascend/log、/var/log/npu/slog、
/usr/slog 下以该 PID 命名的未压缩日志，读取每个文件末尾最多 16 MiB，最多读取 40 个文件，
最多输出 70 行任务线索。可用 `--log-root 路径` 指定实际目录。
日志没有开启、发生轮转、文件命名不含 PID 或信息位于读取范围外时，可能无结果；
不能把无结果当作不存在对应任务。附近任务或关键词行只是线索，不能直接认定为因果链。

## Runtime 超时快照与流身份核对

上述进程的 plog 在同一超时快照中列出：流 20/task 2、流 22/task 64、
流 27/task 773 是 KERNEL_AIVEC；流 23、49、55、57、59 是 EVENT_WAIT；
流 61/task 1358 是 NOTIFY_WAIT，NotifyId=704。
这不是 kernel 名称表，也不证明这些任务同时占据执行核。事件等待可能只是上游停顿的后果，
不能把多个 WAIT 直接解释为死锁，更不能按相邻日志行建立依赖边。

为减少多轮交互，原诊断入口现在同时记录：

- `STREAM_MAP`：通过 aclrtStreamGetId 查询加载流和捕获交接流的 Runtime ID，
  同时保留句柄、代表槽位和回调名称。查询接口不可用时明确标记，不猜编号。
- `PENDING`：超时快照中 request_generation 大于 done_generation 的所有交接槽位，
  包含流句柄、回调类别和层号。它是 CPU 依次读取的状态快照，并非设备原子快照。
- 失败后自动提取同一 PID 的 Runtime 日志，无需再单独运行提取命令。

接口依据：[aclrtStreamGetId](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1038.html)。
图执行可能使用内部调度流，若捕获时交接流 ID 与超时快照不一致，不能强行配对，
仍需图任务映射证据。PENDING 用于区分一个已发布任务未完成与多个已发布任务积压，
不能揭示尚未执行到 publish 的图节点。

继续使用上一节 `--diagnose-scale-context` 的完整模型命令，返回同一份 summary 即可。
本次不改变交接、拷贝或等待策略，仅补充可核对的身份信息和自动日志提取。

## 流映射结果与预取等待方式对照

实验 `moe_host_sync_3250358c` 确认加载流 Runtime ID=27。按需交接捕获流 ID=47，
预取交接捕获流 ID=56；超时快照的其他 AIVEC 位于 21、23，仍不能直接将它们配对为自定义等待算子。
PENDING 中只有 slot 2（layer 1 的 `_prefetch_host_cb`），request=1、done=0。
因此该快照没有显示多个已发布交接请求积压；未发布的图任务不在这项观测范围内。
独立连续 D2D 仍未完成，H2D 仍已完成。

下一轮使用 `--diagnose-prefetch-report-wait` 做等待方式对照：

1. 预取流保留原来的 MTE publish，继续由设备写入 request_generation。
2. 原生轮询 worker 仍执行 `_prefetch_host_cb` 和原始权重拷贝，完成 H2D 同步后发布 done。
3. 仅预取流不再提交 AIV wait，而注册阻塞型 Host 回调，由报告线程调用 Session.wait
   等待同一槽位本轮 done。Session.wait 在原生代码中释放 GIL，并在等待期间释放 mutex，
   不占住 worker 执行所需的 GIL 或锁。
4. 按需交接仍使用原来的 MTE publish + AIV wait。任何完成等待失败都会终止引擎，
   不提前写 done，不让图消费未完成权重。

这不是最终优化方案，也不是将预取计算整体退回旧回调线程。它只改变预取等待机制，
仍会改变设备调度和 Host 时序。若通过，支持预取 AIV 等待与真实图组合参与触发问题；
仍不足以断言具体调度器原因或死锁。若仍失败，该等待方式就不是唯一触发条件。

本轮关闭临时 scale 分段/目标探测，回到原始 Tensor 拷贝；保留逐项原始 ACL 等待诊断：

```bash
python -B script/run_moe_host_sync.py --backend mte --npu 0 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --worker-scan all --worker-probe locked --worker-idle yield \
  --diagnose-native-wait --diagnose-prefetch-report-wait \
  --requests 1 --warmup-requests 1 --request-timeout 180 --profile-timeout 120
```

不要同时加入 `--diagnose-scale-context` 或 `--diagnose-split-scale`。返回 summary.txt 即可，
失败时自动附加 Runtime 日志。正常配置默认关闭此开关，仍使用原有 MTE 等待方式。

## 预取报告线程等待对照完成推理

实验 `moe_host_sync_d0b14946` 明确打印 `prefetch=report_wait; request=MTE;
worker=native; reactive=AIV`，预热和一条测量请求完成，输出 5 个 token，
摘要为 `21e9f347a4cd40b1d4c2263761c4737c6543bdc4027b2c3ade06ce19f176621a`。
原始 scale Tensor 拷贝和后续 scale_refresh 均继续执行，诊断已推进到第八批。
请求耗时 3272.343 ms，但本轮不是性能对照，不能据此宣称提速。

与此前原生 worker + 预取 AIV 等待反复在第二批失败相比，本轮提供了较强的触发条件证据：
保留原生 worker、原始权重布局和拷贝时，更换预取等待机制即可让本次推理完成。
这说明 CPU 重排并非完成原始拷贝的必需条件，也不支持仅因 SyncLaunchStream 或
原生 worker 存在就必然阻塞的解释。但跨轮实验不是相同设备负载下的严格重复对照，
且等待实现改变了调度与时序；具体 Runtime 调度、资源或依赖原因尚未证实。
不能把报告线程等待称为最终的纯轮询优化或已根治所有输入。

同时本轮 profile 不合格：日志出现 `External init callback must run in same thread as
registerClient`、`CANNTimelineParser run failed`，MSTX 标记数为 0，间隔分析也失败。
已有请求完成不因此被抹去，但不能用此 profile 分析空泡或证明交接收益；后续还出现
EngineDeadError，服务完整生命周期也不能标为通过。

修正结果判定：本入口在读取完整服务日志后，将明确解析失败标记为 profile_status=invalid，
只有缺少预期 MSTX 标记而没有明确解析错误时标记为 unverified，同时单独记录
engine_error_seen。旧 summary 中的 complete 是错误的有效性结论，不能继续沿用。
本次只修正结果记录，不宣称解决 profiler 的线程初始化或解析问题。

## 撤掉逐项诊断同步后的连续请求结果

实验 `moe_host_sync_0e2f336b` 只保留预取报告线程等待开关，关闭 native wait、scale 分段
和目标探测。一次预热后，三条测量请求全部完成：510 次卸载回调，errors=0、dropped=0，
其中预取 252 次、按需 258 次。三次输出摘要一致，均为上一节记录的摘要，每条输出 5 个 token。
因此本次通过不依赖前八批逐项插入的诊断同步，也不依赖 CPU 重排或额外的目标布局适配。
覆盖仍然只有同一个短提示词，不能据此宣布所有形状和负载均正确。

请求均值 3412.143 ms，范围 3405.121～3423.089 ms。预取的 load_submit 均值
208.407 μs、synchronize 129.443 μs，二者合计 337.850 μs；按需分别为
834.358 μs、709.979 μs，合计 1544.337 μs。它们是本次回调区间统计，
不是 NPU MEMCPY 空泡，也不包含独立报告线程等待回调的整套交接成本。
没有本轮相同条件的原始 callback 对照，不能由这些数字认定加速或退化。

两轮等待对照共同支持：预取 AIV 等待和真实图运行的组合参与触发原始拷贝阻塞。
已能定位到修改类别，但还没有证明底层的核资源占用、图调度约束或具体依赖环。
从实现上，strict wait 是单 block 的 AIV kernel，并非一次启动占满所有核；
不能仅凭存在这个 kernel 就声称它占满 AIV 或必然导致死锁。
后续应围绕真实图的任务执行/调度证据调查，而不再重复已通过的布局与简单分叉汇合单测。

profile 仍为 invalid：同线程初始化错误、CANNTimelineParser/RelationParser/CANNAnalyzeParser
失败、MSTX 标记数为 0，engine_error_seen=true。请求结果成功和采集阶段故障要分开记录；
目前不能确定 EngineDeadError 是 profiler 故障引起，不能把两者时间上的相邻当作已证实因果。
报告线程等待保留为诊断对照，不启用为默认优化，更不能称为已经完成纯轮询方案。

## AIV 退出前后完成状态实验

仅替换报告线程等待已经得到触发条件线索，但没有解释为什么。下一轮保留真实预取 AIV
等待，设置有限轮询预算，并紧接一个报告线程完成守卫：AIV 退出不代表权重就绪，
后面的阻塞回调仍等待本槽本轮 done，成功前不允许图继续消费权重。
默认纯 MTE 路径不变，按需仍使用 strict AIV wait；有限预算只用于这项诊断。

使用 `--diagnose-prefetch-poll-budget 1000000` 自动启用报告线程守卫。记录前八次预取：

- `AIV_EXIT`：报告回调开始时，AIV 已退出；读取 request、done、error 及正在执行的 worker。
- `AFTER_AIV_EXIT`：原生 worker 真正完成，记录完成代数及报告回调守卫耗时。

如果 AIV 退出后仍未 done，而 worker 随后完成，结合此前 strict wait 超过十秒仍不完成，
将提供等待驻留与拷贝推进关系的更强证据。但单次预算耗尽也可能只是工作量超出预算，
还需结合原始 ACL 拷贝阶段记录，不能单凭 done_at_report_entry=0 宣布资源死锁。
报告回调守卫耗时是 Host 测量，包含调度和打印，不是精确设备空泡时间。
若守卫也超时，则该实验没有解除触发条件，进程仍会退出，不伪造完成信号。

```bash
python -B script/run_moe_host_sync.py --backend mte --npu 0 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --worker-scan all --worker-probe locked --worker-idle yield \
  --diagnose-native-wait --diagnose-prefetch-poll-budget 1000000 \
  --requests 1 --warmup-requests 1 --request-timeout 180 --profile-timeout 120
```

本轮只判读交接和拷贝推进证据，不依赖已知存在问题的 profiler 导出。
正常模型配置不启用此选项；禁止将有界等待移除后续完成守卫单独用于推理。

## 百万次预算实验结果与设备退出原因取证

`moe_host_sync_b5f92e1b` 完成请求，输出摘要与先前一致，但耗时 49.522 秒。
前八次预取中，槽位 2、4、6、8、10 在报告回调开始时仍为 done=0，原生 worker 正在
处理相应槽位；随后分别在 225.438、175.042、184.827、205.568、100.811 μs 的
报告守卫区间内完成。槽位 12、14、16 在报告回调开始时已经完成。
这些证据支持 AIV 驻留与真实拷贝推进的相互影响，但报告回调起点晚于设备退出，
不能把 guard 耗时直接当作 AIV 退出到拷贝结束的精确时延。
49.522 秒是人为有界轮询诊断的整请求时间，不能称为优化结果；轮询预算对它的贡献
尚需对照，不能直接将全部增加时间算成等待 kernel 的耗时。

下一轮预算改为 100000。为直接区分退出原因，将每个 bounded wait 的 32 字节 state
在同一预取流回传到专用 pinned CPU 缓冲区，然后才执行报告回调。
回调只读取 CPU 数据，不额外发起 NPU 同步。device_exit 的四个数分别为
`[generation, exit_code, observed_done, poll_count]`：code=1 是观察到完成；code=3 是
耗尽轮询预算。错误或代数不一致会终止引擎，不作为成功放行。
每 32 次诊断预取打印累计完成退出/预算退出次数，覆盖不止最初八个样本。
这个新增 D2H 会改变少量时序，因此不能将两轮耗时差全部归因于预算；重点是退出类型
及退出后真实 worker 的完成证据。正式纯 MTE 路径不增加此回传。

使用同一命令，将 `--diagnose-prefetch-poll-budget 1000000` 改为
`--diagnose-prefetch-poll-budget 100000`。只返回 summary.txt，不依赖 profile。

## 十万次预算结果与 Runtime 任务级定位

`moe_host_sync_cbeedfd2` 请求完成，耗时 8.506 秒。前八次预取的设备退出记录均为
`[1,3,0,100000]`，确认退出原因是预算耗尽，而不是已观察到完成。
其中五次在报告回调开始时仍未完成，随后在 57.396～147.130 μs 内完成；其余三次
在设备退出与报告回调开始之间完成。最后一次累计打印为正常完成退出 4 次、预算退出
156 次，共 160 次，包含预热，不是整个实验最终总数。

这进一步支持“预取 AIV 等待驻留与真实模型内拷贝推进发生相互影响”的判断，
尚不能区分具体资源竞争、图调度依赖或可见性问题。停止继续调整轮询预算。

下一次回到纯 MTE 无限等待、原始 Tensor 拷贝，通过
`bash script/diagnose_mte_runtime.sh 0` 一次执行构建、复现及日志摘录。
只对子进程设置 CANN 官方 `ASCEND_MODULE_LOG_LEVEL=RUNTIME=0`，不修改系统日志配置。
不启用布局重排、有界退出或报告线程替代。日志会扰动时间，本轮不用于性能比较。

原生 worker 在拷贝提交后、同步前调用 `aclrtGetThreadLastTaskId`，记录调用线程的
最后提交任务编号，再与超时中的 stream/task 以及 Runtime DEBUG 下发记录交叉核对。
这个编号不是完成证据，也不保证包含算子内部全部任务；接口不可用会明确记录。
日志提取先扫描受限范围内的全部候选文件，再优先打印 stream/task 同时匹配的记录，
避免普通错误行挤占下发记录。搜索仍受文件数量、尾部大小及日志轮转限制。

本轮目标是给超时任务补充具体下发操作与 kernel 线索；即使匹配成功，也不能仅凭
下发日志认定任务实际占用了哪些 AIV 核，或直接认定死锁根因。详细日志保留绿区，
只需返回经检查脱敏的 summary.txt。

## Runtime DEBUG 已确认原始 scale 拷贝对应超时 AIVEC 任务

`moe_host_sync_8e2b8992` 在第二次交接、layer=1 的预取重现阻塞。
本次 w13、w2 原始拷贝均同步成功；w13_weight_scale 原始拷贝提交返回后同步超时。
查询返回编号 74901，Runtime 分配和下发日志明确对应
`stream_id=27, task_id=711, task_sn=74901, KERNEL_AIVEC`，与超时任务一致。
同一 CPU 线程 66757 的附近下发日志显示 `blockDim=56`。
这将阻塞定位到原始 scale.copy_ 提交的设备任务，而不是 Python 尚未提交任务。
`DavidSendTask drvRet=0` 只证明下发接口成功，不能判断该 kernel 尚未调度还是执行中停滞。

本次日志同时证明 task_id=711 在同一进程、同一流中循环复用：更早记录对应
task_sn=71843 的 MEMCPY_ASYNC。因此不能只凭 stream/task 数字把早期 DMA 与当前
AIVEC 当成同一个任务。新增只读脚本按 task_sn=最后提交序号提取上下文，并从设备函数
地址查找注册线索，不启动模型、不写文件。blockDim=56 是下发参数，不能直接等同于
同时占用 56 个物理核，也尚不能据此断言等待算子与拷贝算子构成资源死锁。

```bash
python -B script/inspect_mte_task_serial.py bench_results/moe_host_sync_8e2b8992
```

本轮要补全具体 kernel 身份及调度参数，沿用已有 DEBUG 日志，无需重新进行推理实验。

## 已确认转置任务身份；用加载流分核检验资源需求假设

后续只读提取明确显示：task_sn=74900 是 524288 字节 H2D，Runtime 已记录
finishTaskId_=710 并回收该 DMA；task_sn=74901 则是
`Transpose_int8_int64_high_performance_10001`，coreDim=56、schemMode=0，随后超时。
这是原始 copy_ 内部触发的设备转置，不是后来增加的 CPU 重排。
同一函数地址、同样 coreDim 的转置在前一个交接中曾反复提交，前一交接已完成。
因此，仅转置算子名称或 56 个 block 本身都不足以解释为什么第二次交接阻塞。

之前已知的“scale 拷贝后超时”不能重复算作根因定位进展。本次新增的是具体转置名称
以及原始 copy_ 内 H2D 已完成的直接日志证据；尚缺的是转置为何不完成的解释。

下一次只改变原生 worker 加载流的 Vector Core 上限为 28，仍使用原始 Tensor
拷贝、SyncLaunchStream、预取严格 AIV 等待。默认上限配置为 0，不调用分核接口。
诊断选项为 `--diagnose-load-vector-cores 28`，设置流资源后绑定到当前原生 worker，
不设置整个进程的资源上限，不改变模型图和等待算子的下发代码。
接口失败会显式报错。API 成功还不足以证明算子采用新上限，必须核对日志中实际 coreDim。

依据官方接口说明：
[aclrtSetStreamResLimit](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1947.html)
设置流资源上限，再用
[aclrtUseStreamResInCurrentThread](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1949.html)
在当前线程使用该设置。这不是将某些物理核独占分配给该流。

一次执行并汇总实际加载流 kernel 核数：

```bash
python -B script/diagnose_mte_vector_limit.py --npu 0
```

判定标准：实际核数仍是 56，则本轮干预未得到验证；核数降低且仍阻塞，则降低需求
不足以解除阻塞；核数降低且原始转置/请求完成，则说明分核或其导致的 tiling 改变
影响了阻塞，下一步才区分这两者。任何结果均不能直接证明具体硬件死锁。
保留原始输入与配置，结果不用于性能对比；只返回汇总文本。
