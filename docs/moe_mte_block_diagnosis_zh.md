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
