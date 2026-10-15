# MoE 卸载的 CPU/NPU 流程：从入图到 host callback

本文面向只有 LLM 基础的读者，说明本项目如何在 CPU 内存与 NPU 显存之间管理专家权重，
重点解释 `_launch_host_func`、`_update_weights` 和图捕获/重放。
以当前 `script/test_graph2.sh` 的单卡、LRC、TorchCopy 路径为主，核对日期为 2026-10-15。
本文解释已有实现，不新增同步优化，也不把待验证的性能猜测当作结论。

## 1. 先认识参与工作的对象

| 概念 | 含义及在本项目中的作用 |
| --- | --- |
| CPU / Host | 服务器处理器，执行 Python、LRC C++ 扩展、缓存规划和运行时控制逻辑 |
| NPU / Device | 昇腾加速卡，执行 Attention、专家矩阵计算及相关设备算子 |
| CPU 内存 | 保存卸载的专家权重；不是磁盘，逐层加载不等于逐层从模型文件读取 |
| NPU 显存 | 保存计算所需的数据和部分专家权重；容量有限，需要替换 |
| MoE 专家 | Transformer 中一组可选择的前馈网络；路由器为每个 token 选择若干专家 |
| 路由器 / gate | 根据输入等规则决定专家 ID 和权重；并不是把所有专家都计算一次 |
| 槽位 slot | NPU 上容纳一个专家权重的位置；替换专家时复用这个位置 |
| `log2phy` | 逻辑专家 ID 到物理槽位的映射；本路径以负值表示未驻留 |
| H2D / D2H | Host→Device / Device→Host，即 CPU→NPU / NPU→CPU 数据传输 |
| kernel | NPU 执行的设备任务，例如矩阵乘法；CPU 回调不是 NPU kernel |
| Stream | 设备任务的逻辑队列；同一流按依赖顺序执行，不同流可能重叠，但不保证并行 |
| Event | 记录流执行进度的事件；另一条流可以等待这个事件，建立跨流依赖 |
| CANN / ACL | 昇腾软件栈 / 其中的应用编程接口；负责设备任务、内存和流等管理 |
| torch_npu | PyTorch 对昇腾的适配层，本项目通过它提交设备任务及注册 CPU 回调 |

典型 Transformer 层先进行 Attention，再进行前馈网络计算。
MoE 层的前馈网络包含路由与专家计算，也可能有共享专家等分支。
不是“一个 token 只发生一次 MoE”：一个 token 在一次前向中会经过多个 MoE 层。

## 2. Prefill、Decode、MTP 与预热

### 2.1 Prefill：处理输入上下文

用户输入一段 prompt 后，模型需要处理其 token，计算中间状态并建立 KV cache。
这个阶段叫 Prefill，通常一次处理多个 token，也可能被调度器切成多个块。
前缀缓存命中时，已缓存部分的计算可能被复用，不能假设每次相同 prompt 都完整重算。

### 2.2 Decode：继续生成

模型利用已有 KV cache 继续生成输出。普通自回归生成通常每个序列每步推进一个新 token，
但批处理、MTP 等机制会改变一次前向处理的行数。

当前脚本配置 `num_speculative_tokens=2`，启用了 MTP 推测解码。
一次验证可能涉及多个候选位置；接受多少候选决定这一轮实际推进多少输出 token。
因此 `rows=3` 不能简单解释成三个独立用户请求，也不能由 `max_tokens=5` 推断恰好调用五次回调。

本项目专家管理器还有一个实现层面的分支：

```python
offload_threshold = num_device_experts // topk
```

当传入路由的行数超过此阈值时，走称为 prefill 的池化/整层覆盖加载路径；较小时走逐专家分页路径。
这是根据行数选择实现，不完全等于请求语义上的 Prefill/Decode 分类。
很短的 prompt 也可能走逐专家分页路径。本文后面主要解释这一分页路径。

### 2.3 预热不是预取

- 预热：先跑请求，让模型、图和缓存等进入可测试状态。测试脚本默认每组预热 3 个请求。
- 预取：预测未来要用的专家，提前搬运它们的权重，是每次模型执行中的功能。
- profiling：记录某个时间窗口内的执行行为，不是模型生成过程本身。

当前一键脚本先预热，再开启 profiler 和 MSTX 窗口，所以那三个预热请求不在正式 profile 中。

## 3. 为什么需要卸载与缓存

假设模型有很多专家，但某层一次只选少量专家。把所有专家常驻 NPU 可能放不下，
因此让部分专家留在 CPU 内存，只把需要的专家放入 NPU 槽位。

每次路由产生结果后，需要回答：

1. 这次会使用哪些专家？
2. 哪些已经在 NPU 上，哪些缺失？
3. 缺失专家放到哪些槽位，哪些旧专家需要被替换？
4. 什么时候权重与槽位映射都准备好，可以开始专家计算？

LRC 参与访问统计和替换决策：`observe` 更新真实访问统计，`choose_victims` 选择替换对象。
当前已验证的 C++ 模式迁移了这部分，其他规划、日志、加载提交等仍保留原实现。
卸载通常节省显存，但会增加传输、CPU 管理和依赖等待，性能目标是减少这些额外开销。

## 4. “入图”到底是什么意思

### 4.1 不入图：每轮由 CPU 组织并提交操作

普通即时执行（Eager）可以简化为：

```text
CPU 执行模型代码 → 提交 Attention → 提交路由 → 管理权重 → 提交 MoE → ……
下一轮再次执行这些 CPU 代码并提交操作
```

CPU 提交操作与 NPU 完成操作不是同一时刻。CPU 往往只是把工作排入队列，
NPU 稍后执行。因此一个 Python 调用很快返回，不代表对应设备任务已经结束。

### 4.2 入图：捕获可重复执行的任务及依赖

模型每一轮往往执行相似的算子序列。如果每轮都由 Python 逐项提交，调度成本可能明显。
图捕获（Capture）是在受控环境下运行这条路径，将可捕获的设备任务及依赖记录为执行图。
之后图重放（Replay）让运行时重复执行这张图，减少重复的逐算子提交。

可以类比为：第一次排练并记录一套工序，以后发出“再执行这套工序”的命令。
记录的是工作结构和依赖，不是保存上一次输出然后直接返回。

本仓库 `vllm_ascend/compilation/acl_graph.py` 的核心结构是：

```python
# 捕获阶段：执行 Python 前向，记录设备任务。
with torch.npu.graph(aclgraph, pool=...):
    output = self.runnable(*args, **kwargs)

# 后续匹配到这张图时，重放已记录的任务。
entry.aclgraph.replay()
```

重放时输入内容可以改变，但需要满足相应的形状、缓冲区地址和执行结构等约束。
框架通常维护可复用的输入/输出缓冲区，并为不同规模选择不同图或回退路径。
不能把任意动态 Python 分支自动变成图中可变化的逻辑。

### 4.3 入图不是把 Python 编译成 NPU 算子

下面这段代码中的列表、集合操作不会因为放在捕获区域里，就自动成为 NPU kernel：

```python
needed = set(expert_ids)
missing = needed - resident
```

普通 Python 在捕获时执行；重放不会自动重新跑完整 Python 前向。
如果每轮路由不同，需要每轮重新执行 CPU 缓存决策，就必须为图安排一种运行时调用 CPU 的机制。
这正是本项目使用 host callback 的原因。

本项目捕获的也不是一条固定的“永远加载专家 7”的完整卸载计划：
图中保留回调入口，而实际加载哪些专家，在回调执行时根据最新路由决定。

### 4.4 `_launch_host_func` 怎样把 CPU 工作接入图

当前单卡路径的核心代码是：

```python
if _EXTRA_CTX.capturing:
    torch_npu.npu._launch_host_func(
        current_compute_stream,
        self._update_weights_guarded,
        args,
    )
else:
    self._update_weights_guarded(args)
```

- 捕获时：把回调安排在流中的相应位置，使图重放时能够再次触发它。
- 重放时：运行时到达回调节点，CPU 回调线程执行 `_update_weights_guarded`，内部调用 `_update_weights`。
- Eager 时：本代码直接调用 guarded 函数，不走上面的回调注册分支。

这里的 `args` 指向路由暂存缓冲区、层对象、映射等；每轮通过先前的传输更新缓冲区内容。
重放不重新运行外层 Python 分支，也不需要把 `capturing` 每轮重新设置为真才有回调。

“回调在图里”表示图在这个位置触发 CPU 工作，**不表示这段 Python/C++ 在 NPU 上执行**。
C++ LRC 同样在 CPU 上运行，其 Python 包装与回调调度仍然存在。

### 4.5 图模式名称、预热与 profile 是不同概念

当前脚本是 `FULL_DECODE_ONLY`，捕获规模配置为 `[1, 2, 3]`。
它表达的是 Decode 使用完整图的运行策略与候选捕获规模，不代表预取三层或生成三个 token。
实际是否走某张图，还取决于运行时调度与形状等条件。
vLLM 配置沿用 `cudagraph` 名称，在这里实际使用的是 Ascend ACL 图，不表示运行在 NVIDIA GPU 上。

图捕获常发生在启动/预热阶段；profiler 的 start/stop 控制采集窗口。
入图、预热、采集是三件不同的事。图重放期间仍可采集 CPU 回调和设备任务。

## 5. 当前层按需加载：CPU 与 NPU 如何接力

下图是主要依赖关系，箭头不表示两个设备所有工作都串行：

```text
NPU 计算流：真实路由 → 路由/映射 D2H → host callback 节点 → 映射 H2D → MoE 计算
                                            │                  ↑
CPU 回调线程：                              开始                │
                                            ↓                  │
                         observe → 缓存规划 → choose_victims    │
                                            ↓                  │
                                      load_submit              │
                                            │                  │
NPU 加载流：                                └→ 权重 H2D         │
                                                ↓              │
CPU 回调线程：                     synchronize 等加载流完成 → 返回
```

具体步骤：

1. NPU 产生当前层实际专家 ID，以及相关路由数据。
2. `update_weights` 外层入口把需要的路由和映射暂存到 CPU 缓冲区。
3. 图中的回调到达可执行位置，CPU 进入 `_update_weights_guarded` 和 `_update_weights`。
4. `observe` 更新统计；Python 整理驻留、缺失专家以及槽位信息，并执行统计/日志。
5. `choose_victims` 选择被替换专家；外围代码构建 `(slot, eid)` 加载名单并更新 CPU 映射。
6. `load_submit` 准备拷贝任务、提交权重与量化参数拷贝，并刷新相关 scale。
7. CPU 还可能处理统计；随后 `_synchronize_h2d()` 等待加载流完成。
8. 回调返回后，计算流继续把新映射写回 NPU，后续 MoE 才能正确使用槽位中的权重。

`update_weights` 和 `_update_weights` 名字相近但职责不同：前者负责设备数据暂存、
依赖安排、注册回调及映射写回；后者是回调内部的 CPU 权重管理主体。
目前 MSTX 的 `update_weights` 标记对应后者 `_update_weights`，不是整个外层流程。

`_launch_host_func` 的 Python API 提交耗时、回调函数执行时间、设备等待回调的时间，
是不同指标，不能用其中一个直接代表另外两个。

## 6. 三种“同步”不要混为一谈

| 位置 | 谁在等谁 | 本项目的意义 |
| --- | --- | --- |
| 回调前后流依赖 | 流的后续任务等待回调满足执行约束、完成工作 | 不能在最新映射或权重尚未就绪时执行专家计算 |
| `_synchronize_h2d()` | CPU 回调线程等待加载流 | TorchCopy 后端调用 `load_stream.synchronize()` |
| `wait_event(...)` | 一条设备流等待另一条流的事件 | 当前层使用权重前，等待预取流完成相关工作 |

`wait_event` 表达设备依赖，不等于调用它的 Python 线程必须现场等到事件完成。
`stream.synchronize()` 则是主机侧等待。异步提交不意味着没有任何依赖等待，
多个 Stream 也不意味着回调可以与所有计算自由重叠。

CANN 的公开 HostFunc 文档说明异步提交与阻塞本流后续任务可以同时成立，
也列出了回调中可调用 API 的限制。见
[官方 HostFunc 接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1959.html)。
但不能因此认定本环境的 `torch_npu.npu._launch_host_func` 私有接口就是该 C API 的简单直通包装：
本项目还显式使用 `_subscribe_report`，实际绑定实现需结合绿区 torch_npu/CANN 版本核对。
当前代码确实在回调内提交其他流的拷贝并同步；后续若更换底层同步方法，
必须确认接口限制和跨流依赖，不能直接把现有回调原样塞进另一套 C API。

## 7. 预取：提前做可能会需要的加载

### 7.1 目标和限制

当前层按需加载（reactive）解决“现在缺什么”；预取（prefetch）解决“以后可能需要什么”。
理想情况是把未来层的加载提前，与当前计算重叠，缩短未来层真实需要权重时的等待。
如果预测错、预取覆盖不足、触发太晚或引入额外等待，收益会变小，甚至可能变慢。

当前启发式路径用下一 MoE 层的 gate 权重，对当前可用 hidden states 做预测；
hash 路由层则根据 token ID 查表。预测不等于下一层实际 Attention 后一定得到同样的路由。
可选训练型预测器是另一条入口，不应与当前未显式启用的配置混用分析。

`expert_prefetch_num=1` 在当前实现中限制一次目标层预取最多加载一个缺失专家，
不是提前一层的配置，也不是只处理一个 token 的配置。
目标为下一 MoE 层来自 `layer_idx + 1`；预测处理多少路由行由 `expert_prefetch_tokens`
与实际输入行数等共同限制。

### 7.2 预取路径的 CPU/NPU 接力

```text
NPU：预测下一层专家 → 预测 ID/分数和映射回传 CPU → 就绪事件
                                                       ↓
预取流：等待就绪 → host callback → 新映射写回 → 记录预取完成事件
                       ↓
CPU：_update_weights(is_prefetch=True)
     不 observe → 筛选预测缺失专家 → 选择替换槽位 → 提交 H2D → 等完成

目标层计算流：准备使用该层权重 → 等预取完成事件 → 按真实路由补载 → MoE
```

预取回调不调用 `observe`，避免把猜测当成真实访问历史。
它仍会选择替换对象、加载权重并等待 H2D；并不是只生成一份专家名单。
即便预取成功，目标层的 reactive 仍需检查真实路由，并加载尚未覆盖的专家。

### 7.3 预取为什么可能出现在 Attention 后

不要只看注释判断触发位置。当前实际代码存在差别：

- `w4a8_mxfp4.py`：`trigger_next_layer_prefetch(layer, x)` 位于 `fused_experts(...)` 之前，
  尽管附近注释写着 GMM 提交之后。
- `w4a8.py`：调用位于 `fused_experts(...)` 之后。

实际走哪条量化路径要结合运行环境确认。第一种路径可以在当前层 Attention/路由之后、
当前层专家计算之前发起下一层预取。MSTX 预取块的 `layer` 是加载目标层索引，
不能将旁边 Attention 的层号直接套给它；索引来自 `moe_layers`，也不要擅自改成从 1 开始的编号。

## 8. 一个完整例子

以下编号和数量纯属说明，不是实测模型配置。假设下一 MoE 层 L 有 8 个专家、4 个槽位：

```text
槽位：     0  1  2  3
驻留专家： 0  2  4  6
```

1. 在更早的层，预测器认为 L 可能使用专家 5。预取额度是 1。
2. CPU 预取回调发现 5 不在 NPU，按策略选择专家 0 的槽位。
3. 把专家 5 的权重加载到槽位 0；等待加载完成并发布新映射。
4. 如果有独立且已就绪的计算，这段传输可以与其重叠；是否真的重叠要看 profile。
5. 到达 L，实际路由需要专家 `{2, 5, 7}`。此时 2 和 5 已在 NPU，缺少 7。
6. reactive 的 `observe` 统计真实访问；`choose_victims` 保护本轮需要的专家，
   例如选择槽位 2 中的专家 4。
7. 提交专家 7 的权重传输，等待完成，再写回映射。
8. L 的 MoE 计算从槽位 1、0、2 读取专家 2、5、7 的权重。

没有预取时，第 5 步通常还需要同时加载专家 5。预取的潜在收益是把这部分工作提前。
但若第 5 步到来时预取还没完成，计算流会先等待；若真实路由不需要 5，这次传输可能浪费。
“提前加载了一次”不自动等于“请求少等了一次完整传输时间”。

在图重放中，这个例子每轮仍可选择不同专家：图中重复的是路由、回传、回调、写回等依赖结构，
CPU 回调读到的路由内容不同，因此这轮加载 5/7，下一轮可能加载其他专家。

## 9. 用现有 MSTX 标记读懂时间线

| 时间块 | 实际覆盖范围 |
| --- | --- |
| update_weights | CPU `_update_weights` 主体，不含所有外层回调与设备依赖 |
| observe / choose_victims | LRC 的对应函数，C++ 模式仍从 CPU 包装层进入 |
| gap_mapping / gap_missing | 槽位映射整理、缺失专家计算；预取还包括候选筛选 |
| gap_hit_stats / gap_cache_stats | 命中率和缓存统计 |
| gap_slots / gap_debug | 候选槽位整理、debug 日志处理 |
| load_submit | 准备并提交加载，包含下面三项 |
| load_prepare / copy_submit / scale_refresh | 构建任务 / 提交拷贝 / 更新相关 scale |
| synchronize | CPU 等待加载流完成 |
| 设备 MEMCPY_ASYNC | 实际设备传输；可能是权重、量化参数、路由或映射，需看方向和数据量 |

父子块不能重复相加。`outer = total - observe - choose_victims - synchronize` 是不连续区域之和，
包括加载提交和日志等，不等于纯 Python 计算，更不等于 runtime 回调开销。
MSTX 和计时本身有开销；同一阶段的时间跨度还可能包含线程调度暂停。

已观察到 `gap_debug` 在该轮细分项中占比最大，说明日志值得优先排除。
这不能直接证明所有空泡都由日志导致，也不能在没有新对照的情况下把它从旧数据中扣除。
具体开关和图中字段参见 [MSTX 可视化说明](offload_mstx.md)。

## 10. 接下来研究“回调同步开销”时测什么

可以按以下边界拆解，避免把几段时间算成同一项：

```text
A：设备前序数据已就绪，回调具备执行条件
B：CPU 开始执行外层 host callback
C：进入 _update_weights
D：退出 _update_weights
E：外层 callback 返回
F：相关设备后续任务开始
```

- A→B：回调调度/通知等等待的候选范围，需要设备与主机时间轴关联确认。
- B→C、D→E：guard、快照及外层包装等；现有 update_weights 标记没有全部覆盖。
- C→D：已有 MSTX 和计时可细分，含 LRC、外围、提交、H2D 等待。
- E→F：后续任务恢复、映射传输及其他依赖等可能影响这一段。

仅凭两次 `MEMCPY_ASYNC` 的间隔，不能直接求出 A→B 或 E→F。
Python 调用 `_launch_host_func` 的起止，也不等于重放时每次回调的调度开销。
本仓库图重放代码在特定条件下还会执行 replay 前的流同步，
那属于另一处依赖，不能自动归到每层 host callback 上。

当前能确认的是依赖关系和已有函数边界；还没有证据确认“400 微秒空泡中，
剩下 300 微秒一定是 `_launch_host_func` 同步”。
后续替换同步方案必须保持三个正确性条件：CPU 读取的是最新路由；
NPU 读取的是已加载完成的权重；专家 ID 与实际槽位内容一致。
因此不能直接删掉 `_synchronize_h2d()` 或预取 join 来证明优化有效。

## 11. 代码阅读入口

| 文件 | 建议查找的函数或位置 |
| --- | --- |
| [专家管理器](../vllm_ascend/expert_offload/expert_offload_manager.py) | `update_weights`、`_update_weights_guarded`、`_update_weights` |
| 同上 | `_build_expert_h2d_tasks`、`_load_expert_weights_into_slots`、`_synchronize_h2d` |
| 同上 | `predict_next_layer_experts_npu`、`trigger_next_layer_prefetch`、`_dispatch_prefetch` |
| [TorchCopy 后端](../vllm_ascend/expert_offload/h2d_transfer.py) | `TorchCopyH2DTransport.copy_batch`、`synchronize` |
| [图捕获与重放](../vllm_ascend/compilation/acl_graph.py) | `torch.npu.graph(...)`、`entry.aclgraph.replay()` |
| [MXFP4 路径](../vllm_ascend/quantization/methods/w4a8_mxfp4.py) | 预取调用与 `fused_experts` 的实际顺序 |
| [W4A8 路径](../vllm_ascend/quantization/methods/w4a8.py) | 另一种预取调用位置 |
| [计时与 MSTX](../vllm_ascend/expert_offload/callback_timing.py) | `wrap_update`、`wrap`、`callback_stage` |
| [当前启动配置](../script/test_graph2.sh) | 图模式、MTP、预取、debug 开关和模型路径 |
