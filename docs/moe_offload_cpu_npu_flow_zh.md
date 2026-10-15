# MoE 卸载的 CPU/NPU 流程：从入图到 host callback

本文面向只有 LLM 基础的读者，说明本项目如何在 CPU 内存与 NPU 显存之间管理专家权重，
重点解释 `_launch_host_func`、`_update_weights` 和图捕获/重放。
以当前 `script/test_graph2.sh` 的单卡、LRC、TorchCopy 路径为主，核对日期为 2026-10-15。
本文解释已有实现，不新增同步优化，也不把待验证的性能猜测当作结论。

## 先读：一次 MoE 卸载，CPU 与 NPU 到底怎样接力

先不考虑入图，只考虑一个 token 到达一个 MoE 层。
假设路由选择了专家 2、5，其中专家 2 在 NPU，专家 5 的权重仍在 CPU 内存。
本轮的目标是把专家 5 加载到 NPU，之后再执行专家计算。

这里的“卸载”是把部分权重留在 CPU 以节省显存；缺失时进行的是加载回 NPU。
本路径不是先把被替换专家的权重传回 CPU：CPU 已保留权重来源，NPU 槽位可以直接被覆盖。

### 第一步：NPU 算出需要哪些专家，再把名单告诉 CPU

NPU 完成前面的 Attention 和路由计算，得到专家 ID，例如 `[2, 5]`。
CPU 必须知道这个结果，才能决定搬运谁，所以外层 `update_weights` 安排：

```python
# 简化代码：左边是 CPU 暂存区，右边是 NPU 数据。
topk_ids_h.copy_(topk_ids, non_blocking=...)
log2phy_h.copy_(log2phy, non_blocking=...)
```

这是 NPU→CPU 的 D2H，传的是专家名单和槽位映射，不是专家权重。
异步提交后，CPU 暂存区不一定立刻更新完成，不能不顾执行顺序马上读取。

### 第二步：安排 CPU 在数据准备好之后处理权重

图捕获路径随后调用：

```python
torch_npu.npu._launch_host_func(
    current_compute_stream,
    self._update_weights_guarded,
    args,
)
```

它在本项目中的作用可以理解为：
**“在这条流前面的信息回传完成之后，调用这个 CPU 函数；后续依赖结果的任务要等它完成。”**

| 名字 | 负责什么 |
| --- | --- |
| `_launch_host_func` | 安排何时执行 CPU 函数；自己不选择专家，也不搬运专家权重 |
| `_update_weights_guarded` | 实际被调用的 CPU 函数，调用主体并记录异常 |
| `_update_weights` | 真正做缓存判断、选择替换对象、提交加载和等待加载 |

“host callback / 主机回调”就是这次由运行时触发的 CPU 函数执行。
它不是 NPU 在执行 Python。提交这个安排的线程与实际执行回调的线程，也不要当成同一个线程。

### 第三步：CPU 判断搬谁、放哪里

CPU 回调线程进入 `_update_weights`，执行：

```text
observe：记录这次真实访问专家 2、5
    ↓
查缓存：2 已在 NPU，5 不在，需要加载 5
    ↓
整理缓存统计、候选槽位和 debug 日志
    ↓
choose_victims：选择可以覆盖的专家，例如专家 0
    ↓
确定加载任务：把专家 5 放入原来专家 0 所在的槽位
```

这些是 CPU 工作，后续依赖新权重的 MoE 还不能开始。
其他独立设备任务可能继续执行，不能因此认定整个 NPU 都在等待或都在计算。

### 第四步：CPU 提交加载，设备执行传输

CPU 调用 `_load_expert_weights_into_slots`，即 MSTX 中的 `load_submit`：

```text
load_prepare：找到 CPU 权重来源和 NPU 目标槽位，构造任务
    ↓
copy_submit：调用 copy_batch，把拷贝任务提交到加载流
    ↓
scale_refresh：处理有关的 scale 更新
```

加载流 `load_stream` 是另一条设备任务队列。
CPU 的“提交拷贝”和设备的“执行拷贝”不是一件事：提交之后，设备按照依赖执行传输。
相关的 `MEMCPY_ASYNC` 是设备传输时间块。CPU 还在提交后续任务时，前面的传输可能已经开始。

### 第五步：CPU 等待加载流完成

回调内随后执行：

```python
self._synchronize_h2d()
# 当前 TorchCopy 后端最终调用：
self.load_stream.synchronize()
```

意思是“CPU 回调先不要结束，等加载流上的工作完成”。这不是发起传输，传输已经提交了。
此时存在两层等待：

```text
后续 MoE 计算 等 CPU 回调完成
CPU 回调      等 加载流完成权重传输
```

### 第六步：告诉 NPU 新的槽位位置，再进行专家计算

`_update_weights` 结束、外层回调返回后，计算流继续执行映射写回：

```python
log2phy.copy_(log2phy_h, non_blocking=...)
```

例如告诉 NPU“专家 5 现在在槽位 0”。权重和映射都就绪后，后续 MoE 才能正确计算。
CPU 提前更新映射数据，不代表 NPU 可以在权重加载完成前使用它；上述依赖保证二者一致。

### 六步放在一张图里看

这是依赖关系示意，不按真实耗时比例绘制，时间共同向右推进：

```text
NPU 计算流：路由 → ID/映射回传 → [等待 CPU 回调完成] → 新映射写回 → MoE
                                      │       ↑
CPU 回调：                            └→ 规划 → 提交 → 等加载完成 → 返回
                                                │       ↑
NPU 加载流：                                    └→ 权重 H2D ─┘
```

**`_launch_host_func` 连接 NPU 的执行顺序与 CPU 函数；
`_update_weights` 完成这段 CPU 函数内部的实际工作。**

外层 `update_weights`（无下划线）安排回传、回调、映射写回。
内层 `_update_weights`（有下划线）负责 CPU 缓存与加载管理。
当前 MSTX 的 `update_weights` 标记测的是内层，不是上述六步的总时间。

## 再读：入图究竟改变了上面哪件事

### 普通执行：CPU 每次重新逐项提交

前面的卸载流程涉及两个执行者：CPU 执行 Python，经由 PyTorch/torch_npu 组织任务；
NPU 执行底层设备程序，完成数值计算。下面用一段简单计算说明它们的分工，
假设 `x` 和 `w` 已在 NPU 上：

```python
a = x + 1
b = a * 2
y = b @ w
```

执行第一行时，CPU 准备加法任务，将输入/输出地址、形状、数据类型等参数交给运行时。
NPU 根据任务执行加法；收到的是设备任务及参数，而不是 `a = x + 1` 这段 Python 文本。
由于输入已在 NPU 上，提交时可以直接引用其地址，无需再次搬运输入数据。

返回的 `a` 是真正的 Tensor 对象，对应结果存储空间，但数值可能尚未计算完成。
CPU 可以继续执行第二行，安排从这个位置读取结果并乘以 2；设备依赖保证先写入、后读取。
因此 Python 按行执行，并不要求每行都等 NPU 算完。结果形状由操作推导，
例如矩阵乘法的输出形状就可能与输入不同。

对于每份新输入，普通执行通常都要重新经过这几行 Python，组织并提交相应任务：

```text
第 1 个输入：CPU 提交加法、乘法、矩阵乘法；NPU 按依赖计算
第 2 个输入：CPU 重新组织并提交同样的操作；NPU 按依赖重新计算
```

当设备操作很多、每个又比较短时，反复执行 Python 并逐项提交本身就可能拖慢速度。

### 捕获：把设备工序记录下来

如果工序结构一直都是“加法→乘法→矩阵乘法”，可以在准备阶段进入捕获模式。
以下为原理示意，省略图对象创建、预热和流管理：

```python
with torch.npu.graph(graph):
    a = input_buffer + 1
    b = a * 2
    output_buffer = b @ w
```

框架执行这段 Python，记录可捕获的设备任务、缓冲区和任务之间的先后依赖。
得到的“图”可以理解成一张可重复执行的工序表：

```text
从 input_buffer 读数据 → 加 1 → 乘 2 → 与 w 做矩阵乘法 → 写 output_buffer
```

“入图”就是让相应工作进入这张可重复执行的工序表。
它不是 MindStudio 的可视化图片，也不是保存上一次计算结果。

### 重放：更新输入，再执行工序表

下次来了新输入，框架将数据放入相应缓冲区，然后调用：

```python
graph.replay()
```

意思是“再执行这张图记录的任务”。NPU 仍然执行加法、乘法和矩阵乘法，重新计算输出。
省下的是反复通过原来那几行 Python 逐项组织、提交这些操作的部分开销。

```text
准备阶段：捕获加法→乘法→矩阵乘法及其依赖
正式执行：更新输入 → replay → 新输出
下次执行：更新输入 → replay → 另一个新输出
```

重复执行需要满足形状、内存地址和任务结构等约束，不是任意输入都能使用同一张图。
框架可以为不同规模准备不同图；当前 `[1, 2, 3]` 配置与此有关，不是预取三层。

普通执行和重放中，NPU 执行的都是设备程序。变化发生在 CPU 一侧：重放复用已记录的
执行计划，通常不再执行被捕获部分的逐项 Python 调度；图外的程序仍然继续运行。

### MoE 的难点：专家名单每次变化，CPU 必须重新判断

现在回到卸载例子：第一次路由是 `[2, 5]`，第二次是 `[2, 7]`。
如果只在捕获时执行一次 Python 缓存判断，以后都不再运行它，就可能一直沿用第一次的加载决策。

普通 Python 的集合、字典操作不会因为出现在捕获代码里，就自动成为每次执行的 NPU 算子。
同样，固定配置的 Python `if` 在捕获时选择分支，重放沿用该分支记录的设备路径，
不会重新判断这个 Python 条件。

MoE 缓存判断却需要每轮读取新路由，不能只保留捕获时的一次决策。
这种 CPU 动态逻辑需要明确安排在图外，或通过受支持的回调接入。
本项目采用后者，在工序表中写入一项：“到这里，请 CPU 重新处理本轮路由”。
运行时按这个显式安排触发回调，并不会遇到任意 Python `if` 就自动返回 CPU 判断。

### _launch_host_func 就是在图中安排这项 CPU 工作

本项目捕获的流程可以理解为：

```text
[NPU 路由] → [回传名单] → [调用 CPU 回调] → [写回新映射] → [NPU MoE]
                                │
                                └→ _update_weights
                                   根据本次名单动态加载权重
```

| 阶段 | 发生什么 |
| --- | --- |
| 捕获 | Python 执行模型路径；用 `_launch_host_func` 安排回调入口，记录相应设备任务和依赖 |
| 第一次重放 | 图触发回调；CPU 读到 `[2, 5]`，根据当前缓存决定是否加载专家 5 |
| 第二次重放 | 图再次触发回调；CPU 读到 `[2, 7]`，重新决定是否加载专家 7 |

重复的是回调入口，不是固定的加载名单。
回调动态提交的权重拷贝，也不能一概视为捕获时就固定好的拷贝节点。
**CPU 回调在图的流程里，不等于回调内的 Python/C++ 在 NPU 上执行。**

### 为什么 capturing 分支只运行在捕获时，重放仍然有回调

```python
if _EXTRA_CTX.capturing:
    _launch_host_func(stream, self._update_weights_guarded, args)
else:
    self._update_weights_guarded(args)
```

分成三个时机看：

1. 普通执行：Python 正在执行完整前向，走 else，直接调用 CPU 函数。
2. 捕获：Python 正在构建可重放流程，走 if，把回调安排进流程。
3. 重放：外层不重新执行这段 if/else，而是执行已记录的图；图里的回调节点再次触发 CPU 函数。

因此不需要每次重放重新进入 if，CPU 回调也会执行。
图减少了普通前向的逐项调度，但明确安排的 CPU 回调仍然运行。
LRC 改成 C++ 只改变回调内部一部分 CPU 工作，不会自动去掉回调和依赖等待。

### 入图没有消除的四件事

- 路由和 MoE 仍要真实计算。
- 缺失专家权重仍要传到 NPU。
- CPU 仍要处理最新路由和缓存状态。
- 后续专家计算仍必须等正确权重和映射就绪。

分析 `_launch_host_func` 时，要把“调用接口安排回调”“等待回调开始”“执行回调函数”
和“回调结束后设备继续运行”分别看，不能统称为一个同步耗时。

以下章节作为术语、实现细节和代码索引参考；理解主流程时，先看上面两部分即可。

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

## 4. 三种“同步”不要混为一谈

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

## 5. 预取：提前做可能会需要的加载

### 5.1 目标和限制

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

### 5.2 预取路径的 CPU/NPU 接力

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

### 5.3 预取为什么可能出现在 Attention 后

不要只看注释判断触发位置。当前实际代码存在差别：

- `w4a8_mxfp4.py`：`trigger_next_layer_prefetch(layer, x)` 位于 `fused_experts(...)` 之前，
  尽管附近注释写着 GMM 提交之后。
- `w4a8.py`：调用位于 `fused_experts(...)` 之后。

实际走哪条量化路径要结合运行环境确认。第一种路径可以在当前层 Attention/路由之后、
当前层专家计算之前发起下一层预取。MSTX 预取块的 `layer` 是加载目标层索引，
不能将旁边 Attention 的层号直接套给它；索引来自 `moe_layers`，也不要擅自改成从 1 开始的编号。

## 6. 一个完整例子

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

## 7. 用现有 MSTX 标记读懂时间线

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

## 8. 接下来研究“回调同步开销”时测什么

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

## 9. 代码阅读入口

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
