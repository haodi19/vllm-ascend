# MoE 卸载的 CPU/NPU 流程：从入图到 host callback

本文面向只有 LLM 基础的读者，说明本项目如何在 CPU 内存与 NPU 显存之间管理专家权重，
重点解释 `_launch_host_func`、`_update_weights` 和图捕获/重放。
以当前 `script/test_graph2.sh` 的单卡、LRC、TorchCopy 路径为主，本文更新日期为 2026-10-20。
容器已确认 torch_npu 版本为 2.10.0；底层分析以公开 v2.10.0 源码为依据，定制构建的差异需另行核对。
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

`current_compute_stream` 指定把回调放入哪条设备流，从而建立它与前后设备任务的顺序关系。
这里没有另设一个 Python Event 来检测“回传结束”；运行时在前序任务完成、执行到回调任务时触发它。
它也不指定回调中的权重加载流：加载流由 `_update_weights` 内部选择。

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

这里要分清提交与执行：CPU 调用 `_launch_host_func` 返回后，可以继续执行映射写回那一行，
但这只是提前安排后续工作。回调采用阻塞属性，所以映射拷贝的设备执行必须等回调结束。
图重放时，回调返回给运行时，随后执行已记录的映射拷贝；不会返回某一行外层 Python 继续执行。
普通 Eager 路径则是直接调用 guarded 函数，返回后由外层 Python 继续安排映射拷贝。

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

本项目的捕获入口在 `vllm_ascend/compilation/acl_graph.py`。框架进入
`with torch.npu.graph(...)` 后执行模型前向；执行到 MoE 层的路由之后、专家计算之前，
调用外层 `update_weights`。它安排路由/映射回传、CPU 回调和新映射写回，
这些任务及依赖成为被捕获流程的一部分。捕获的不是 `_update_weights` 中每一条 Python 语句。

这段流程可以理解为：

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

路由暂存区可以保持同一个地址，但每轮 D2H 会写入本轮数据：第一次是 `[2, 5]`，
第二次是 `[2, 7]`。回调通过保存的 Tensor 对象读取暂存区，因而能作出不同的加载决策。
固定地址本身并不保证读到最新数据，还必须保证本轮写入完成、读取期间不被提前覆盖。
图中的任务顺序正是保障这些条件的一部分；图固定的是执行结构和缓冲区安排，而不是输入内容。

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

## 4. 回调的底层执行与同步关系

- **CPU 提交线程**：执行模型 Python 代码、向 NPU 提交任务的线程，也就是本文中的“CPU 模型执行线程”。
- **CPU 报告线程**：接收设备回调通知，并执行 `_update_weights` 等 CPU 回调工作的线程。
- **NPU 流（Stream）**：组织设备计算、传输和等待任务的有序队列，不是 CPU 线程。

torch_npu 部分依据公开的 2.10.0 源码；Runtime 与驱动部分依据公开源码中的相关分支，
尚未与绿区实际加载的 CANN、驱动版本和硬件路径逐项匹配。
下文用 STARS 的 Event 实现串联设备侧依赖，其他硬件分支可能使用不同的完成通知机制。

### 4.1 第一步：CPU 提交线程执行到 `_launch_host_func`

外层 `update_weights` 先安排路由回传，再登记 CPU 回调，最后安排映射写回。
下面是图捕获分支的简化代码：

```python
topk_ids_h.copy_(topk_ids, non_blocking=True)  # 路由结果 NPU → CPU
log2phy_h.copy_(log2phy, non_blocking=True)    # 当前映射 NPU → CPU

torch_npu.npu._launch_host_func(
    current_compute_stream,
    self._update_weights_guarded,
    args,
)

log2phy.copy_(log2phy_h, non_blocking=True)    # 更新后的映射 CPU → NPU
```

此时 CPU 提交线程是在安排任务，不是在这里直接执行 `_update_weights_guarded`。
`_launch_host_func` 返回后，提交线程可以继续执行下一行，提前安排映射写回。
设备需要遵守的顺序则是：路由回传完成，CPU 回调处理完成，然后才能执行映射写回。
接下来的几步说明这层依赖如何建立。

### 4.2 第二步：torch_npu 将 Python 回调包装成 C++ 可调用的形式

Python `_launch_host_func` 将参数转交给 `_C` 扩展。CANN 接口不能直接执行 Python 函数对象，
因此 C++ 绑定保存 Python 函数与参数，并登记一个桥接函数 `LaunchCallFunc`：

```text
保存的 Python 函数：_update_weights_guarded
保存的 Python 参数：args

交给 CANN 的函数：LaunchCallFunc
交给 CANN 的参数：指向上述保存信息的指针
```

等到设备通知可以执行回调时，`LaunchCallFunc` 再进入 Python 调用业务函数。
`PyFuncStruct` 保存对象引用、保持对象有效，并不在登记时自动搬运 Tensor 内容；
CPU 所需的路由数据由第一步的显式 `copy_` 准备。
实现见 [Graph.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/npu/Graph.cpp)
和 [Graph.h](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/npu/Graph.h)。

### 4.3 第三步：调用 `AclrtLaunchCallback`，将回调安排到指定流

包装完成后，调用继续沿下面的路径进入 Runtime：

```text
torch_npu.npu._launch_host_func(stream, guarded, args)
    → torch_npu._C._launch_host_func
    → c10_npu::launch_callback(stream, LaunchCallFunc, data)
    → AclrtLaunchCallback(..., ACL_CALLBACK_BLOCK, stream)
    → CANN aclrtLaunchCallback
    → Runtime rtCallbackLaunch
```

`launch_callback` 将类型明确设为 `ACL_CALLBACK_BLOCK`，该 Python 接口没有提供非阻塞选项。
虽然名字包含 host_func，这条路径使用的是 `aclrtLaunchCallback`，并非简单调用同名的
`aclrtLaunchHostFunc`。依据见 [C++ 绑定](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/npu/Graph.cpp)
和 [底层提交](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUGraph.cpp)。

大写开头的 `AclrtLaunchCallback` 是 torch_npu 的封装，获取并缓存 CANN 接口函数地址后
转发调用。真正的回调任务安排继续进入 CANN Runtime。
实现见 [AclInterface.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/interface/AclInterface.cpp)。

`stream` 决定回调放在哪条流上，从而确定其前序任务；`ACL_CALLBACK_BLOCK` 要求该流
后续任务等待回调完成。提交接口本身不以业务回调执行完毕为返回条件：

```text
CPU 安排顺序：提交路由回传 → 安排回调 → 提交映射写回
设备依赖顺序：路由回传完成 → 回调执行并完成 → 映射写回执行
```

因此，“异步提交”和“阻塞后续设备任务”同时成立。这里的阻塞不直接作用于所有设备流。
若换成不阻塞后续任务的机制，必须另行保证权重和映射就绪，否则映射写回可能读取旧内容。

### 4.4 第四步：Runtime 为设备安排通知 CPU 和等待 CPU 的任务

相关流在使用回调前，已通过 `_subscribe_report(stream)` 与报告线程建立订阅关系。
torch_npu 按需创建常驻报告线程，设置设备上下文，随后循环调用 `aclrtProcessReport`。
订阅不是每次回调都创建线程；多条流的订阅也不意味着每条流都有一个独立执行回调的线程。
Runtime 保存线程、设备、流之间的关系，并分配或复用用于回调的命令队列和报告队列。
订阅管理见 [subscribe.cc](https://github.com/hicann/runtime/blob/master/src/runtime/core/src/utils/subscribe.cc)。

提交线程调用 `aclrtLaunchCallback` 后，ACL 层将 `ACL_CALLBACK_BLOCK` 转为阻塞标志，
交给 Runtime。在查到的 STARS 分支中，`StarsLaunchEventProc` 依次安排：

```text
携带 Event 信息的回调通知任务
    → 等待对应 Event 的设备任务
    → 重置该 Event 的设备任务
    → 原本的后续设备任务
```

Event 在这里表示运行时内部的完成状态。提交等待任务后，CPU 提交线程可以返回并继续
工作；设备执行到等待节点时，后续依赖任务才受到限制。项目不需要显式创建这个内部 Event。
流程见 [Runtime 回调实现](https://github.com/hicann/runtime/blob/c9dda362ac0fbe2507d1357b6341df186facd2f9/src/runtime/api/impl/api_impl.cc)。

继续向下，设备等待被编码为 `RT_STARS_SQE_TYPE_EVENT_WAIT` 类型的 SQE。
SQE 是交给设备调度系统的任务描述，其中携带 Event 编号、流和任务信息等。
这里提交的是专门的 Event 等待任务，不是 AIV 计算 kernel。
公开代码展示了任务构造，但没有完整展示设备调度器内部如何检查 Event 状态，
因此不能进一步认定硬件采用某种轮询频率或由一个 AIV 核持续读标志。
任务编码见 [event_task_v100.cc](https://github.com/hicann/runtime/blob/c9dda362ac0fbe2507d1357b6341df186facd2f9/src/runtime/core/src/task/task_info/event/event_task_v100.cc)。

### 4.5 第五步：NPU 执行到回调任务，通知 CPU 报告线程

前四步介绍 CPU 如何安排任务，从这里开始跟随设备的实际执行过程。
设备按依赖执行前面的路由回传，到达回调通知任务后，产生回调报告。
报告携带识别和执行该回调所需的信息，包括函数地址、参数地址以及流、任务等信息。
执行时机由设备到达任务节点决定，CPU 不需要检查路由 Tensor 来猜测何时开始回调。

报告线程此时可能正在 `aclrtProcessReport` 内等待。在查到的公开驱动回调路径中，
报告接收的调用链为：

```text
aclrtProcessReport
    → Runtime ProcessReport
    → halCqReportIrqWait
    → 回调报告接收逻辑
    → halEschedWaitEvent
    → ioctl：进入内核驱动，等待对应事件
```

`halCqReportIrqWait` 和 `halEschedWaitEvent` 都是函数。
内核驱动通过等待队列检查事件是否就绪：条件已满足时直接继续；条件未满足时，
使用 Linux 的 `wait_event_interruptible_timeout` 等接口等待，报告线程可以休眠、让出 CPU。
驱动处理到达的回调事件后，将对应线程状态置为 READY，并通过 `wake_up_interruptible`
唤醒等待者。唤醒使线程具备运行条件，线程实际恢复执行仍取决于操作系统调度。

因此，外层虽然循环调用 `aclrtProcessReport`，这条路径并不是持续忙读共享内存 flag。
等待超时是一次等待的上限，也不是每次回调必须额外等待的固定时间。
报告已经就绪时，不必经历一次休眠和唤醒。
用户态与内核路径分别见
[事件等待入口](https://github.com/hicann/driver/blob/master/src/ascend_hal/esched/comm/event_sched.c)、
[内核等待与唤醒](https://github.com/hicann/driver/blob/master/src/sdk_driver/esched/dc/host_adapt/common/esched.c)
和 [Linux 等待接口映射](https://github.com/hicann/driver/blob/master/src/sdk_driver/kernel_adapt/include/ka_task_pub.h)。

接收成功后，驱动将报告整理到缓冲区并返回有数据的队列信息。
Runtime 再调用 `halCqReportGet` 获取报告缓冲区地址和数量。
在这条实现中，`Get` 取的是已接收的报告，不再次等待 NPU，也不直接执行业务函数。
实现见 [报告接收与获取](https://github.com/hicann/driver/blob/master/src/ascend_hal/trs/core/trs_sqcq.c)
和 [回调事件转换](https://github.com/hicann/driver/blob/master/src/ascend_hal/trs/core/trs_cb_event.c)。

### 4.6 第六步：报告线程执行 `LaunchCallFunc`，进入 `_update_weights`

Runtime 从报告中取出函数与参数地址，在报告线程中调用 `LaunchCallFunc`。这个包装函数获取 GIL，
构造 Python 调用参数，通过 `PyObject_CallObject` 调用 `_update_weights_guarded(args)`。
GIL 是常规 CPython 中线程执行 Python 所需的解释器锁；存在这一步不代表它一定是耗时瓶颈。

```text
报告线程取得回调报告
    → Runtime 调用 LaunchCallFunc
    → 获取 GIL、构造 Python 调用参数
    → _update_weights_guarded
    → _update_weights：缓存规划、提交加载、等待加载流
    → 返回 guarded，再返回 C++ 包装层
    → 清理临时对象、释放获取的 GIL 状态
    → 返回 Runtime，进入完成通知处理
```

在回调路径中，内层 `_update_weights` 的 LRC、日志、加载提交和加载等待都由报告线程执行。
“CPU 提交线程”只是模型执行线程在前文中的角色名称，并不意味着只有它能提交设备操作；
报告线程也会在 `_update_weights` 中向加载流提交权重拷贝。
Python 回调开始前的报告处理与 GIL 获取、结束后的 C++ 清理，不完全包含在现有
`host_callback` 的 Python MSTX 范围内。

执行 `_update_weights` 期间，原计算流仍在等待回调完成，权重拷贝则在加载流上执行。
其中 `_synchronize_h2d()` 是 CPU 等待加载完成，属于业务函数内部的等待，
不能全部算成回调通知和线程调度的成本。

### 4.7 第七步：CPU 回调结束，Runtime 向设备发送完成通知

业务函数返回后，`LaunchCallFunc` 清理 Python 临时对象、释放获取的 GIL 状态，
再返回报告处理逻辑。此时 Runtime 仍需完成设备侧通知。
在上述 STARS 分支中，Runtime 填写包含流、任务和 Event 编号的完成命令，
经 `SqCommandSend`、驱动 `halSqMsgSend` 发送。
这一步仍由报告线程执行 Runtime 代码；Runtime 是运行时库，不是另外一条固定线程。
驱动接口见 [npu_driver_res.cc](https://github.com/hicann/runtime/blob/master/src/runtime/driver/npu_driver_res.cc)。

因此，Python 回调返回、原生回调返回和后续设备任务开始，是不同的时间点。
回调结束不会返回到一段被挂起的外层 Python `update_weights`；提交线程早已可以继续
组织后续工作，设备侧通过已安排的等待依赖决定何时放行。

### 4.8 第八步：设备收到完成状态，原来的流继续执行

设备处理完成通知后，对应 Event 的等待条件满足，第四步安排的等待任务才可以通过。
随后执行 Event 重置和原本的后续任务：

```text
CPU 回调完成通知到达设备
    → Event 等待通过
    → Event 重置
    → 更新后的 log2phy 映射 H2D
    → 后续专家计算
```

这使得第一步提前提交的映射写回不会读取尚未更新好的内容。
CPU 侧等待队列用于等待回调报告，设备侧 Event 用于阻止后续任务提前执行，
两者分处交接的两端，不是同一个等待对象。

#### 图重放中的完整时间线

图捕获时，CPU 提交线程运行外层 `update_weights`，安排路由 D2H、登记回调节点，
并安排映射 H2D。图重放时，这段外层 Python 不再逐行执行；提交线程提交图执行，
设备复用图中的任务和依赖。每轮到达回调节点后，报告线程仍需实际执行内层函数，
并完成上述通知往返。

非图路径则直接调用 `_update_weights_guarded(args)`，不经过 `_launch_host_func`。
此时内层 `_update_weights` 由调用外层函数的 CPU 提交线程执行。
不能将图回调路径的报告线程分工套到所有执行模式。

下面以图重放为例，时间从上向下，箭头表示依赖，不表示精确等时：

```text
CPU 提交线程             CPU 报告线程                  NPU
      │                      │                        │
提交 graph.replay        等待回调报告              执行图中路由计算
      │                      │                    路由结果 D2H
      │                      │                        │
      │                      │ ←── 回调报告 ──── 回调通知节点
      │                  等待返回                 Event 等待
      │                  获取报告                     │
      │                  获取 GIL                     │
      │                  _update_weights              │
      │                    ├─ LRC、日志等             │
      │                    ├─ 提交权重 H2D ─────→ 加载流执行拷贝
      │                    └─ 等待加载完成            │
      │                  Python/C++ 回调返回          │
      │                  发送完成通知 ──────────→ Event 等待通过
      │                  再次等待报告             映射 H2D、专家计算
```

原计算流等待期间，独立的加载流仍需能够执行；某条流上的 Event 等待不代表整张卡停止工作。
图中“等待报告”是 CPU 线程的内核等待队列机制，“Event 等待”是设备流的执行依赖，
二者位于不同侧，不能视为同一个等待对象。

#### 时间线中的等待与开销

| 位置 | 谁在等谁 | 本项目的意义 |
| --- | --- | --- |
| 回调前后流依赖 | 流的后续任务等待回调满足执行约束、完成工作 | 不能在最新映射或权重尚未就绪时执行专家计算 |
| `_synchronize_h2d()` | CPU 回调线程等待加载流 | TorchCopy 后端调用 `load_stream.synchronize()` |
| `wait_event(...)` | 一条设备流等待另一条流的事件 | 当前层使用权重前，等待预取流完成相关工作 |

`wait_event` 表达设备依赖，不等于调用它的 Python 线程必须现场等到事件完成。
`stream.synchronize()` 则是主机侧等待。异步提交不意味着没有任何依赖等待，
多个 Stream 也不意味着回调可以与所有计算自由重叠。

当前代码在回调内向另一条加载流提交拷贝，并等待该流完成；这是业务函数引入的等待，
不能全部算成底层回调调度成本。报告处理、GIL 获取、回调包装和结束处理也是潜在开销，
但目前没有各项独立测量值。更换底层同步机制前，需要核对对应接口的调用限制并保留跨流依赖。

一次 `halCqReportIrqWait` 的持续时间还可能包含等待前序 NPU 任务完成的正常时间，
不能将其整体当成交接成本。后续 AIV 与共享内存 polling 方案要比较的是通知、等待和
恢复执行的机制成本，同时保留路由数据可见、权重加载完成及映射就绪的正确顺序。

### 4.9 参考方案：vLLM-ascend-DSA 的任务标志与完成标志交互

当前项目内的 `vLLM-ascend-DSA` 提供了另一种 CPU–NPU 交接方式：设备发布任务，
Host 常驻线程轮询处理，完成后回写标志，设备检测到完成再继续。
下面沿 raw-Hcomm token-gather 与 remote Attention 路径介绍。
它处理的是 KV Cache 数据，尚未接入本项目的 MoE 卸载，也不是已验证的回调替代方案。

这条路径包含三个角色：设备侧 Hcomm 代码发布搬运任务，Host 工作线程处理任务，
设备侧 AIV 消费数据。AIV 是 NPU 上执行向量类计算的单元；这里的 AIV 代码属于
Attention 算子，在读取远端搬来的 KV 前检查完成状态。

#### 第一步：设备准备任务，发布提交标志

每次任务使用一个序号 `generation`。以第 17 次任务为例，交接涉及两处存储：

```text
Host 内存中的任务槽：
    generation       = 17    任务内容中的编号
    commitGeneration = 17    该任务已发布，可以处理

NPU 内存中的完成槽：
    expectedGeneration = 17  本轮等待的编号
    observedGeneration = 0   尚未收到本轮完成状态
```

设备先初始化完成槽，再准备搬运任务的内容，包括数据来源、目标位置和本轮编号。
随后构造 Hcomm WRITE 操作，先发送任务内容，再发送 `commitGeneration`。
Host 轮询的是发送到 Host 内存的任务槽；设备侧另有构造任务用的暂存数据，
其中同名字段不能与 Host 轮询的那份混为一谈。

这里的标志通过注册内存与通信地址交互，不是一个可以由双方任意访问的普通 Python 变量。
序号用于区分不同轮次，还需要配合正确的发布顺序和槽位复用，避免读取旧任务。
字段与发布代码见参考仓库中的
`hcomm_kernel/raw_ub_hybrid_protocol.h`（`RawUbCompletionSlot` 等结构）和
`hcomm_kernel/kvcache_scatter_copy_hcomm.cpp`（任务内容与提交序号的 WRITE 描述符）。

#### 第二步：Host 常驻线程轮询并处理任务

Host Mock Pool 中的 `PollCommands` 持续检查自己负责的任务槽：

```text
循环检查任务槽
    → commitGeneration 非零
    → 进入 ProcessCommand
    → 确认 commitGeneration 与任务内容的 generation 匹配
    → 按任务描述整理、搬运 KV 数据
```

这里是主动读取内存状态的 polling，不经过前文 `_launch_host_func` 的报告等待路径。
线程由 Host Mock Pool 服务组织，不能与当前 MoE 进程的报告线程混为一谈。

`--poll-threads` 配置轮询线程数，代码默认值为 1；`--poll-cpus` 可指定 CPU 绑定。
多个轮询线程按编号分担槽位，例如两个线程时，线程 0 检查 0、2、4 等槽位，
线程 1 检查 1、3、5 等槽位。它们是同类工作线程，发现任务后都会执行处理逻辑。
另有 `--gather-threads` 配置数据整理工作线程，与轮询线程的职责不同。
相关实现位于 `test/urma_mock_pool/urma_mock_pool.cpp` 的 `PollCommands`、
`ProcessCommand` 和启动参数处理代码。

#### 第三步：Host 完成数据传输，回写完成序号

Host 整理数据、提交传输后，先通过 `PollJfc` 等待相关数据传输完成，
再通过 URMA WRITE 将本轮序号写到 NPU 内存中的 `observedGeneration`。
URMA 是这里使用的底层通信接口；完成标志的回写本身也是一次通信操作。

```text
整理 KV 数据 → 提交数据传输 → 确认传输完成
    → 回写 observedGeneration = 17
    → 确认完成标志写入结束 → 清理任务槽提交标志
```

这个顺序使完成状态代表数据已准备好，而不仅仅是 Host 开始处理或已经提交了传输。
代码仍在 `ProcessCommand` 中，数据传输完成后构造独立的 `completionWrite`，
再清理 `commitGeneration`。

#### 第四步：设备检查完成状态，继续消费数据

同一次 Host 完成等待可以放在不同阶段。在 Hcomm kernel 中，代码通过
`commandEntryCount != 0 && !deferCompletion` 决定是否等待：

- 不延后等待：Hcomm kernel 读取完成序号，匹配本轮编号后再继续。
- 延后到 Attention：当 `localOnly == 0`、`tokenLayout` 成立、`scatterAfterGather == 0`
  且 `deferCompletionToAttention != 0` 时，跳过这里的完成等待，由后续 AIV 消费阶段检查。

延后路径中的 `WaitRemoteReady` 反复将完成槽从设备全局内存读入算子本地缓冲区，
执行流水屏障，再检查期望编号非零且完成编号与期望编号相等。
条件满足后，才读取远端搬来的 KV 并继续 Attention。
这是明确写在算子中的轮询循环，与前文由 Runtime 下发的专用 `EVENT_WAIT` 任务不同。
实现位于参考仓库的
`csrc/src/a5_sparse_and_tail_attention_and_remote_scatter_copy/op_kernel/a5_sfa/arch35/sparse_flash_attention_service_vector_mla.h`
中的 `WaitRemoteReady` 和 `CopyInSingleRemoteKv`。

上述分支将这次完成等待从 Hcomm 阶段移到 AIV 消费阶段，并不意味着程序中其他传输同步
也被取消。两个阶段都在 NPU 设备侧；下面将它们放在同一列，时间从上向下：

```text
NPU 设备侧                                  Host CPU 侧
    │                                            │
Hcomm 代码准备任务                               │
发送任务内容、commitGeneration ───────────→ 轮询发现新任务
    │                                      检查本轮编号
    │                                      整理 KV、提交传输
AIV 到达数据消费位置                        等待数据传输完成
读取 observed/expected                          │
不匹配则继续读取 ←──────────────────────── 回写完成序号
编号匹配后读取 KV                          清理任务槽
继续 Attention                             继续轮询
```

#### 与本项目 MoE 卸载的对应关系

这套方案可供借鉴的是交接协议：把“路由准备好”对应为任务发布，把 CPU 专家管理工作
对应为 Host 任务处理，再由明确的完成状态放行设备后续计算。
它替换的是通知和等待机制，不会自动消除 LRC 计算、权重传输或映射更新。
迁移时必须明确完成标志代表 CPU 规划结束还是权重与映射已经就绪，
并重新核对共享状态、内存可见性、设备资源和跨流依赖。
参考代码中的 KV 布局、Hcomm/URMA 通信和 Host 服务组织不能直接套用到 MoE。

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
