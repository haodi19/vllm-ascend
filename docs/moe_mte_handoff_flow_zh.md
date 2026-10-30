# MoE 卸载 MTE 交接改造：修改内容与完整执行流程

> 当前进度（2026-10-30）：**第12节介绍当前已通过有限请求验证的阻塞规避方案**：
> 预取请求仍提前发布，把严格AIV等待和映射写回放到消费者汇合处。
> 保留原始权重拷贝，不新增CPU重排或scale的Storage字节适配，不限核、不替换成报告回调。
> 已有callback/MTE输出一致性对照，用户已能查看最新两组profile；尚不能宣称底层阻塞机制完全查明或获得稳定加速。

第1–11节保留历史说明：本文原于2026-10-22记录早期实现，2026-10-29从
`D:\华为\nano2\vllm-ascend\docs`复制，随后补入旧阻塞基线的诊断记录。
其中CPU重排、scale Storage适配和旧性能结果属于当时的方案，**不能当作当前worker-poll方案**。
旧轮次profile缺失也不代表当前轮次profile不可用。新方案的流程、代码位置和验证边界以第12节为准；
完整实验演进见[重新审视与实测记录](moe_mte_reassessment_20261029_zh.md)。

原流程与入图基础见 [CPU/NPU 卸载流程说明](moe_offload_cpu_npu_flow_zh.md)；
实验命令和联调记录见 [MTE 接入说明](moe_host_sync_integration_zh.md)。

## 1. 修改目标与范围

原方案在图内使用 `_launch_host_func` 安排 CPU 回调：设备流到达回调节点后，
由报告线程执行权重加载，回调结束后恢复该流的后续任务。本次尝试用设备算子和
本地共享控制内存完成同样的交接，研究能否减少这部分开销。

当前不是简单地把一个函数名替换成另一个函数名。CPU 执行线程、设备等待方式改变后，
真实模型暴露了量化参数拷贝的兼容性问题，因此最终方案也包括加载流和 scale 搬运的适配。

| 部分 | 默认 callback 路径 | 可选 MTE 路径 |
| --- | --- | --- |
| 图内交接入口 | `_launch_host_func` | `MteHostDispatcher.launch` |
| 执行卸载回调的线程 | Torch/CANN 报告线程 | 新建的常驻 C++ 工作线程 |
| 设备等待方式 | 原 Runtime 回调机制 | AIV 算子读取完成序号并轮询 |
| 权重加载流 | 普通 `Stream` | `SyncLaunchStream` |
| 量化参数搬运 | 原 Tensor `copy_` | 校验布局后使用 Storage 字节拷贝，必要时 CPU 重排 |
| LRC 与专家选择 | 原 `_update_weights` | 仍调用原 `_update_weights`，不是重写一套算法 |
| 默认开关 | `host_sync_backend=callback` | 显式指定 `host_sync_backend=mte` |

本轮范围为单卡、Ascend950、torch_npu 2.10、Torch H2D。图内预取与按需加载均接入。
普通 eager 执行仍直接调用原回调函数；MTE 配置下的加载流、量化参数适配也会用于初始化和 eager 加载。
原 Python/C++ LRC 后端仍保留；当前交接对比脚本让两组都使用 Python LRC，以避免混入 LRC 后端差异。

## 2. 阅读流程前需要知道的角色与数据

### 2.1 线程和设备流

- **CPU 提交线程（原有，沿用）**：执行模型的 Python 代码，在捕获时构造图，重放时提交图执行。
  两版都有这个角色；它与执行卸载回调的线程不是同一个角色。
- **CPU 报告线程（原 callback 使用，MTE 卸载交接不再使用）**：原路径通过
  `_subscribe_report(stream)` 订阅报告，由 Torch/CANN 报告线程处理回调通知、执行卸载回调。
  MTE 单卡按需和预取入口跳过该订阅，改由下面的工作线程执行卸载回调。
  本次没有删除 Torch/CANN 的报告线程实现，也没有全局关闭报告线程；其他功能仍可能使用，
  切回 callback 后端也仍使用原机制。因此不能说整个进程已经没有报告线程。
- **CPU 工作线程（MTE 新增）**：一个 C++ 常驻线程，轮询任务槽，执行预取和按需回调。
  空闲轮询不持有 Python GIL；调用 Python 回调时获取 GIL。不是每层新建一个线程。
- **CPU 监控线程（MTE 新增）**：额外的 Python 线程，检查长时间未完成的任务，
  负责诊断与失败退出，不计算专家选择；原 callback 路径不启动本次新增的监控线程。
- **计算流／预取流（原有，沿用）**：NPU 上的有序任务队列，原来的流间事件依赖保留。
  callback 版在交接位置使用 Runtime 回调机制；MTE 版在对应位置新增发布、等待算子。
  MTE 当前流的后续任务必须等等待算子退出。其他流能否推进，还取决于流依赖和设备资源。
- **加载流 `load_stream`（原有角色，MTE 改变流类型）**：两版都由 CPU 回调向它提交权重传输，
  随后等待该流完成。callback 仍创建普通 `Stream`，MTE 改为 `SyncLaunchStream`，
  不是在原加载流旁边额外再放一条承担同样职责的流。
  `SyncLaunchStream` 的“同步提交”表示绕过 Torch 的 Host 任务队列，不表示每次拷贝自动完成；
  实际 H2D 完成仍由 `_synchronize_h2d()` 保证。

这里被替换的是卸载交接处的报告回调路径，没有删除原后端或上述原有设备流。

**AIV** 是 NPU 原有的向量计算单元；新增的是发布、等待 kernel，不是新增硬件。
本方案的这两个 kernel 都在设备侧运行，使用 AIV 类型。
**MTE** 是设备的数据搬运引擎；这里的 `AscendC::DataCopyPad` 用于访问控制信号。
大体积专家权重仍通过现有 Torch 搬运路径，不是由这个等待 kernel 搬运。

### 2.2 两类内存与三种编号

控制信号放在注册的 **Host 内存**中，CPU 用 Host 地址访问，NPU 用注册返回的设备别名访问。
这不是 CPU、NPU 各保存一份需要手动对拷的信号副本。
这块控制区、NPU `states` 和 Python `entries` 都是 MTE 新增的数据结构。
原 callback 路径依靠 Runtime 管理回调通知和完成依赖，不使用本项目新增的 generation 协议。
专家权重、路由缓冲区和 `log2phy` 则是原有数据；MTE 只在必要时新增量化参数的 CPU 重排临时存储。

| 数据 | 存放位置 | 谁写、谁读 |
| --- | --- | --- |
| `request_generation` | 注册 Host 内存 | NPU 发布算子写，CPU 工作线程读 |
| `done_generation` | 注册 Host 内存 | CPU 成功完成回调后写，NPU 等待算子读 |
| `error_generation`、状态 | 注册 Host 内存 | C++ 会话写，诊断与错误处理读取 |
| 每槽位 `states`，4 个 64 位字段 | NPU 内存 | 发布算子更新本轮序号，等待算子记录结果、观察值和轮询次数 |
| `entries[slot]` | CPU Python 对象 | 捕获时存回调与参数引用，CPU 工作线程执行时读取 |
| 专家权重、路由、`log2phy` | 各自原有 CPU/NPU 缓冲区 | 沿用原卸载逻辑，与控制信号分开 |

需要区分三个编号：**专家 ID** 标识模型里的专家；**权重槽位**标识专家驻留的 NPU 缓冲区位置；
**交接槽位**标识某个捕获回调。本方案日志中的 `slot=2 generation=1` 是交接槽位 2 的第 1 次请求，
不是专家 2，也不是第 2 个权重缓存槽。

generation 是同一交接槽位逐次递增的序号。用序号而不是一个始终为 true 的完成 flag，
可以让设备区分“上一轮完成”与“这一轮完成”。

**交接槽位是 MTE 新增的任务登记位置**，将三个对象用同一个 slot 编号关联：
Python 中的函数及参数 `entries[slot]`、Host 控制区、NPU 中的 `states[slot]`。
例如，同一层的预取和按需交接分别分配槽位；不同捕获图中的交接位置也分别分配。
下一次重放复用这个槽位，不按请求重新分配。原 callback 模式由 Runtime 管理自己的回调任务，
没有本项目新增的这张交接槽位表。

**request 区**是“设备通知 CPU 开始任务”的控制内存区域，首字段是 `request_generation`。
**completion 区**是“CPU 通知设备任务完成或失败”的区域，包含 `done_generation`、
`error_generation` 和执行状态。这两个名称指内存区域，不是线程，也不是模型请求文本。

NPU 张量 `states` 的形状为 `[2048, 4]`，每个槽位有四个 64 位字段：

| 字段 | 含义 | 写入时机 |
| --- | --- | --- |
| `states[slot, 0]` | 当前 generation，即设备本轮期望完成的序号 | 发布算子读取旧值并加一 |
| `states[slot, 1]` | 等待结果码：0 表示刚发布，1 成功，2 回调失败，3 轮询到限，4 非法序号 | 发布时初始化，等待退出时记录；2/3 用于有界测试等待 |
| `states[slot, 2]` | 等待算子最后读到的 done 序号 | 等待退出时记录 |
| `states[slot, 3]` | 本轮等待算子的轮询次数 | 等待退出时记录 |

所以不能说“所有状态都在 CPU 内存”，也不能把 `states` 当成 CPU 要轮询的 flag。
**跨 CPU/NPU 通知用的 request/done 信号都在注册 Host 内存；NPU 内存中的 `states`
保存设备自己的期望序号和等待结果。**它不是独立的“CPU 写入、NPU 轮询”的完成信号副本。
正式推理的严格等待遇到错误不会放行，结果记录也不保证在错误情况下执行，由 Host 监控负责终止进程。

### 2.3 调度对象与新增算子

**Python 类 `MteHostDispatcher`** 是 MTE 交接的管理器，不是线程。其实例持有交接槽位表、
NPU 状态张量、Host 会话和设备算子动态库。方法 `launch` 在捕获期间登记任务并提交算子；
方法 `_run` 在 CPU 工作线程上执行已登记的任务。
**C++ 类 `Session`** 经 pybind 暴露给 Python，负责注册 Host 控制内存、创建工作线程、轮询和发布完成状态。

| 名称及类型 | 功能 | 是否进入真实 MoE 图 |
| --- | --- | --- |
| AIV 算子 `host_sync_publish` | 更新 NPU 本轮序号，通过 MTE 写 Host request，通知工作线程 | 是 |
| AIV 算子 `host_sync_wait` | 通过 MTE 反复读取 Host completion，完成序号匹配才放行当前流 | 是 |
| AIV 测试算子 `host_sync_consume` | 在独立数据流测试中读取权重、映射、路由并产生校验结果 | 否；真实模型仍使用原专家计算算子 |

C 接口函数 `host_sync_publish_launch`、`host_sync_wait_launch`、`host_sync_wait_strict_launch`
是 CPU 侧的算子启动包装，不是另外三个设备算子。两个 wait 包装都启动同一个 `host_sync_wait`：
普通包装传入有限轮询次数用于初始化/测试，strict 包装传入 0 表示推理中不按轮询次数提前退出。
测试接口函数 `host_sync_consume_launch` 同样只是启动上述测试算子。

### 2.4 轮询替换了什么，仍有哪些开销

MTE 图内卸载交接不再调用 `_launch_host_func`，也不通过该处的报告队列等待、唤醒报告线程来启动工作。
设备算子写 request 后，已经存在的工作线程直接轮询发现变化；设备算子轮询 done 等待 CPU 完成。
这正是本次替换报告通知机制、尝试减少通知和唤醒成本的方向。

但“仍调用 Python 回调函数”与“仍使用 ACL 回调机制”是两回事：`_update_weights_guarded`
仍是登记后调用的任务函数，执行它的是 MTE 工作线程。MSTX 的 `host_callback` 名称沿用旧名称。

当前实现不是完全无系统调用、无锁或无中断的纯忙等：工作线程空闲时调用
`std::this_thread::yield()`，可能进入操作系统调度；检查槽位使用互斥锁，调用 Python 需要 GIL，
竞争时可能阻塞；完成任务还调用条件变量的 `notify_all()`。
设备任务提交、Host 内存分配/注册、权重 H2D 完成同步、监控线程等待等也仍可能使用 Runtime、驱动或系统调用。
因此只能说替换了卸载交接的报告回调机制，不能宣称整个推理已经没有中断或系统调用，
更不能把旧版本所有空泡都归因于中断，再认为已被消除。

## 3. 按八步理解早期完整流程（历史实现）

### 第一步：初始化加载资源和常驻工作线程

卸载管理类 `ExpertOffloadManager` 根据后端配置创建加载流。模型权重准备、热专家预加载完成后，
方法 `_finalize_offload` 创建交接管理类 `MteHostDispatcher` 的实例，此时还没开始正式的图捕获。

该 Dispatcher 对象创建 NPU 状态张量 `states`，并通过 Host 扩展创建 C++ 会话类 `Session` 的实例。
ACL 后端先调用内存分配接口函数 `aclrtMallocHost`，再对其中对齐的区域调用注册接口函数
`aclrtHostRegister`，获得设备别名。
因此 `aclrtHostRegister` 在这里负责注册映射，不是单独承担内存分配。

当前预分配 2048 个交接槽位，每槽位 1024 字节：请求控制区 request 在偏移 0，完成控制区 completion 在偏移 512。
注册好所有槽位的 Python 入口后，启动一个 C++ 工作线程，它进入对应 ACL context 后开始轮询。
槽位 0 用于一次初始化握手，其余槽位留给捕获的真实回调；同时启动监控线程。

### 第二步：捕获图时，为每个回调记录固定的发布／等待节点

捕获执行到外层方法 `update_weights` 时，CPU 提交线程仍会执行 Python：提交路由等数据的
D2H（设备到 Host 拷贝），准备参数，然后进入分派方法 `_launch_offload_callback`。
MTE 分支调用交接管理器的方法 `MteHostDispatcher.launch`。

方法 `launch` 为这个捕获的回调分配一个新交接槽位，将函数与参数引用保存到 Python 表项 `entries[slot]`。
它在前面的复制、事件任务提交之后读取流对象的句柄属性 `stream.npu_stream`，随后通过 Python 外部函数接口模块
`ctypes` 调用独立动态库的 C 启动函数 `host_sync_publish_launch` 和 `host_sync_wait_strict_launch`，
把发布和等待两个算子提交到同一条当前流中。
当前流处于捕获状态，因此这两个算子成为图节点。

按需路径保存带错误保护的任务函数 `_update_weights_guarded`；预取路径保存预取任务函数 `_prefetch_host_cb`。
不同捕获位置分别保留参数，避免预取和按需任务互相覆盖。这里只记录调用入口和参数引用，
不会把 Python 回调函数编译成 NPU kernel。

### 第三步：重放图，NPU 先计算本轮路由并回传给 CPU

正式推理重放图时，CPU 提交线程不再逐行重跑已捕获的外层方法 `update_weights`，
也不再为这些图节点分配交接槽位。NPU 按图中的操作和依赖执行。

按需路径先取得本层实际路由结果，再执行已捕获的张量拷贝方法 `topk_ids_h.copy_`、映射回传等操作；
其中 Host 张量 `topk_ids_h` 存放回传的专家 ID。
预取路径先预测下一层可能需要的专家，再回传预测结果，通过原来的事件依赖衔接预取流。

地址固定，内容随每次推理变化。CPU 回调随后读取的是这些固定 Host 缓冲区中的本轮结果。
发布算子排在这些前置操作之后，因此不应在路由回传尚未完成时通知工作线程处理该任务。

### 第四步：发布算子写入本轮 request，等待算子开始等 done

设备发布算子 `host_sync_publish` 读取该槽位 NPU 状态张量 `states` 中的旧序号，加一得到本轮序号 generation，
写回设备状态，并通过 MTE 把 8 字节 request 写到注册 Host 内存。

紧随其后的设备等待算子 `host_sync_wait` 读取本轮期望序号，不断通过 MTE 搬运引擎读取 Host 完成控制区 completion，
直到观察到完成序号字段 `done_generation` 等于本轮序号且没有对应错误，才退出。
控制区访问使用禁用 L2 缓存的提示，以及 MTE/标量流水间的完成屏障；
跨 CPU/NPU 的实际可见性此前已在目标机器握手测试中验证，不能仅由缓存提示推导通用保证。

此时 NPU 正在执行一个等待 kernel，而不是把程序控制权“跳到 CPU”。
CPU 工作线程独立运行，双方通过内存信号配合。等待 kernel 本身会占用设备执行资源，
这正是它与原 Runtime 回调等待方式的重要区别之一。

### 第五步：CPU 轮询发现请求，调用原卸载回调

C++ 会话方法 `Session::poll` 在前述常驻 CPU 工作线程上逐槽位检查请求区 request。
发现非零且尚未完成的序号 generation 后，将执行状态设为 Running（处理中），
取得 Python 解释器锁 GIL，调用该槽位绑定的 Python 调度方法 `MteHostDispatcher._run`。

方法 `_run` 取出任务表项 `entries[slot]` 中的函数和参数，执行按需任务函数 `_update_weights_guarded` 或
预取任务函数 `_prefetch_host_cb`，最终进入内层方法 `_update_weights`。它们仍是 Python 函数，
不是此次改造新写的一套 C++ 专家选择逻辑。

工作线程串行执行这些回调；没有任务时调用 C++ 调度让出函数 `std::this_thread::yield()`。
当前没有按层创建线程，也没有为预取、按需分别创建两个计算 worker。

### 第六步：CPU 选择专家，提交实际权重加载，并等待加载完成

内层方法 `_update_weights` 沿用原来的缓存逻辑：根据路由判断命中与缺失，按对应路径调用
策略方法 `observe`（记录专家使用信息）、`choose_victims`（选择被替换专家），
决定需要加载的专家及缓存槽位，构造 H2D（Host 到设备）任务并维护映射。
预取使用预测信息，其策略调用并不必然与按需路径逐项相同；例如当前统计中预取没有 observe 耗时。

计时段 `load_submit` 表示 CPU 构造、提交这些搬运任务及相关处理的过程，**不等于传输已经完成**。
实际搬运在加载流上异步执行。权重任务、量化参数任务及必要的 scale 刷新提交后，
同步方法 `_synchronize_h2d()` 等待加载流完成；随后才可以允许回调正常返回。

为解决真实模型中的量化参数 Tensor 拷贝阻塞，MTE 路径使用以下适配：

1. 检查 scale 两端的 dtype、shape、字节数，要求目标为 ND 且布局无重叠、无空洞。
2. 源、目标物理排列一致时，直接取带正确 `storage_offset` 的字节视图。
3. 排列不一致时，在 pinned CPU 张量中按目标 stride 准备内容，再取得字节视图。
4. 继续通过 Torch Storage `copy_` 提交 H2D；临时源存储保留到同步成功后释放。

这保持原 Tensor `copy_` 的逻辑结果，但准备过程可能增加 CPU 开销。它不是跳过 scale，
也不是把尚未完成的传输标成完成。当前 ND 限制和边界检查仍保留。

**为什么不能直接沿用原来的 Tensor 拷贝路径。**原 callback 版由报告线程在 Runtime 的回调协议下执行任务；
MTE 版换成了独立工作线程，并且设备流中有一个尚未退出的 AIV 等待算子。
从 CPU 看似乎只是换了任务入口，但线程上下文、Host 提交队列和设备资源等待条件已经改变。

原来的普通加载流可能先把操作排入 Torch Host 任务队列。MTE 的工作线程需要在设备等待期间把
H2D 真正提交出去，不能依赖被其他等待占住的提交环节，所以改用直接提交的流类型 `SyncLaunchStream`。
这解决的是提交方式的兼容要求，不等于已证明原来所有阻塞都由 Host 队列造成。

实际联调中，换流后仍有阻塞。逐项等待日志显示：第一处出问题的预取加载中，
8 MiB 的 w13 和 4 MiB 的 w2 任务均完成；512 KiB 的 `w13_weight_scale` 任务已提交，
但等待其完成没有返回。后面的 scale 刷新尚未执行，因此不能归因于刷新操作。
量化参数使用张量级 `Tensor.copy_`，而权重已有底层 Storage 搬运路径，这使量化拷贝成为明确的排查点。

张量级拷贝要求保持逻辑元素对应关系。即使形状和类型相同，源、目标 stride（物理排列步长）不同，
也不能简单逐字节复制；后端可能安排额外的布局转换、临时数据处理或同步。
若其中有工作依赖正被设备等待算子占住的资源或流，就可能形成
“NPU 等 CPU 完成 → CPU 等拷贝完成 → 拷贝依赖尚不能执行的设备工作”的等待环。
这是与本次执行模型变化相符的机制解释，**尚未由原生调用栈或设备调度证据证明为这次阻塞的唯一根因**。
不能据此断言 Tensor 拷贝一定发出了某个具体转换 kernel，或已经确认就是 AIV 资源耗尽。

当前适配函数 `storage_quant_copy` 把必要的布局准备提前放在 CPU：目标 ND 布局满足检查后，
若源布局不同，就调用 CPU 分配函数 `torch.empty_strided(..., pin_memory=True)` 和 CPU 张量方法
`packed.copy_(source)`，按目标 stride 重排；随后构造原始字节视图并走 Storage 搬运。
这样缩小了 H2D 阶段需要后端处理的张量语义范围，也保留了原拷贝的逻辑结果。
这条路径在目标机器上消除了已复现的阻塞，但不能从“换路径后通过”反推所有底层细节都已证实。

因此，这是 MTE 改变执行条件后引入的兼容性改动，不是原版普遍需要修复，也不是 generation 协议必然要求的步骤。
其代价包含每任务的布局检查、字节视图构造，以及布局不一致时的 pinned 内存分配和 CPU 重排。
这些都计入任务准备段 `load_prepare`；现有计时尚不能拆出每一项的独立耗时。

### 第七步：CPU 发布完成序号，NPU 等待算子退出

Python 任务函数成功返回后，C++ 工作线程在 Host 完成控制区 completion 更新成功状态，
以 release 内存序的原子写发布完成序号字段 `done_generation`。
注意，done 由 C++ 会话在回调返回后发布，不是外层方法 `update_weights` 在提交时写入。

NPU 等待算子观察到本轮 done 后记录结果并退出，当前流才继续执行后面的图节点。
这一步没有“返回回调提交函数 `_launch_host_func`”：MTE 分支没有调用它。
返回链是 Python 回调 → Dispatcher → C++ worker；设备侧则独立从等待节点继续。

### 第八步：映射写回，继续专家计算或结束预取

按需路径接着执行已捕获的张量拷贝方法 `log2phy.copy_(log2phy_h, ...)`，其中 `log2phy_h`
是 Host 映射张量，`log2phy` 是 NPU 映射张量；该操作把专家逻辑 ID 到物理权重槽位的
映射写回 NPU，随后按原事件、预测和计算依赖继续运行。专家计算使用本轮映射和已加载权重。

预取路径接着写回下一层映射，并记录预取完成事件。后续按需流程按原逻辑与该事件衔接，
确认实际路由是否命中预取结果；预测不完整时仍需补齐缺失专家。
预取成功不代表可以省掉下一层的实际路由判断。

同一张图下次重放时，再读取新路由，复用同一个交接槽位，generation 从 g 变为 g+1。
新的 done 必须匹配 g+1，上一轮的 done=g 不会使它提前放行。

## 4. 把上述流程放到一条时间线上

下面两图均以单个按需交接的图重放说明顺序，省略其他模型任务和 Runtime 内部细分步骤。
纵向表示先后，不表示各阶段耗时比例。图中的通知箭头是线程／设备间协作，不是同一线程跳转。

### 4.1 原 callback 版

捕获时 `_launch_host_func` 将回调交接记录到图中，并由原订阅机制准备报告处理。
重放时，设备到达该节点后通过 Runtime 报告机制通知 CPU 报告线程：

```text
CPU 提交线程        NPU 计算流                 CPU 报告线程           NPU 加载流
提交 graph replay
                    路由计算
                    路由 D2H
                    到达回调交接节点 ──────→  接收、处理 Runtime 报告
                    阻塞流后续任务            执行 _update_weights
                    │                         选择专家、准备任务
                    │                         提交 H2D ───────────→  权重与 scale 搬运
                    │                         等 load_stream 完成 ←─ 完成
                    │                         回调返回
                    Runtime 完成机制放行 ←──  报告处理结束、完成通知
                    映射 H2D
                    专家计算及后续图节点
```

这条路径没有本次新增的 request/done 控制区，也没有本次新增的 AIV 轮询等待算子。
报告线程如何等待通知、Runtime 如何解除设备侧依赖，详见原 `_launch_host_func` 流程文档。

### 4.2 早期 MTE 版（紧接发布便等待）

捕获时交接位置记录发布和等待算子；重放时通过共享控制区通知新增工作线程：

```text
CPU 提交线程        NPU 计算流                 CPU 工作线程           NPU 加载流
提交 graph replay
                    路由计算
                    路由 D2H
                    publish：request=g  ───→  轮询发现 g
                    wait：等 done=g           执行 _update_weights
                    │                         选择专家、准备任务
                    │                         提交 H2D ───────────→  权重与 scale 搬运
                    │                         等 load_stream 完成 ←─ 完成
                    │                         回调返回、写 done=g
                    wait 读到 done=g ←──────  worker 继续轮询
                    映射 H2D
                    专家计算及后续图节点
```

两版中，CPU 提交线程都可能已去做其他工作，也可能因模型后续依赖等待事件。
MTE 版由新增 worker 轮询 request，提交线程不参与这项轮询；监控线程不在正常交接链上，故未画入图中。

共同顺序是：路由回传 → CPU 加载专家 → 等待实际 H2D 完成 → 映射写回 → 后续计算。
改变的是中间的通知、执行回调的线程及设备等待方式，同时包含前述加载流和量化参数适配。
两版的预取交接都发生在预取流，前后由原事件连接到计算流；MTE 预取与按需共用同一个 CPU worker。

## 5. 精简流程与调用链

初始化时，交接管理对象 `MteHostDispatcher` 创建 Host 会话、控制内存、NPU 状态张量以及一个常驻 CPU 工作线程。
捕获时，CPU 提交线程把任务函数及参数登记到交接槽位，在当前设备流中记录发布和等待两个 AIV 算子。
重放时，NPU 先回传路由，再由发布算子递增 `states` 中的本轮序号，并将序号写到 Host request。
CPU 工作线程轮询发现新序号，执行登记的卸载函数，选择专家、提交 H2D 并等待实际完成；
之后把相同序号写到 Host completion 的 done 字段。设备等待算子通过 MTE 读到匹配的 done 后退出，
当前流才能写回映射并继续。两端始终各自执行，通过内存序号交接，不是 NPU 开始执行 Python。

捕获期间的提交链如下；箭头表示同一提交线程上的函数调用：

```text
CPU 提交线程
  按需：外层方法 update_weights
    → 分派方法 _launch_offload_callback
    → 管理器方法 MteHostDispatcher.launch(stream, _update_weights_guarded, args)
      → 登记 entries[slot]，计算 Host 控制区及 NPU states 地址
      → C 启动函数 host_sync_publish_launch → 将发布算子记录进当前图
      → C 启动函数 host_sync_wait_strict_launch → 将等待算子记录进当前图
    → 继续提交映射 H2D 等后续图节点
```

预取也经分派方法进入 `launch`，但登记的是预取任务函数 `_prefetch_host_cb`，使用预取流及相应参数。
下面是图重放后，由工作线程独立触发的执行链；不是 `launch` 同步调用了这些函数：

```text
NPU 发布算子：写 request_generation=g
  ↓ 共享内存可见，工作线程轮询发现
CPU 工作线程：C++ Session::run → Session::poll
  → 捕获前注册的槽位入口 lambda
  → Python 方法 MteHostDispatcher._run(slot)
  → 按需 _update_weights_guarded / 预取 _prefetch_host_cb
  → 内层 _update_weights → 提交权重 → _synchronize_h2d 等待加载完成
  → 逐层返回到 Session::poll → 写 done_generation=g
  ↓ 设备算子通过 MTE 轮询读到
NPU 等待算子：退出 → 当前流执行映射 H2D 及后续任务
```

## 6. 一个具体例子

假设当前图中的某个预取回调绑定交接槽位 2，上次完成序号是 6。
本次预测下一层需要专家 139，且它不在 NPU 缓存中，策略决定放入权重槽位 20。

NPU 先回传预测结果，再由发布算子写 `request[2]=7`。CPU worker 发现第 7 次请求，
读取预测结果，准备专家 139 的 w13、w2、量化 scale，向加载流提交，等待完成。
回调成功后写 `done[2]=7`，NPU 等待算子放行，执行映射写回，使逻辑专家 139 对应权重槽位 20。

这里交接槽位 2、专家 ID 139、权重槽位 20 各有不同用途。下一次重放仍可使用交接槽位 2，
但预测专家和最终权重槽位都可以变化；图固定的是操作与缓冲区地址，不是专家选择结果。

## 7. 故障处理、诊断与资源释放

默认推理使用严格等待：不能因为轮询次数达到某个上限就继续使用旧权重。
回调抛异常，或原 guarded 回调记录了失败，Dispatcher 会终止当前引擎进程。
监控线程每秒检查状态，发现请求持续未完成约 10 秒时输出一次线程栈和槽位信息，
持续超过约 120 秒时退出引擎。这些保护依赖 Host 线程仍能得到调度，不是硬实时超时保证。

显式 `--diagnose-load-stages` 会在前 8 个非空回调加载批次中增加逐项等待，输出
BEGIN、SUBMITTED、COMPLETE，定位第一个未完成的操作。它会改变重叠关系，不用于性能对比。
正式对比不加此参数。MSTX 与回调计时沿用原包装，仍可观察内层 `_update_weights` 和各子阶段。

正常关闭时先同步设备，确保图不再访问控制地址，再停止监控线程、结束设备访问租约、
停止 worker、注销 Host 映射并释放内存。不能在图仍等待或访问控制区时先释放它。

## 8. 2026-10-22 的验证结果与当时边界

独立测试已经覆盖 Host worker、注册内存、MTE 握手、图重放，以及实际路由/权重/映射传输链路。
真实模型最初卡在第 1 层预取；分阶段诊断证明大权重拷贝完成，随后 scale Tensor 拷贝无法完成。
改为支持转置布局的 Storage 拷贝后，`moe_host_sync_ecf9e013` 完成预热和 3 个测量请求，
510 次回调无错误，生成文本摘要与旧 callback 基线一致，profile 保存成功。

这说明当前短请求的数据流已跑通，但尚未从原生栈或设备调度证据证明原 Tensor 拷贝阻塞的完整底层原因。
同样，不能仅凭这一次短请求推广到全部输入、模型和设备。

上述早期 MTE 请求均值 3365.015 ms，旧 callback 基线 3348.679 ms，尚无整体加速证据。
随后 `moe_host_sync_01` 的同轮对比关闭诊断，两组各预热 3 次、测量 10 次：生成文本一致，
各 1700 次计时调用无错误；callback 请求均值 3357.783 ms，MTE 为 3468.949 ms，增加约 3.31%。
按需 `load_prepare` 均值从 137.670 μs 增至 742.821 μs，不能把增量全部归为某个未单独计时的子操作。
该轮 callback 时间线可读，MTE 缺少 start/end 信息及标准位置的设备原始 data，不能据此比较交接空泡。
比较的是完整 MTE 方案，不是纯同步原语的孤立耗时。

## 9. 图捕获与模型请求的关系

图的生命周期属于模型执行器，不属于单个用户请求。当前模型执行器的方法 `capture_model`
进入图捕获流程，通常在服务初始化、模型加载及预热阶段按配置准备图；图包装层也可能在首次遇到
某种执行规格时捕获。具体时机由运行版本与图配置决定，不是每收到一个请求都先捕获一次。

当前脚本配置 `FULL_DECODE_ONLY` 和捕获尺寸 `[1, 2, 3]`，表示准备相应 Decode 执行规格的图，
不是分别为第 1、2、3 个请求捕获。图缓存还可能区分其他执行描述信息，不能据三个尺寸断言只有三张图。
Prefill、MTP 草稿步骤、未匹配规格的操作是否入图由各自配置决定；例如本脚本的 MTP 配置有
`enforce_eager=true`，不能把整个请求理解成一张统一重放的图。

一个请求包含多个模型执行步骤：Prefill 处理输入，后续 Decode/推测验证继续产生 token。
符合已捕获图规格的步骤直接重放已有图；同一张图可跨多个请求复用。
MTP 一步可以涉及多行 token，不能简单认为一个 token 就对应一次 replay。

例如服务已准备好某个三行输入的 Decode 图，请求 A 的验证步骤和之后请求 B 的兼容步骤都能重放它。
每次重放仍使用原交接槽位与地址，但路由数据变化，generation 递增；只有新捕获的图节点才经
`MteHostDispatcher.launch` 分配新的交接槽位。重放不重新执行这个 Python 方法，
但 CPU 工作线程仍需每次执行内层卸载任务。

## 10. 历史阻塞定位：旧版全 MTE 路径复现（2026-10-27）

可以这样向导师汇报：**共享内存轮询交接已经能够正常发布任务、唤起 CPU 工作线程；目前阻塞发生在 CPU 执行预取时，等待 `w13_weight_scale` 量化参数的设备任务完成。已经定位到具体操作，但它在设备侧无法完成的底层原因尚未证实。**

下面按改造流程把因果关系串起来。

**1．原回调版和共享内存版改变了什么**

原回调版通过 `_launch_host_func` 在 NPU 流中登记 CPU 回调。执行到对应位置后，由 runtime 的报告线程执行 `_update_weights`，加载所需专家；回调结束后，对应流继续向后执行。

共享内存版把这段交接替换为：

```text
NPU 执行发布算子
  → 通过 MTE 更新 CPU 可读的 request_generation
  → 常驻 CPU 工作线程轮询到新任务
  → 执行原来的 _update_weights
  → 完成权重加载后更新 done_generation
  → NPU 等待算子读到完成标志，退出等待
  → 后续映射写回、专家计算继续执行
```

其中，CPU 加载流也改成了 `SyncLaunchStream`，用于直接提交设备操作。**所以这次改造除了替换通知方式，也改变了 CPU 执行线程、设备等待节点，以及加载任务的提交环境。** 原来的 `copy_` 即使保留，也不能据此认为它的运行条件完全相同。

**2．这次具体在哪里卡住**

我们回到未加入 Storage 拷贝、CPU 重排等适配的旧提交 `942868dd`，先复现了原始阻塞，再启用逐项同步诊断。

第 0 层的加载任务正常完成。随后，第 1 层的预取任务进入 CPU 工作线程，准备将专家 139 加载到设备缓存槽位 20：

| 操作 | 数据量 | 诊断结果 |
|---|---:|---|
| 等待此前加载流任务 | — | 完成 |
| `w13` 主权重拷贝 | 8 MiB | 完成 |
| `w2` 主权重拷贝 | 4 MiB | 完成 |
| `w13_weight_scale` 量化参数拷贝 | 512 KiB | 调用返回，但随后的流同步不返回 |

因此，**这次不是 8 MiB 的 `w13` 主权重没搬完，而是其 512 KiB 的量化缩放参数所关联的设备任务没有完成。**

这里的“调用返回”只说明 CPU 从拷贝接口返回，不能说明设备操作已执行完毕。线程栈停在：

```text
预取 CPU 任务
  → _update_weights
  → 逐项加载诊断
  → w13_weight_scale 拷贝调用返回
  → SyncLaunchStream.synchronize() 长时间等待
```

**3．为什么这一处等待会导致整个请求停住**

阻塞时，共享内存中的状态是：

```text
request_generation = 1
done_generation    = 0
```

这与代码执行流程一致：

```text
CPU 等待 scale 对应的设备任务完成
  → _update_weights 无法返回
  → CPU 不会写入 done_generation
  → NPU 等待算子不能放行
  → 依赖此次预取完成的后续操作无法继续
  → 推理请求不能返回
```

超过 120 秒后，watchdog 主动终止引擎，避免后续使用未完成更新的权重。因此 HTTP 500 是阻塞后的结果。

**这不是“CPU 没有收到 NPU 通知”。** 日志已经证明工作线程进入了预取函数，且前两个权重拷贝完成了。完成标志没有更新，是因为加载函数没有执行完，不能直接解释为共享内存标志写回失败。

**4．目前对根因的判断**

最需要核实的是：**预取路径新增的 NPU 等待节点，与 scale 拷贝内部的设备操作之间，是否存在执行资源或调度依赖冲突。**

原因在于，Tensor 的 `copy_` 不一定只对应一个纯 H2D 搬运任务。如果源、目标布局不同，内部还可能包含临时张量、设备侧布局转换等操作。因此：

- 主权重的搬运可以完成；
- scale 的拷贝调用也可以返回；
- 但其内部某个设备任务可能无法在当前图与等待节点的执行环境下完成。

如果该设备任务必须等待某个被 MTE 等待节点挡住的条件，就可能形成：

```text
CPU 等设备任务完成
设备任务等某个执行条件
该条件又依赖 CPU 完成标志被写回
```

**这是当前待验证的依赖环假设，不是已经证明的死锁机制。** 现有日志还不能区分究竟是 H2D 本身、布局转换、其他隐含同步，还是具体设备调度造成的等待。

此前把预取恢复为原回调、按需保留 MTE 后，原始 Tensor 拷贝能够完成有限请求，这支持问题与**预取交接的执行环境**有关；但由于这同时改变了等待节点和执行线程，尚不能只归因于其中一个因素。

**5．初始化警告与本次阻塞的关系**

`npu_format_cast` 警告是主权重格式转换失败后走基础格式回退；随后主权重拷贝确实完成了。目前没有证据说明它直接造成这次 scale 阻塞。

`PREFILL_POOL allocated 2 layers × 256 experts` 是 Prefill 专家缓冲池的正常分配信息，也不是阻塞错误。

向导师汇报时，最准确的结论是：

> 共享内存轮询方案已经完成任务发布和 CPU 接收，但在全 MTE 预取路径中，原始 Tensor 拷贝的 `w13_weight_scale` 操作出现设备完成等待不返回。该等待使 CPU 无法发布完成标志，进而阻塞 NPU 后续执行。现在已定位到具体操作，怀疑与新增等待节点和 scale 拷贝内部设备任务的依赖有关，尚未证明具体底层依赖环。

**1．为什么改成 `SyncLaunchStream`，和原来有什么区别？**

目的是**减少 MTE 工作线程提交权重任务时，对 torch_npu 的 CPU 提交队列的依赖**。这是当时针对阻塞采取的修改，不是共享内存通信本身必需的步骤。

在启用任务队列的情况下，普通流的典型过程是：

```text
CPU 调用 copy_
  → 操作进入 torch_npu 的 CPU 任务队列
  → 后台提交线程将操作提交给 runtime
  → NPU 执行
```

`SyncLaunchStream` 的相关操作则采用直接提交方式：

```text
CPU 调用 copy_
  → 当前调用线程直接向 runtime 提交操作
  → NPU 执行
```

当时担心的情况是：MTE 工作线程需要提交权重加载任务，但提交队列的推进又受到图执行等待影响，于是尝试绕过这层队列。

**注意，名字里的 “Sync” 是提交方式，不代表每次 `copy_` 都等到数据搬完才返回。**

两种流都仍然需要：

```python
load_stream.synchronize()
```

来确认设备任务完成。修改后的主要区别如下：

| 项目 | 原普通加载流 | `SyncLaunchStream` |
|---|---|---|
| CPU 侧提交 | 启用队列时，可经后台提交线程 | 相关操作由调用线程直接提交 |
| `copy_` 返回是否代表搬完 | 不一定 | 不一定 |
| 是否仍需等待设备完成 | 是 | 是 |
| 提交开销可能计入哪里 | 部分发生在后台线程或后续等待 | 更多可能体现在调用线程的 `copy_` 时间中 |

**现在必须承认：这一修改没有解决当前阻塞，也没有证明它是必需的。** 它是为了排除提交队列依赖而加入的措施，本身也改变了计时和执行环境，后续应作为独立变量检查。

## 11. 对照代码阅读

| 文件 | 主要内容 |
| --- | --- |
| [expert_offload_manager.py](../vllm_ascend/expert_offload/expert_offload_manager.py) | 加载流选择、预取/按需分派、量化参数任务适配、临时源生命周期 |
| [host_sync.py](../vllm_ascend/expert_offload/host_sync.py) | Dispatcher、交接槽位、CPU 布局准备、监控与诊断 |
| [host_sync.cpp](../csrc/host_sync/host_sync.cpp) | pybind 会话、常驻 C++ worker、轮询与完成状态发布 |
| [host_memory.h](../csrc/host_sync/host_memory.h) | Host 内存分配、ACL 注册映射、context 与释放 |
| [protocol.h](../csrc/host_sync/protocol.h) | generation 字段、状态、对齐与偏移约定 |
| [handshake.cpp](../csrc/host_sync_device/handshake.cpp) | AscendC 发布/等待算子，MTE 控制信号访问 |
| [run_moe_host_sync.py](../script/run_moe_host_sync.py) | 两组真实推理、profile、summary 和只读日志提取 |

Host 扩展通过 pybind 暴露 `Session` 给 Python；设备算子动态库通过 `ctypes` 暴露 launch 包装。
两者职责不同：前者管理 CPU 线程与共享信号，后者向指定设备流提交 kernel。

## 12. 当前阻塞规避：提前发布预取，在消费者处严格等待（2026-10-30）

**简略版：阻塞原因、排查与修复。**

- **阻塞原因（目前确认到的层次）：** 旧MTE方案在发布预取请求后立即执行AIV严格等待，
  该等待与真实模型计算并行时，出现量化参数`w13_weight_scale.copy_()`对应的设备Transpose任务
  无法按时完成，CPU工作线程因此停在加载流同步，不能发布完成信号。对照支持提前等待的位置参与触发，
  但具体硬件资源或Runtime依赖机制尚未证实，不能直接定性为已证明的资源死锁。
- **排查依据：** 独立握手和拷贝实验能完成；真实模型日志把超时对应到scale布局转换任务。
  在映射写回均延后的对照中，延后严格等待能够完成请求，把严格等待恢复到计算门控之前则再次阻塞，
  因而将处理重点收敛到预取等待与模型计算的重叠关系。
- **修复方式：** 保留提前发布和CPU加载，将预取流的严格等待移到主计算流到达消费者门控事件之后；
  等待通过后才写回映射并通知主流继续。没有跳过权重完成检查，也没有新增CPU重排或scale Storage适配。
  当前有限输入对照及C++ LRC采集对照均完成请求、输出一致；这是已有验证的阻塞规避方案，
  不等于所有负载稳定性及底层根因均已验证。

### 12.1 问题边界与本次修改的目标

旧全MTE方案在预取流上连续提交发布算子`host_sync_publish`和严格等待算子`host_sync_wait`。
CPU工作线程收到请求后执行原权重加载；在真实模型中，量化参数`w13_weight_scale.copy_()`引出的
加载流设备任务无法按时完成，工作线程卡在加载流同步，因而不能写回完成序号。
后续Runtime日志把相应加载流任务进一步对应到Transpose布局转换算子；前面的主权重拷贝已经通过。
因此不能再把现象笼统说成“w13主权重没有搬完”，也不能只凭`copy_`返回就认为其所有设备任务已完成。

现有对照支持**预取AIV严格等待与真实模型并行执行的组合参与触发**：独立握手/拷贝图可以完成，
延后等待可以完成真实请求；在映射同样延后的条件下，把严格等待恢复到早发布之后又复现加载同步超时。
这把修改目标缩小到等待的执行位置，但尚未证明底层一定是算力资源互等、轮询访存争用或其他Runtime依赖。

本次修复的目标是改变这段重叠关系，同时保留“真实权重和映射就绪后才允许消费者继续”的条件。
它是一条经过有限验证的阻塞规避路径，不是通过跳过完成检查掩盖超时，也不是权重搬运算法重写。

### 12.2 与旧方案相比，具体改了什么

| 部分 | 旧全MTE早等待 | 当前消费者处等待 |
| --- | --- | --- |
| 预测路由回传、发布预取请求 | 提前进行 | 保持提前进行 |
| CPU工作线程、专家选择、加载流 | 原MTE工作线程和SyncLaunchStream | 保持相同角色和原始拷贝路径 |
| 预取AIV严格等待 | 紧接publish，可能与中间模型计算同时运行 | 先等主计算流到达消费者gate，再执行严格等待 |
| 预取映射H2D写回 | 原预取入口中，等待之后 | 消费者gate之后、严格等待之后 |
| 主流使用权重 | 等待预取done事件 | 仍等待done；done必须在映射写回之后 |
| 按需交接 | 主流publish→strict wait | 保持原MTE按需交接 |

当前没有追加CPU重排，没有把scale Tensor改成Storage字节搬运，也没有通过限核或有界轮询超时放行。
主权重原本就通过Storage切片拷贝；这里“没有Storage适配”专指没有新增历史方案中的scale适配，
并不表示所有权重都必须用Tensor接口。

### 12.3 新流程需要知道的对象

- 方法`MteHostDispatcher.defer_prefetch`：捕获时记录这一层的提前发布和待完成的映射写回。
- 方法`publish_for_join`：提交发布算子并返回Python令牌`token`，令牌用于找到同一交接槽位的等待信息。
  它不是传给NPU的新flag；跨CPU/NPU的信号仍是注册Host内存中的request/done generation。
- 字典`deferred_prefetch`和`deferred_waits`：捕获期间保存尚未配对的预取及等待记录，不是每轮重放都要CPU查表。
- 事件对象`gate`：记录在主计算流上，表示该流已经执行完消费者之前的任务；这是流顺序事件，
  与模型中计算专家路由的gate网络是两个概念。
- 事件对象`done`：在预取流完成严格等待和映射写回之后记录，供主流等待。
- 方法`finish_prefetch_at_join`：在消费者位置补上gate、严格等待、映射写回和done节点。

`gate`和`done`进入同一张捕获图；重放时这些依赖由NPU执行。新增的是图内事件依赖与捕获记录，
不是每次重放唤醒一个新CPU线程，也不是把完整Python流程重新执行一遍。

### 12.4 按八步串起CPU与NPU的实际流程

**第一步：初始化角色和控制内存。** 卸载管理器`ExpertOffloadManager`创建加载流`SyncLaunchStream`，
交接管理器`MteHostDispatcher`创建注册Host控制区、NPU状态张量和常驻C++工作线程。
这些机制沿用当前MTE基线；本次晚等待没有改变request/done协议或为预取额外创建线程。

**第二步：捕获时提前发布预取。** 方法`_dispatch_prefetch`在预测路由已经回传、原有ready事件已满足的位置，
调用方法`defer_prefetch`。后者通过方法`publish_for_join`登记回调、参数和交接槽位，
向原预取流提交发布算子`host_sync_publish`。此处返回，不立即提交严格等待、映射写回或done事件。
预测及预取启动位置仍然提前，不能把“晚等”理解为“晚开始搬权重”。

**第三步：在消费者位置记录主流gate。** 下一层外层方法`update_weights`调用方法
`finish_prefetch_at_join`。它在主计算流上记录事件`gate`，表示此前模型操作的完成边界。
随后在原预取流提交对`gate`的等待。这个步骤使预取AIV等待在主流到达这一位置后才具备执行条件。

**第四步：补齐预取完成依赖。** 方法`wait_for_join`使用第二步的令牌找到同一个交接槽位，
仍在原预取流提交严格等待算子`host_sync_wait`。其后调用Tensor方法`destination.copy_(source)`，
把CPU维护的`log2phy_h`映射写回该层NPU映射；然后记录事件`done`。
外层方法`update_weights`沿用原有的预取事件汇合逻辑，让主流等待`done`。
至此捕获结构为`预取流等gate→严格wait→映射写回→record done`，主流为`record gate→wait done`。

**第五步：重放时NPU和CPU并行推进。** NPU重放到第二步的发布节点，写入本轮request generation；
常驻C++工作线程轮询到变化，获取GIL，执行Python方法`MteHostDispatcher._run`，
再进入预取任务函数`_prefetch_host_cb`和内层方法`_update_weights`。
与此同时，主计算流可以执行发布位置与消费者gate之间的模型任务；预取流此时先受gate事件约束，
还没有提前进入长期驻留的AIV严格等待。

**第六步：CPU提交并确认真实权重加载。** 内层方法`_update_weights`仍按缓存策略选专家与槽位，
调用原加载路径提交主权重、scale等任务。方法`_synchronize_h2d`仍等待加载流实际完成，
包括Tensor拷贝所需的设备布局转换。CPU任务成功完成后，会话类`Session`才发布对应done generation。
启用`cpp` LRC时，仅策略的observe、评分/淘汰与状态计算进入C++，加载任务及该同步条件保持不变。

**第七步：消费者到达后检查完成并发布映射。** 主流到达gate，预取流的gate等待解除，
AIV严格等待开始检查本轮done generation。若CPU已经完成则很快通过；若仍未完成则继续严格等待。
因此等待只是换了位置，并没有被删除。只有该等待通过后，NPU才执行映射H2D并记录done事件。

**第八步：按需补齐，再执行专家计算。** 主流观察到done后，继续处理该层实际路由和按需交接。
实际需要但预取未加载的专家仍由原按需`_update_weights`处理；按需MTE路径保持publish、严格wait、
映射写回的原顺序。主流完成这些条件后才执行专家计算，随后进入下一层的预取/按需周期。

第二至第四步描述捕获时的Python建图过程，第五至第八步描述重放时的执行。
CPU完成信号可能早于或晚于主流到达gate；图内依赖在两种情况下都要求权重和映射就绪后再消费。

### 12.5 时间线与不会反向相等的依赖

```text
旧早等待（省略细节）
主计算流：路由/预测 ───── 中间模型计算 ───── 等预取done ─→ 按需/专家计算
预 取 流：等ready → publish → [AIV严格等CPU] → 映射H2D → record done
CPU worker：             选专家 → 提交权重 → 等加载流完成 → 写done generation

当前晚等待
主计算流：路由/预测 ───── 中间模型计算 ───── record gate → wait done ─→ 按需/专家计算
预 取 流：等ready → publish → wait gate ────────────────→ strict wait → 映射H2D → record done
CPU worker：             选专家 → 提交权重 → 等加载流完成 → 写done generation
加载流：                          主权重/scale H2D及原有布局转换
```

新增依赖是“主流先记录gate，再等待预取done”。因此没有把gate放到主流的wait done之后，
避免构造“主流等done、预取流等尚未记录的gate”的显式依赖环。
两个事件都在同一捕获范围内建立；代码还检查目标层、原始预取流、令牌是否重复/过期和捕获结束时是否仍有未汇合记录。
事件对象在Dispatcher中保留，防止捕获后立即失去引用。共享预取暂存区在前一任务未汇合时不允许第二个生产者覆盖。
这些检查保证修改后的顺序可以审查，不替代真实NPU上的正确性验证。

### 12.6 代码调用链与开启方式

```text
图捕获包装 ACLGraphWrapper
  → MteHostDispatcher.prefetch_join_capture()
  → 预取触发 → ExpertOffloadManager._dispatch_prefetch()
      → MteHostDispatcher.defer_prefetch()
          → publish_for_join() → _publish_task() → host_sync_publish_launch()
  → 消费者外层 ExpertOffloadManager.update_weights()
      → finish_prefetch_at_join()
          → 主流record gate → 预取流wait gate
          → wait_for_join() → host_sync_wait_strict_launch()
          → log2phy映射copy_ → 预取流record done
      → 主流wait done → 原按需交接 → 专家计算

每次重放的CPU任务：Session工作线程
  → MteHostDispatcher._run() → _prefetch_host_cb() → _update_weights()
  → 选专家/构造加载任务 → TorchCopyH2DTransport.copy_batch()
  → _synchronize_h2d() → 成功后发布done generation
```

相关实现位于[host_sync.py](../vllm_ascend/expert_offload/host_sync.py)、
[expert_offload_manager.py](../vllm_ascend/expert_offload/expert_offload_manager.py)和
[acl_graph.py](../vllm_ascend/compilation/acl_graph.py)。
需要显式选择`host_sync_backend=mte`和`host_sync_diagnose_prefetch_wait_at_join=true`；
默认后端仍为callback，单独选择mte也不自动打开晚等待。
当前实验不启用`host_sync_diagnose_prefetch_wait_before_gate`，后者是恢复早等待的诊断对照。
配置限制为单卡、现有fate预取与完整图捕获，不允许混用报告等待、有限轮询放行、限核或scale改写。

### 12.7 当前copy_submit：调用内容相同，提交上下文不同

计时段`copy_submit`包住原Torch搬运后端逐项提交任务的过程。两组都走原主权重Storage切片与
scale Tensor的`copy_`，不会因打开C++ LRC就换成C++拷贝实现。
回调版由Torch/CANN报告线程调用，加载流是普通`Stream`；MTE版由常驻工作线程调用，
加载流是`SyncLaunchStream`。

torch_npu开启Host任务队列时，普通流的异步拷贝可以先入队，由后台提交线程向Runtime下发；
SyncLaunchStream则由当前调用线程直接调用相应Runtime接口。算子提交也有相同的分支，
因此Tensor拷贝可能需要的布局转换在Host端的提交工作，也会影响该计时段。
依据为torch_npu 2.10的[异步拷贝提交实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.cpp#L80-L103)
与[算子提交实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/framework/OpCommand.cpp#L169-L212)。
服务器使用`2.10.0.post4.dev20260715`构建，上述公开源码解释提交机制，不代替对该构建每项耗时的实测归因。

所以两组`copy_submit`都是调用线程的墙钟时间，但包含的入队/直接提交工作可能不同；
它不是设备上的MEMCPY持续时间。两组都仍通过`_synchronize_h2d()`确认设备任务完成。
当前实测提交增加、后续等待减少，符合这类计时分布差异的可能性，不能据此把全部增量归因于队列，
也不能忽略提交＋等待仍存在的差异。

`SyncLaunchStream`由提交`be7b674d`引入，目的是避免MTE工作线程的权重提交依赖Torch Host队列推进。
当时怀疑图执行等待会影响该提交路径，因此绕过队列；这是一项针对提交依赖的排查措施，
并非共享内存协议本身要求，也没有因此证明原阻塞来自该队列。改流后仍曾阻塞，
当前完成请求的关键改动是预取严格等待位置后移。晚等待方案是否仍必须使用该流尚未单独对照验证。
因此可以把加载流类型列为copy_submit差异的待验证因素，不能宣称已确定它贡献了全部额外时延。

### 12.8 已验证内容和仍待确认的内容

三种输入各32个生成token的callback/MTE对照，在图调试转储开启和关闭的两轮中均输出摘要、usage一致。
最新短profile对照则为每组3次预热、1次测量，`max_tokens=5`；两组均完成请求并导出可视化。
该轮两组均为Python LRC，回调请求3185.172ms、MTE请求3229.634ms；不能说已获得端到端收益。
这些数据支持当前路径在有限负载下能推进且与基线输出一致，不覆盖所有输入、捕获尺寸和长期运行；
部分结束阶段异常与具体底层资源依赖仍未解释。

随后两组同时使用`cpp` LRC的`cpp_mte_join_profile_01`也完成请求，文本摘要与usage一致，
两份profile完整可用。预取→按需的Host回调平均间隔为115.506→93.866μs，
按需→预取为105.744→65.774μs；请求时间3196.486→3228.607ms，尚未取得端到端收益。
本轮两组均只在清理边界之后记录EngineDeadError，不能据此认定推理中发生原阻塞，
也不能仅凭合并日志顺序确定结束异常的原因。
运行入口及完整结果见[实测记录的C++ LRC对照章节](moe_mte_reassessment_20261029_zh.md#17-开启cpp-lrc后的回调与晚等待mte对照2026-10-30)。
