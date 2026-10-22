# MoE 卸载 MTE 交接改造：修改内容与完整执行流程

本文说明截至 2026-10-22 的实现，对应代码基线 `db76c76f`。
当前可选 MTE 后端已跑通一次真实模型的短请求测试；关闭诊断后的 callback/MTE 同轮对比正在进行，
尚不能认定已获得性能收益。本文描述当前实现，不把尚未验证的推测写成结论。

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

- **CPU 提交线程**：执行模型的 Python 代码，在捕获时构造图，重放时提交图执行。
- **CPU 工作线程**：本次新增的一个 C++ 常驻线程，轮询任务槽，执行预取和按需回调。
  空闲轮询不持有 Python GIL；调用 Python 回调时获取 GIL。不是每层新建一个线程。
- **CPU 监控线程**：额外的 Python 线程，检查长时间未完成的任务，负责诊断与失败退出，不计算专家选择。
- **计算流／预取流**：NPU 上的有序任务队列。各自先执行发布算子，再执行等待算子；
  当前流的后续任务必须等等待算子退出。其他流能否推进，还取决于流依赖和设备资源。
- **加载流 `load_stream`**：CPU 回调向它提交权重传输，随后等待该流完成。
  `SyncLaunchStream` 的“同步提交”表示绕过 Torch 的 Host 任务队列，不表示每次拷贝自动完成；
  实际 H2D 完成仍由 `_synchronize_h2d()` 保证。

**AIV** 是 NPU 的向量计算单元。本方案的发布、等待 kernel 都在设备侧运行，使用 AIV 类型。
**MTE** 是设备的数据搬运引擎；这里的 `AscendC::DataCopyPad` 用于访问控制信号。
大体积专家权重仍通过现有 Torch 搬运路径，不是由这个等待 kernel 搬运。

### 2.2 两类内存与三种编号

控制信号放在注册的 **Host 内存**中，CPU 用 Host 地址访问，NPU 用注册返回的设备别名访问。
这不是 CPU、NPU 各保存一份需要手动对拷的信号副本。

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

## 3. 按八步理解当前完整流程

### 第一步：初始化加载资源和常驻工作线程

`ExpertOffloadManager` 根据后端配置创建加载流。模型权重准备、热专家预加载完成后，
`_finalize_offload` 创建 `MteHostDispatcher`，此时还没开始正式的图捕获。

Dispatcher 创建 NPU `states` 数组，并通过 Host 扩展创建 `Session`。ACL 后端先使用
`aclrtMallocHost` 分配 Host 内存，再对其中对齐的区域调用 `aclrtHostRegister`，获得设备别名。
因此 `aclrtHostRegister` 在这里负责注册映射，不是单独承担内存分配。

当前预分配 2048 个交接槽位，每槽位 1024 字节：request 在偏移 0，completion 区在偏移 512。
注册好所有槽位的 Python 入口后，启动一个 C++ 工作线程，它进入对应 ACL context 后开始轮询。
槽位 0 用于一次初始化握手，其余槽位留给捕获的真实回调；同时启动监控线程。

### 第二步：捕获图时，为每个回调记录固定的发布／等待节点

捕获执行到外层 `update_weights` 时，CPU 仍会执行 Python：提交路由等数据的 D2H，
准备参数，然后进入 `_launch_offload_callback`。MTE 分支调用 `MteHostDispatcher.launch`。

`launch` 为这个捕获的回调分配一个新交接槽位，将函数与参数引用保存到 `entries[slot]`。
它在前面的复制、事件任务提交之后读取 `stream.npu_stream`，随后通过 `ctypes` 调用独立动态库的
`host_sync_publish_launch` 和 `host_sync_wait_strict_launch`，把两个算子提交到同一条当前流中。
当前流处于捕获状态，因此这两个算子成为图节点。

按需路径保存 `_update_weights_guarded`；预取路径保存 `_prefetch_host_cb`。
不同捕获位置分别保留参数，避免预取和按需任务互相覆盖。这里只记录调用入口和参数引用，
不会把 Python 回调函数编译成 NPU kernel。

### 第三步：重放图，NPU 先计算本轮路由并回传给 CPU

正式推理重放图时，CPU 不再逐行重跑已捕获的外层 `update_weights`，
也不再为这些图节点分配交接槽位。NPU 按图中的操作和依赖执行。

按需路径先取得本层实际路由结果，再执行已捕获的 `topk_ids_h.copy_`、映射回传等操作。
预取路径先预测下一层可能需要的专家，再回传预测结果，通过原来的事件依赖衔接预取流。

地址固定，内容随每次推理变化。CPU 回调随后读取的是这些固定 Host 缓冲区中的本轮结果。
发布算子排在这些前置操作之后，因此不应在路由回传尚未完成时通知工作线程处理该任务。

### 第四步：发布算子写入本轮 request，等待算子开始等 done

`host_sync_publish` 读取该槽位 NPU `states` 中的旧序号，加一得到本轮 generation，
写回设备状态，并通过 MTE 把 8 字节 request 写到注册 Host 内存。

紧随其后的 `host_sync_wait` 读取本轮期望序号，不断通过 MTE 读取 Host completion 区，
直到观察到 `done_generation` 等于本轮序号且没有对应错误，才退出。
控制区访问使用禁用 L2 缓存的提示，以及 MTE/标量流水间的完成屏障；
跨 CPU/NPU 的实际可见性此前已在目标机器握手测试中验证，不能仅由缓存提示推导通用保证。

此时 NPU 正在执行一个等待 kernel，而不是把程序控制权“跳到 CPU”。
CPU 工作线程独立运行，双方通过内存信号配合。等待 kernel 本身会占用设备执行资源，
这正是它与原 Runtime 回调等待方式的重要区别之一。

### 第五步：CPU 轮询发现请求，调用原卸载回调

C++ `Session::poll` 逐槽位检查 request。发现非零且尚未完成的 generation 后，
将状态设为 Running，取得 GIL，调用该槽位绑定的 Python 入口 `MteHostDispatcher._run`。

`_run` 取出 `entries[slot]` 中的函数和参数，执行 `_update_weights_guarded` 或
`_prefetch_host_cb`，最终进入内层 `_update_weights`。它们仍是 Python 函数，
不是此次改造新写的一套 C++ 专家选择逻辑。

工作线程串行执行这些回调；没有任务时使用 `std::this_thread::yield()` 让出调度机会。
当前没有按层创建线程，也没有为预取、按需分别创建两个计算 worker。

### 第六步：CPU 选择专家，提交实际权重加载，并等待加载完成

内层 `_update_weights` 沿用原来的缓存逻辑：根据路由判断命中与缺失，按对应路径调用
`observe`、`choose_victims` 等策略，决定需要加载的专家及缓存槽位，构造 H2D 任务并维护映射。
预取使用预测信息，其策略调用并不必然与按需路径逐项相同；例如当前统计中预取没有 observe 耗时。

`load_submit` 表示 CPU 构造、提交这些搬运任务及相关处理的过程，**不等于传输已经完成**。
实际搬运在加载流上异步执行。权重任务、量化参数任务及必要的 scale 刷新提交后，
`_synchronize_h2d()` 等待加载流完成；随后才可以允许回调正常返回。

为解决真实模型中的量化参数 Tensor 拷贝阻塞，MTE 路径使用以下适配：

1. 检查 scale 两端的 dtype、shape、字节数，要求目标为 ND 且布局无重叠、无空洞。
2. 源、目标物理排列一致时，直接取带正确 `storage_offset` 的字节视图。
3. 排列不一致时，在 pinned CPU 张量中按目标 stride 准备内容，再取得字节视图。
4. 继续通过 Torch Storage `copy_` 提交 H2D；临时源存储保留到同步成功后释放。

这保持原 Tensor `copy_` 的逻辑结果，但准备过程可能增加 CPU 开销。它不是跳过 scale，
也不是把尚未完成的传输标成完成。当前 ND 限制和边界检查仍保留。

### 第七步：CPU 发布完成序号，NPU 等待算子退出

Python 回调成功返回后，C++ worker 在 Host completion 区更新成功状态，
以 release 原子写发布 `done_generation`。注意，done 由 C++ 会话在回调返回后发布，
不是外层 `update_weights` 在提交时写入。

NPU 等待算子观察到本轮 done 后记录结果并退出，当前流才继续执行后面的图节点。
这一步没有“返回 `_launch_host_func`”：MTE 分支没有调用它。
返回链是 Python 回调 → Dispatcher → C++ worker；设备侧则独立从等待节点继续。

### 第八步：映射写回，继续专家计算或结束预取

按需路径接着执行已捕获的 `log2phy.copy_(log2phy_h, ...)`，把专家逻辑 ID 到物理权重槽位的
映射写回 NPU，随后按原事件、预测和计算依赖继续运行。专家计算使用本轮映射和已加载权重。

预取路径接着写回下一层映射，并记录预取完成事件。后续按需流程按原逻辑与该事件衔接，
确认实际路由是否命中预取结果；预测不完整时仍需补齐缺失专家。
预取成功不代表可以省掉下一层的实际路由判断。

同一张图下次重放时，再读取新路由，复用同一个交接槽位，generation 从 g 变为 g+1。
新的 done 必须匹配 g+1，上一轮的 done=g 不会使它提前放行。

## 4. 把上述流程放到一条时间线上

下面以单个按需交接说明顺序，省略并行的其他模型任务：

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

CPU 提交线程可能已去做其他工作，也可能因模型后续依赖等待事件；它不负责轮询 request。
预取的交接发生在预取流，前后由原事件连接到计算流，使用同一个 CPU worker 和同一套协议。

## 5. 一个具体例子

假设当前图中的某个预取回调绑定交接槽位 2，上次完成序号是 6。
本次预测下一层需要专家 139，且它不在 NPU 缓存中，策略决定放入权重槽位 20。

NPU 先回传预测结果，再由发布算子写 `request[2]=7`。CPU worker 发现第 7 次请求，
读取预测结果，准备专家 139 的 w13、w2、量化 scale，向加载流提交，等待完成。
回调成功后写 `done[2]=7`，NPU 等待算子放行，执行映射写回，使逻辑专家 139 对应权重槽位 20。

这里交接槽位 2、专家 ID 139、权重槽位 20 各有不同用途。下一次重放仍可使用交接槽位 2，
但预测专家和最终权重槽位都可以变化；图固定的是操作与缓冲区地址，不是专家选择结果。

## 6. 故障处理、诊断与资源释放

默认推理使用严格等待：不能因为轮询次数达到某个上限就继续使用旧权重。
回调抛异常，或原 guarded 回调记录了失败，Dispatcher 会终止当前引擎进程。
监控线程每秒检查状态，发现请求持续未完成约 10 秒时输出一次线程栈和槽位信息，
持续超过约 120 秒时退出引擎。这些保护依赖 Host 线程仍能得到调度，不是硬实时超时保证。

显式 `--diagnose-load-stages` 会在前 8 个非空回调加载批次中增加逐项等待，输出
BEGIN、SUBMITTED、COMPLETE，定位第一个未完成的操作。它会改变重叠关系，不用于性能对比。
正式对比不加此参数。MSTX 与回调计时沿用原包装，仍可观察内层 `_update_weights` 和各子阶段。

正常关闭时先同步设备，确保图不再访问控制地址，再停止监控线程、结束设备访问租约、
停止 worker、注销 Host 映射并释放内存。不能在图仍等待或访问控制区时先释放它。

## 7. 已验证结果与当前边界

独立测试已经覆盖 Host worker、注册内存、MTE 握手、图重放，以及实际路由/权重/映射传输链路。
真实模型最初卡在第 1 层预取；分阶段诊断证明大权重拷贝完成，随后 scale Tensor 拷贝无法完成。
改为支持转置布局的 Storage 拷贝后，`moe_host_sync_ecf9e013` 完成预热和 3 个测量请求，
510 次回调无错误，生成文本摘要与旧 callback 基线一致，profile 保存成功。

这说明当前短请求的数据流已跑通，但尚未从原生栈或设备调度证据证明原 Tensor 拷贝阻塞的完整底层原因。
同样，不能仅凭这一次短请求推广到全部输入、模型和设备。

本次 MTE 请求均值 3365.015 ms，旧 callback 基线 3348.679 ms，尚无整体加速证据。
load_prepare 变长，需要进一步区分布局准备、检查和其他因素；当前同轮对比正在运行，
先保留最终实现，取得数据后再决定是否优化。比较的是完整 MTE 方案，不是纯同步原语的孤立耗时。

## 8. 对照代码阅读

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
