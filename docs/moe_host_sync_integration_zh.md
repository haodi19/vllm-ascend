# MoE 卸载中的可选 MTE 交接后端

## 当前进度

第五阶段已在绿区通过：callback 与 MTE 两组均完成 20 次图重放、21 次 Host 回调，
每轮搬运 512 KiB 权重，路由和映射各 64 字节，完整权重及消费结果检查一致。
独立验证到此结束，下一步直接运行真实 MoE 推理。

本轮已完成可选后端接入代码，尚待绿区验证实际模型。此前结果只证明独立数据流正确，
不能代替真实模型的正确性和性能结论。

首轮真实模型结果：callback 组完成 3 次测量，510 次卸载回调无错误，请求均值约 3349 ms；
停止 profiler 超时，profile 标记为不完整。MTE 组服务启动成功，但首个预热请求中
`slot=2 generation=1` 超过 120 秒仍未完成，保护机制终止引擎。这不是性能对比成功，
也不能由该日志确定是权重拷贝还是流等待卡住。

针对这个接入问题，MTE 后端在未完成请求持续约 10 秒时输出一次槽位状态、活动回调
及工作线程/主线程 Python 栈，不读取 NPU tensor，也不执行额外设备同步。
诊断块会保留在 `summary.txt`，不再被后续 HTTP 错误堆栈挤出摘要。
下一轮可只执行 `python -B script/run_moe_host_sync.py --npu 1 --backend mte`，
继续真实模型联调，不需要重跑 callback 基线。

第二轮诊断已确认：槽位 1 的真实回调完成；槽位 2 为第 1 层预取，工作线程已进入
`_update_weights`，停在 `_synchronize_h2d()` → `load_stream.synchronize()`。
主线程停在下一轮输入准备的采样结果 Event 等待。尚无 MTE 请求时延和输出一致性结果。

针对加载流与 Host 提交队列的依赖，本轮仅在 MTE 后端使用 `SyncLaunchStream` 创建权重加载流。
torch_npu 2.10 的普通流 `synchronize()` 先经 `stream()` 排空提交队列，
而 SyncLaunchStream 不启用该任务队列，相关 `stream()` 分支也跳过队列排空。
这样让回调内权重传输直接提交，避免依赖可能正在等待图执行的公共提交队列。
这是根据 Python 栈和源码确定的修复方向，尚未通过原生栈证明具体等待环，也尚待实机确认。
仍保留 `copy_` 和 `_synchronize_h2d()`，不会提前发布 done；默认 callback 的普通加载流不变。
接口缺失时直接报错，不回退到普通流。
接口语义见 [SyncLaunchStream 源码](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/streams.py)。

第三轮结果：日志确认 SyncLaunchStream 已生效，但仍在同一个第 1 层预取的
`synchronize()` 中超时；此前绕开提交队列的修改未解决问题。加载流不仅有权重拷贝，
还可能有 `_refresh_expert_fp32_scale` 的类型转换和设备复制，不能将流等待直接解释为
权重 H2D 未完成，也不能据此确认 AIV 资源争用。

为定位第一个未完成的设备操作，增加显式诊断模式：

```bash
python -B script/run_moe_host_sync.py --npu 1 --backend mte --diagnose-load-stages
```

该模式只在前 8 个非空真实回调加载批次中，先等待已有流任务，再逐项提交原拷贝任务并等待，
最后逐槽位执行原 scale 刷新并等待。日志打印 BEGIN、SUBMITTED、COMPLETE：
停在 SUBMITTED 表示该阶段已返回提交接口但流未完成；停在 BEGIN 则是提交接口尚未返回。
日志由摘要保留，不包含权重内容。该模式改变任务重叠，只用于定位，不能比较性能。
默认关闭，不改变原 callback 路径。若失败，继续返回 summary.txt；若通过，也要关闭诊断后再验证。

第四轮分阶段结果：第 0 层加载与刷新全部完成；第 1 层预取中已有流任务完成，
8 MiB 的 w13 和 4 MiB 的 w2 拷贝也完成，随后 512 KiB 的 w13_weight_scale
打印 SUBMITTED 但没有 COMPLETE。未进入该批次的 scale_refresh，故本次阻塞点
在量化参数 Tensor 拷贝的完成等待，不能归因于后续刷新。

本轮仅对 MTE 后端的量化参数拷贝改用 Storage 字节视图，与大权重的拷贝方式一致。
严格要求 CPU→NPU、相同 dtype/shape、连续内存、目标为 ND 格式、字节数匹配且不越界；
视图包含 storage_offset，避免写错专家槽位。条件不满足直接报错，不进行隐式布局或类型转换。
默认 callback 仍使用原 Tensor 拷贝。此修改尝试绕开 Tensor copy 的额外处理，
尚不能据此断言底层死锁机制，仍需用上述诊断命令确认量化参数拷贝能否完成。

第五轮在模型初始化阶段退出，未执行预热；旧摘要只保留了 API Server 的外层异常，
没有保留 EngineCore 根因，因此不能确认是否由新增布局校验触发。
摘要提取现已保留初始化 traceback，也可直接读取已有日志，无须重跑模型：

```bash
python -B script/run_moe_host_sync.py --read-log bench_results/moe_host_sync_db462862/mte/inference_python.log
```

此入口只向控制台输出既有日志中的诊断信息，不构建、不启动服务、不生成结果文件。

补取日志确认启动失败来自新增的 `storage_quant_copy` 校验，发生在热专家预加载；
两端 dtype 均为 uint8。原错误信息没有列出各项布局值，不能仅凭它确定具体失败条件。
源码中的 MXFP4 scale 布局包含 reshape/transpose，因此只接受逻辑连续张量过于严格。
现支持 ND 格式下无重叠、无空洞的转置布局：源、目标物理顺序相同时直接取字节视图；
不同时在 pinned CPU 张量中按目标 stride 排列，再传输字节。不在 NPU 上新增布局转换。
仍检查类型、形状、字节数和边界，错误信息补充两端 shape、stride 及目标格式。
CPU 测试使用真实 uint8 转置 Tensor，覆盖顺序匹配/不匹配及非零槽位偏移，检查相邻槽位不变。
布局重排产生的 pinned CPU 源存储由 manager 保留，只有 H2D 同步成功后才释放，
避免异步拷贝仍在读取时临时缓冲区被回收；同步失败时继续保留。
继续运行带 `--diagnose-load-stages` 的真实模型命令；本改动尚待 NPU 验证。

## 首次真实推理通过及下一步

绿区 `moe_host_sync_ecf9e013` 已完成预热 1 次、测量 3 次，510 次回调
（预取 252、按需 258），errors/dropped/MSTX marker errors 均为 0。
生成文本摘要和 token 用量与此前 callback 基线一致，仅代表本次短输入覆盖。
停止 profile 耗时约 21 秒，导出成功，位于该结果目录的 `profile/mte/python/`。
辅助 gap 分析失败不代表 profile 丢失；解析器现补充根数组和 BOM 支持，
对象格式仍保留头部长度限制，实际导出格式兼容性需下一轮确认。

请求均值为 3365.015 ms，旧 callback 基线为 3348.679 ms，约高 0.49%，尚无整体加速证据。
预取/按需的 load_prepare 均值分别为 183.799/878.153 us，旧基线为 33.421/135.052 us；
新增布局处理是需要关注的候选开销，但跨轮差异不能直接归因为某一个步骤。
该次开启前 8 批诊断屏障，屏障日志出现在预热中，不能声称所有测量请求都被额外屏障拖慢。

下一轮保持实现不变，关闭诊断，从冷启动同时复测两组并保存 profile：

```bash
python -B script/run_moe_host_sync.py --npu 1 --requests 10 --warmup-requests 3
```

脚本显式关闭两组的加载诊断开关，避免测试脚本中残留配置。仍输出 summary.txt。
本轮对比的是当前完整 MTE 方案（含兼容性修改），不是单独测量交接原语的纯收益。

## 接入位置与执行流程

原来的 `host_sync_backend=callback` 是默认路径，继续使用 `_launch_host_func`。
显式选择 `mte` 后，按需加载和预取中的图内回调改走 `MteHostDispatcher`：

1. 外层 `update_weights` 或预取流程照常提交路由 D2H 及必要的流依赖。
2. 在原回调提交位置取得当前流的原生句柄，提交 AIV 发布算子和等待算子。
3. 发布算子通过 MTE 更新注册 Host 内存中的 request generation。
4. 一个常驻 C++ 工作线程观察到新请求，调用原来的 guarded 回调，执行真实
   `_update_weights`，包括 LRC、Torch 权重 H2D 和传输完成等待。
5. 回调成功返回后，工作线程更新 done generation，AIV 等待算子放行当前流。
6. 后续映射 H2D 和专家计算继续执行。

没有改写 LRC 和权重搬运算法。普通 eager 回调仍直接调用；MTE 后端的加载流统一使用
SyncLaunchStream，图内交接使用发布/等待算子。
每个捕获的回调拥有独立槽位及参数引用，图重放复用该槽位并递增 generation。
预取和按需任务、不同图尺寸不共用同一个参数槽位。当前预分配 2048 个槽位，其中一个用于初始化。

Torch 操作存在 Host 提交队列，不能在初始化时缓存流句柄后跨操作使用。
本实现每次在路由复制等操作提交之后读取 `stream.npu_stream`。
torch_npu 2.10 的该属性会调用 `NPUStream::stream()` 排空对应 Host 提交队列，
然后才提交原生 MTE kernel，保持提交顺序；这不是等待所有设备计算结束。
相关实现见 [Stream.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/npu/Stream.cpp)
和 [NPUStream.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUStream.cpp)。

实际模型接入支持 Ascend950、torch_npu 2.10、单卡、Torch H2D。
等待算子不会因为轮询次数耗尽就放行未就绪的权重。回调异常或请求持续未完成超过约 120 秒时，
Host 监控线程终止当前引擎进程，测试脚本保留日志并清理本轮服务。
该监控线程不执行预取计算；预取和按需回调由同一个工作线程串行执行。

## 绿区执行

更新代码后，在容器内的项目根目录执行：

```bash
python -B script/run_moe_host_sync.py --npu 1
```

脚本构建独立扩展和 MTE kernel，随后依次启动 callback、MTE 两组真实模型，
不再运行第一至第五阶段的独立 NPU 测试。模型路径、MTP、预取数量和图尺寸等继承
`script/test_graph2.sh`；两组均使用 Python LRC 和 Torch H2D，便于单独比较交接机制。
默认每组预热 1 次、采集 3 次请求，沿用原短 prompt、`max_tokens=5` 和 `temperature=0`。

只跑一组可添加 `--backend mte` 或 `--backend callback`。
通过 `--requests`、`--warmup-requests` 修改请求数，`--profile-timeout` 控制停止采集的等待时间。
停止采集超时仍保留已获得的计时数据，标记 profile 不完整，清理本轮服务后继续下一组。

默认结果目录为 `bench_results/moe_host_sync_<随机编号>`，可用 `--output-dir` 指定新的目录：

- `summary.txt`：两组关键时延、请求用量、生成文本摘要及一致性结论。只需复制此文件内容返回。
- `callback/inference_python.log`、`mte/inference_python.log`：各组服务日志；文件名中的 python 指 LRC 后端。
- `profile/callback/python/`、`profile/mte/python/`：MindStudio profile，包含原有 MoE MSTX 标记。

脚本确认 MTE 组实际执行过卸载回调，并比较两组生成文本摘要与 token 用量。
短请求一致仅代表本次输入通过；两组均启用计时和 profiler，所报请求时延是诊断结果，
不能直接作为关闭采集后的生产性能收益。若失败，返回 `summary.txt` 中的失败信息和自动附带的日志尾部。
