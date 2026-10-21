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
