# MTE 可视化与量化拷贝复核

## 最新修复：加载时建立设备布局，运行时禁止重排

`moe_debug_01` 的真实模型实验已复现 Tensor 模式阻塞：第 0 层完成，
第 1 层预取中 w13、w2 原始权重分别搬运完成，随后 512 KiB 的 w13 scale
在 Tensor.copy_ 返回后停在加载流同步。不是 profiler 停止问题，发生在预热阶段。

源码检查补充了此前遗漏的路径：torch_npu v2.10.0 的
`copy_h2d_baseformat_opapi` 对非连续目标创建连续临时 NPU 张量，先执行 H2D，
再经 `copy_d2d_baseformat_opapi` 调用 `aclnnInplaceCopy` 写入目标。
因此该 Tensor.copy_ 不是单纯的 MEMCPY，而可能包含设备计算与临时内存管理。
参考：[torch_npu v2.10.0 CopyKernelOpApi.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp)。
用户安装的是开发构建，公开版本不是该二进制源码的精确快照。
日志与源码支持优先移除这条运行期设备布局拷贝路径，但尚不能确认具体是算子资源、
图调度还是内部事件造成阻塞；不能把其中一种推测写成已证实的底层死锁原因。

修复选项 `--quant-copy prepacked` 在 CPU scale 分配时使用 NPU 目标的 stride，
沿用原加载函数写入逻辑值。仅改变同一块 pinned 内存的视图，不增加第二份专家缓存。
原来的 `_copy_scale_assembled` 和 `_copy_scale_direct` 已执行 MXFP reshape/transpose，
现在直接写入正确物理位置；无需在推理回调中再反复分配和重排。
初次注册和后注册的 MTP 层都经过相同的缓冲区分配入口。

推理阶段仍走已跑通的 Storage 字节搬运，保持 H2D 完成等待与 MTE generation 协议。
`allow_pack=False` 禁止静默回退：如果源、目标布局不匹配，立即报错，不在热点里偷偷重排。
该选项针对当前 MXFP uint8 scale；其他量化布局如需重排会明确拒绝，不能当作所有模型通用修复。
回调路径、原 `storage` 路径和实验 `tensor` 路径均保留。

CPU 回归检查覆盖真实 `(64,4096,2)`、`(32,4096,2)` 布局，
w1/w3 拼接、w2 加载、与旧缓冲区逻辑数值相等、字节搬运后数值正确、相邻槽位不被覆盖、
不增加存储字节数、运行时禁止重排与布局不匹配失败。真实 NPU 验收仍需绿区执行。

下一轮直接验收修复，不开启会改变执行顺序的逐项同步诊断：

```bash
python -B script/run_moe_host_sync.py --npu 4 --backend both --quant-copy prepacked --requests 1 --warmup-requests 1 --profile-timeout 300 --output-dir bench_results/moe_prepacked_01
```

观察 MTE 能否完成预热和推理、两组输出摘要和 token 用量是否一致、
MTE 的 `quant_allocate/quant_pack` 调用次数是否为 0，以及 profile 是否完整。
单次请求只用于快速验收，不能据此宣称稳定的性能收益。

## 当前证据与边界

2026-10-23 返回的 `moe_load_detail_01` 诊断显示输出盘剩余 **0.00 GiB**；
MTE 采集缺少设备 start_info 和原始 data，最终 trace_view.json 没有闭合。
当前空间不足是确定的阻塞条件，但不能据此倒推此前每次失败都由磁盘满造成。
这份数据不能通过补 JSON 结尾或重复离线解析恢复成完整 CPU/NPU 时间线。

之前“停止超时就跳过”的机制只保证计时数据能保留、下一组能继续，**不保证 profile 完整**。
之前未在昂贵实验前检查空间、缺失原始输入时仍允许复制重解析，是脚本的缺陷。

现在运行器在构建前、每组启动前和开始采集前检查空间；采集输出盘至少保留 20 GiB，
构建所在盘至少 2 GiB。20 GiB 是本实验的保守预算，不是任意长度采集的容量保证。
其他用户同时写盘仍可能耗尽空间。离线解析在复制前检查原始设备输入和空间预算。
MSTX 模式关闭 `data_simplification`，保留 CANN 中间输出和日志供恢复与定位；
普通非 MSTX 采集保持原设置。这不是认定精简机制删掉了本次原始数据。

## 先验证 CPU 重排是否必要

现有证据仅证明原 Tensor 量化拷贝曾停在同步处，Storage 拷贝加 CPU 重排可以运行。
尚未证明 Tensor 路径底层卡住的确切依赖，也没有证明 CPU 重排是唯一解决方式。

模型量化 scale 的目标布局经过 reshape/transpose，逻辑 shape 相同不代表物理排列相同。
直接删掉重排但保留 Storage 字节拷贝会有错值风险。应先验证恢复 Tensor.copy_ 的可行性。

`script/probe_mte_quant_copy.py` 使用现有原生 worker、MTE 发布和有界等待算子、
SyncLaunchStream，构造 512 KiB uint8 量化张量，使用与模型相同形式的转置布局。
这是合成数据，不代替真实模型形状、dtype 和其他流依赖的最终验证。

依次在独立进程运行四组，每组三次图重放并核对数据和相邻槽位：

1. 布局不同，Tensor.copy_，发布后不运行 AIV 等待：排除单独的工作线程/布局拷贝问题。
2. 布局相同，Tensor.copy_，AIV 等待：检查普通张量拷贝与等待共存。
3. 布局不同，Tensor.copy_，AIV 等待：检验布局处理与等待是否存在依赖冲突。
4. 布局不同，现有 CPU 重排加 Storage 拷贝，AIV 等待：工作路径对照。

AIV 等待有轮询上限；达到上限后拷贝才完成也判失败，不能误报通过。
每个子进程最多 90 秒；硬超时后停止整个测试，先确认设备恢复，不能连续启动更多测试。
不加载模型、不记录 profile、不写实验文件。保持现有生产路径不变。

若只有第 3 组失败，可将范围收窄为不同布局 Tensor 拷贝与 AIV 等待的交互，
仍不能仅凭此声称已定位到某个 runtime 锁或设备资源。
若四组全通过，应回到真实模型的 shape/dtype/流依赖重现，不能继续声称该兼容处理必需。
若第 1 组也失败，应先排查基本 Tensor 拷贝、设备状态和测试条件。

### 首次隔离结果与真实模型复核

首次返回的前三组均通过，包括不同布局 Tensor 拷贝与 AIV 等待共存。
因此不能将“布局不同 + AIV 等待”直接认定为必然卡死原因。
第四组尚未执行拷贝：测试的四维目标默认分配成 NCHW，被只接受 ND 的 Storage 适配器拒绝。
脚本已改成显式 ND 格式，并在异常路径先同步设备用户再释放交接内存。
原前三组的结论限定于当时的 NCHW 合成数据，不能当作实际模型的完整验证。

真实模型运行器新增实验选项 `--quant-copy tensor`：仅 MTE 组绕过 `storage_quant_copy`，
保留原 Tensor 任务，不分配重排临时张量，交给原 `TorchCopyH2DTransport.copy_batch` 提交。
MTE 同步协议、SyncLaunchStream、权重主体拷贝和 scale_refresh 保持原样。
默认仍为 `storage`，回调组不受此选项影响。

下一轮直接验证真实模型，不必等待合成测试重新通过才启动：

```bash
python -B script/run_moe_host_sync.py --npu 4 --backend both --quant-copy tensor --diagnose-load-stages --requests 1 --warmup-requests 1 --profile-timeout 300 --output-dir bench_results/moe_quant_tensor_01
```

该轮逐项同步前八批真实加载，记录第一批张量的 dtype/shape/stride，定位提交或同步阻塞点；
它用于正确性和兼容性排查，不作为性能对比。运行器会比较回调组与 MTE 组的输出摘要和 token 用量。
若完整通过，说明当前实际配置可以不走我们新增的 CPU 重排，但仍需去掉诊断同步后验证正常执行。
若卡住，返回 summary.txt 内的首次卡点及诊断栈，再按真实张量布局与流依赖继续定位。

## 绿区执行

先释放空间或选用有空间的挂载盘，查看容量（不删除文件）：

```bash
df -h . /tmp /data /mnt
```

不需要重新构建 MTE 库；隔离脚本默认选取本项目最近构建的库并打印路径：

```bash
python -B script/probe_mte_quant_copy.py --npu 2
```

空间检查通过后，先只用一个请求验证 MTE 可视化，减少采集与导出压力：

```bash
python -B script/run_moe_host_sync.py --npu 2 --backend mte --requests 1 --warmup-requests 1 --profile-timeout 300 --output-dir bench_results/moe_profile_recheck_01
```

把 2 改成空闲卡编号。若更换输出盘，直接替换 `--output-dir` 为该盘内有写权限的新目录绝对路径。
只有出现 `PROFILE SAVED` 和 `profile_status: complete` 才作为可视化成功；
导入输出目录下 `profile/mte/python/*_ascend_pt`。
返回隔离测试控制台输出和本次 `summary.txt` 文本即可。
