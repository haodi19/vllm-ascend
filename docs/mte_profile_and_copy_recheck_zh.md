# MTE 可视化与量化拷贝复核

## 2026-10-26：无 profiler/MSTX 的结果与 Prefill 布局修复

`moe_debug_03` 的两组各完成 3 个 CPU 分项计时请求和 1 个可视化请求。
两组输出及 token 用量一致，计时无 errors/dropped，profile 均完整导出，未再次触发拷贝阻塞。
同一 CPU 计时窗口中，两组均为 252 次预取、258 次按需更新，分别加载 236、1294 个专家，
因此以下分项具有相同调用量及加载量。单位为微秒/次更新：

| 指标 | 预取 callback | 预取 MTE | 按需 callback | 按需 MTE |
| --- | ---: | ---: | ---: | ---: |
| `load_prepare` | 39.830 | 70.355 | 147.757 | 287.223 |
| `copy_submit` | 38.691 | 65.669 | 139.143 | 234.946 |
| `load_submit` | 83.726 | 141.439 | 308.839 | 536.724 |
| `synchronize` | 237.579 | 190.334 | 1153.032 | 1003.585 |
| `load_submit + synchronize` | 321.305 | 331.772 | 1461.871 | 1540.309 |

关闭 profiler/MSTX 后，准备与提交的新增耗时仍存在，不是纯打点假象。
合计预取增加 3.3%，按需增加 5.4%，不能仅凭等待下降宣布搬运提速。
后续可视化窗口的平均加载量与此窗口不同，不能用两窗口时延直接相减估算 profiler 开销。

Prefill 的全专家加载也是搬权重，需单独看：两组均调用 129 次、每请求 43 层。
每层均值从 **66.597 ms 增至 109.052 ms**，每请求累计从 **2863.692 ms 增至 4689.251 ms**，
增加约 1825.559 ms。作为交叉检查，整请求均值增加约 1903.326 ms，主要增量落在
Prefill 搬权重阶段，不能归咎于 Decode 的几十微秒交接差距。

### 本轮实际修改

上一轮只让 CPU scale 按 Decode 目标布局预打包，Prefill 目标池仍由普通 `torch.empty(shape)`
连续分配。这会使非连续的 CPU scale 在 Prefill `Tensor.copy_` 中再次面对布局差异。
这是预打包方案遗漏的消费路径；本次计时证实该阶段显著退化，但尚不能把新增时间全部精确归因于某一个底层转换算子。

本轮仅对 `mte + prepacked` 的 uint8 w13/w2 scale 修正两点：

1. Prefill 池使用一维底座分配，再建立与 Decode 每专家 stride 相同的视图；
   内存字节数不增加，不建立第二份 CPU 权重缓存。
2. Prefill 初始化和推理加载均使用已校验的 Storage 字节拷贝，禁止运行期自动重排。
   shape、dtype、ND 格式、stride 或字节范围不匹配时明确报错，保留异步提交后的原同步等待。

原回调、`storage`、`tensor` 路径以及非 uint8 scale 不变；Decode 搬运逻辑本轮不变。
CPU 回归覆盖两种真实 scale 形状、逻辑值、目标槽位偏移、相邻槽位、存储大小及不兼容布局拒绝。
设备端 GMM 对新 Prefill 视图的实际兼容性和性能仍需真实模型验证，不能以 CPU 字节拷贝测试代替。

同步代码后运行两组真实模型，由脚本自动构建、比对输出、分别计时及采集 profile：

```bash
python -B script/run_moe_host_sync.py --npu 4 --backend both --quant-copy prepacked --timing-requests 3 --requests 1 --warmup-requests 1 --profile-timeout 300 --output-dir bench_results_tmp/moe_debug_04
```

返回 `bench_results_tmp/moe_debug_04/summary.txt` 文本。重点比较新增的
`PREFILL ms/layer`、`PREFILL accumulated ms/request` 与后面的 Decode 搬权重表。
可视化中查看 `prefill_load_layer` 的主机时间块及对应 MEMCPY，Decode 仍查看
`update_weights/load_prepare/copy_submit/synchronize`；两阶段不要混为同一组样本。

## 最新结果：优先比较搬权重分项

2026-10-23 的 `moe_debug_02` 使用 `prepacked`，两组均完成预热、推理和 profile 导出，
输出摘要及 token 用量一致。两组各记录 84 次预取、86 次按需加载，分别加载 80、455 个专家。
MTE 的 Decode `quant_allocate/quant_pack` 均为 0：此前在每次 Decode 搬运时分配和重排 scale
的操作已消除。本次结果只覆盖该请求与配置，不代表所有模型输入已经验收。

下表单位为微秒，表示每次内层 `_update_weights` 的均值，**包含计时器及 profiler/MSTX 扰动**。

| 搬权重阶段 | 预取 callback | 预取 MTE | 按需 callback | 按需 MTE |
| --- | ---: | ---: | ---: | ---: |
| `load_prepare`：组织搬运任务 | 38.330 | 99.944 | 161.466 | 493.653 |
| `copy_submit`：提交拷贝 | 45.651 | 60.750 | 198.414 | 245.665 |
| `scale_refresh` | 1.455 | 1.662 | 5.778 | 6.124 |
| `load_submit`：上述提交阶段总计 | 96.542 | 172.279 | 409.704 | 783.613 |
| `synchronize`：提交之后的剩余等待 | 226.527 | 191.056 | 1148.227 | 1040.040 |
| `load_submit + synchronize` | 323.069 | 363.335 | 1557.931 | 1823.653 |

当前不能宣布搬权重提速：虽然剩余等待减少，但准备与提交增加更多，二者合计预取增加约 12.5%，
按需增加约 17.1%。`synchronize` 不是完整 H2D 时间；提交过程中设备已经可能在搬运，
只看它下降会遗漏等待向提交阶段移动的情况。`load_submit` 已包含前三项，不能再次相加。
MTE 使用 `SyncLaunchStream` 直接提交加载任务，回调版保留普通加载流；两者 CPU 提交方式不同，
也会影响工作落在 `copy_submit` 还是后续等待中的比例。不能将 `copy_submit` 增长直接解释成
NPU 上 MEMCPY 本体变慢，需要同时查看合计值与设备 MEMCPY 时间块。

`load_prepare` 的 MTE 专用适配器占预取 53.526、按需 282.379 微秒，
包含格式与布局检查、Storage 视图创建、任务对象替换。它在 `prepacked` 下不再重排数据，
但仍有这些准备步骤。适配器的 residual 还包含嵌套计时与标记开销，不能称为纯 Python 计算时间。
本次 MTE 比 callback 多 5350 个 MSTX 区间，正好对应 1070 次量化任务各 5 个额外子区间；
这也是为什么下一轮必须把分项计时和可视化分开。

### 与 MTE 修改的关系

量化 scale 原本就有非连续布局，原回调版通过 Tensor 拷贝处理。
MTE 改变了交接方式及加载流的提交方式，原 Tensor 路径在新组合下曾卡在 scale 拷贝后的同步，
这是迁移中的兼容性问题；仍未证实底层究竟是哪条 runtime 锁、事件或设备资源依赖。
为绕过该路径而加入的 Storage 适配及早期 CPU 重排是我们新增的开销，不能归咎于原版。
`prepacked` 已去掉 Decode 热路径重排，剩余准备开销与计时扰动需要分开判断。

另一个待测边界是 Prefill 全专家加载：本配置阈值为 `24 / 6 = 4` 个 token，
5-token prompt 可能走 `_prefill_load_layer`，是否发生取决于实际送入该层的 token 数。
该函数不在 Decode `_update_weights` 统计中。其池仍为连续布局，而预打包 CPU scale 是 Decode 布局，
仍可能在 Tensor 拷贝中整理数据。因此新增独立的 `prefill_load_layer` 标记与 `prefill_load` 汇总，
不把这部分混入 Decode 搬运指标。本轮不改变 Prefill 数据布局。

### 本轮改动与执行

1. 原生 Host worker 只扫描已启用的交接槽位，避免每轮固定扫描全部 2048 个槽位。
   启用范围在设备发布任务前扩大，图重放可能用到的旧槽位始终保留。
   这是交接路径的改进，不等于已经证明 `load_submit` 会因此下降。
2. 新增 CPU 分项计时窗口：profiler 未启动，MSTX 与细分诊断标记关闭，
   仍测 `load_prepare/copy_submit/scale_refresh/load_submit/synchronize`。
   它依然包含基础计时器开销，不能当作零扰动测量。
3. 之后单独采集一轮完整 profile，原细分 MSTX 保留。回调版与 MTE 版顺序执行，
   汇总表同时列出回调次数、加载专家数与各项差值；负差值表示 MTE 更快。

同步代码后，在绿区执行一次即可；脚本会重建 Host 扩展 API 3 并顺序运行两组：

```bash
python -B script/run_moe_host_sync.py --npu 4 --backend both --quant-copy prepacked --timing-requests 3 --requests 1 --warmup-requests 1 --profile-timeout 300 --output-dir bench_results/moe_weight_03
```

`--timing-requests 3` 是无 profiler/MSTX 的搬权重分项测量请求数；
`--requests 1` 是后续可视化采集请求数。两者不是预取层数或 MTP token 数。
返回 `bench_results/moe_weight_03/summary.txt` 的文本即可。
主要看 `WEIGHT STAGE COMPARISON`；可视化保存在该目录的 `profile/callback/python` 和
`profile/mte/python` 下。可选的 `--latency-requests` 仅用于额外的整请求测量，默认关闭。

若先检查现有设备时间线，不启动模型、不写文件：

```bash
python -B script/summarize_mte_timeline.py bench_results/moe_debug_02/profile
```

它按设备进程 metadata 识别真正的 Ascend Hardware 事件，输出交接算子及设备累计耗时。
设备事件可重叠，AIV 等待也是设备事件；这些时间不能直接全部相加当作请求关键路径。

## 前一轮修复：加载时建立设备布局，运行时禁止重排

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
不增加存储字节数、运行时禁止重排与布局不匹配失败。后续绿区结果见本文开头。

最初验收命令如下，不开启会改变执行顺序的逐项同步诊断；新一轮使用本文开头的分项比较命令：

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
