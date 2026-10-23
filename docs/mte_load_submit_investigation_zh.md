# MTE load_submit 耗时排查

## 当前证据

`moe_host_sync_02` 的按需调用中，两组平均加载专家数相同，均为 4.0256。
callback 的 `load_submit` 均值为 299.617 μs，MTE 为 1015.621 μs，增加 716.004 μs。
其中 `load_prepare` 增加 604.670 μs，约占该增量的 84.5%；`copy_submit` 增加 111.092 μs。
这将首要排查范围缩小到任务准备，而不是直接把增量解释为设备传输变慢。

代码确认，MTE 量化参数适配在每次布局不一致时都重新分配 pinned CPU 张量并重排，未复用布局准备结果。
但实际每次是否进入此分支、各操作占多少时间，必须由目标机器的细分计时确认。
源布局一致时不分配、不重排；不能假设所有量化任务都做了 CPU 数据复制。

## 本轮测量

原有单命令入口 `script/run_moe_host_sync.py` 会将新增数据输出到同一份 `summary.txt`。
沿用 MSTX 测量窗口，不新增配置项，不更换拷贝方式，不增加设备同步或逐任务日志。

| 指标 | 对应操作 |
| --- | --- |
| `quant_adapter` | 整个 `storage_quant_copy` 调用，是下列子项的父区间 |
| `quant_get_format` | 查询目标 NPU 张量格式 |
| `quant_allocate` | 分配目标 stride 的 pinned CPU 张量，仅布局不匹配时执行 |
| `quant_pack` | CPU 张量 `packed.copy_(source)` 重排 |
| `quant_source_view` | 源 storage 获取、偏移与边界检查、字节切片 |
| `quant_destination_view` | 目标 storage 获取、偏移与边界检查、字节切片 |
| `quant_task_replace` | 构造替换后的 H2D 任务对象 |

每项 mean/p95/max 都按一次 `_update_weights` 中累计的该项耗时统计，单位 μs；
`calls` 是整个测量窗口内的执行次数，不是每次调用平均值。
`quant_allocate` 的 calls 可以确认有多少任务真正分配/重排，和 `quant_adapter` calls 对比可判断比例。

摘要还输出两个差值：

- `quant_adapter_residual_mean_us`：适配总时间减去已测子项，包含布局检查、分支、Python 调用和埋点开销。
- `prepare_outside_adapter_mean_us`：`load_prepare` 减去适配总时间，包含原有权重任务构造、量化源 reshape 等。

父子指标不能相加；新增细分埋点也有开销，本轮用于定位，不能拿新增后的 total 直接宣称性能变化。
仅在 MSTX 已启用且内层回调正处于测量窗口时记录细分数据。原 callback 后端没有 MTE 适配，
不会输出这张 MTE 细分表。相关回归测试只在蓝区执行，不加入绿区实验日志。

## 判断顺序

先检查 `quant_allocate`/`quant_pack` 的调用次数及耗时，确定重复内存准备的占比。
然后检查格式查询和两端字节视图，避免误把潜在后端调用开销全部归为重排。
若主要时间仍在两个 residual 中，再对对应区域进一步定位。
`copy_submit` 的新增耗时单独保留分析，不能通过缩短 `load_prepare` 就声称整个 load_submit 问题已解决。

本轮不回退到之前卡住的量化 Tensor H2D 路径；其卡死的底层根因与当前 CPU 准备开销是两个问题，
不能从这次计时反推之前卡死的根因。
