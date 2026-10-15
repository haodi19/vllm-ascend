# 在 MindStudio 中查看专家卸载回调

MSTX 是昇腾提供的自定义时间标记接口。这里使用
`torch_npu.npu.mstx.range_start(message)` 和 `range_end(range_id)`，不传入 stream，
在 CPU 回调线程上记录开始和结束时刻。标记随同一份 torch_npu profile 导出，
可与 NPU 的 `MEMCPY_ASYNC` 在同一时间轴上对照。没有改变缓存算法、加载方式或同步方式。

接口参考：[Ascend PyTorch MSTX 源码](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/mstx.py)。
采集端通过 `_ExperimentalConfig(mstx=True)` 开启标记导出，需安装支持此参数的 torch_npu。

## 绿区一次执行

同步本次代码后，在容器内项目根目录执行（`--npu 1` 换成实际物理卡号）：

```bash
python -B script/run_verified_cpp_tests.py --only inference --mstx --npu 1 --requests 1 --profile-timeout 300 --output-dir bench_results/mstx_01
```

输出目录必须不存在。脚本依次启动 Python、C++ 两组服务，沿用 `script/test_graph2.sh`
模型、预取和 MTP 配置；各预热 3 个请求，采集 1 个请求，每个请求最多生成 5 个 token。
减少采集请求是为了先看清时间线，降低导出量；这不是用于下性能结论的统计实验。
原有可选 C++ 扩展的构建步骤保留，无需新增 C++ 构建步骤。

关键输出在 `bench_results/mstx_01/summary.txt`，详细服务日志另外保存。
每组 profile 位置在摘要的 `INFERENCE ... profile=...` 行；两组各自导入 MindStudio。
摘要新增 `MSTX ... marker_errors` 和 `MSTX TRACE ... label_occurrences`：

- `marker_errors=0`：包装层没有发现失败的标记调用，但仍需确认 trace 有标记。
- `label_occurrences>0`：导出的 `trace_view.json` 包含本次标记；此值是字符串出现次数，
  不是严格的回调数量或 profile 完整性校验。
- `PROFILE STATUS` 不完整：保留现有文件，但不能认为完整采集成功。
  超时仍会清理本轮服务并继续下一组。未导出文件时，仅凭已有计时表不能完成可视化验证。

请只复制经检查脱敏后的 `summary.txt` 内容；profile 保留在绿区。

## 查看时间块

导入 profile 后，搜索 `moe_offload/`，展开匹配的 Host/线程/MSTX 轨道。
不同 MindStudio 版本的轨道名称可能不同。CPU 标记不会变成 `Ascend Hardware` 的
NPU Stream 上的 kernel；应将匹配线程与 `MEMCPY_ASYNC` 所在 Stream 一起展开、缩放到同一时段。

例如一个时间块名称为：

```text
moe_offload/update_weights|backend=cpp|layer=12|kind=reactive|rows=3|call=42
```

| 字段 | 含义 |
| --- | --- |
| backend | 本组 LRC 实现：python 或 cpp；cpp 组的加载逻辑仍是 Python |
| layer | 回调参数中的 layer_idx，原样记录，不重新编号 |
| kind=reactive | 当前层按实际路由更新权重 |
| kind=prefetch | 预取回调 |
| rows | 本次回调接收到的路由张量行数，不等于用户请求数或预取层数 |
| call | 此进程内采样回调编号；父子块共享编号，两组之间编号不能直接配对 |

单击时间块查看 Start/Duration（或对应的开始时间、持续时间字段），注意界面单位。
块的宽度是这段 CPU 函数的经过时间，包含函数内部的等待和可能的线程调度停顿，
不是纯 CPU 指令执行时间。

| 时间块 | 对应执行范围 | 看图时的含义 |
| --- | --- | --- |
| update_weights | `_update_weights` | 整个回调函数，父范围 |
| observe | LRC `observe` | 更新访问统计；走到该函数才出现，预取不一定有 |
| choose_victims | LRC `choose_victims` | 选择被替换的专家 |
| load_submit | `_load_expert_weights_into_slots` | 准备并提交加载工作的父范围，不是实际搬运完成时间 |
| load_prepare | `_build_expert_h2d_tasks` | 构造传输任务 |
| copy_submit | TorchCopy `copy_batch` | CPU 提交拷贝；真实设备搬运查看 `MEMCPY_ASYNC` |
| scale_refresh | `_refresh_expert_fp32_scale` | 更新相关 scale 数据的函数范围 |
| synchronize | `_synchronize_h2d` | CPU 等待先前提交的 H2D 工作完成 |

父子范围存在包含关系，不能将所有块宽度直接相加。某个函数没有被调用时，不会绘制零时长块。
`outer = total - observe - choose_victims - synchronize` 是多个不连续区域的总和，
因此没有绘制单个 `outer` 块；它包括 `load_submit` 和其他剩余逻辑。
`host_without_sync`、`load_other` 同样是差值指标，保留在计时表中。

具体看空泡时，先圈出两个 `MEMCPY_ASYNC` 之间的间隔，再看同一时段是否覆盖
`observe`、`choose_victims`、`copy_submit` 或 `synchronize`。
时间重叠提供定位线索，不独自证明因果关系。`update_weights` 开始之前的空白没有被本次标记覆盖，
不能直接认定为 `_launch_host_func` 的同步开销；本次也没有记录每次图重放的回调入队时刻。

## 开关和测量影响

### observe 到 choose_victims 之间的细分

开启 `--mstx` 时，这段外围逻辑另有六个时间块，并在 `summary.txt` 的
`GAP BREAKDOWN` 中按平均耗时从大到小输出。按 backend、kind 和 token_rows 分组，
给出每个回调中的平均、P95、最大耗时（微秒）。无需等待 profile 导出成功即可读取这些统计。

| 标记 | 对应步骤 |
| --- | --- |
| gap_mapping | 映射转列表、构造槽位归属字典和驻留专家集合 |
| gap_missing | 当前层缺失/命中集合计算；预取路径还包括预测候选评分和筛选 |
| gap_hit_stats | 计算命中率等统计值 |
| gap_cache_stats | 更新缓存统计，包含达到间隔时的统计整理与可能的统计日志 |
| gap_slots | 构造可替换槽位列表 |
| gap_debug | debug 参数整理、排序、张量格式化及日志输出；关闭 debug 时仅经过分支判断 |

`measured_share` 的分母是六个小项平均时间之和，不是可视化中整个间隔的宽度。
没有执行的分支贡献零耗时（或范围内分支判断开销）；这里不是按实际操作调用次数求平均。
子范围之间仍有打标、Python 控制流和可能的线程调度开销。
这些子项属于 `outer`，不要再加到 `outer` 或 `total` 上。
若要解释某一个特别大的空泡，需在 MindStudio 对照同一 `call` 的子块，
不能只用跨回调的平均值代替该次耗时。预取不执行 observe，所以其小项不是 observe 后间隔。

复测可使用新目录：

```bash
python -B script/run_verified_cpp_tests.py --only inference --mstx --npu 1 --requests 1 --profile-timeout 300 --output-dir bench_results/mstx_gap_01
```

### 默认行为

默认关闭。`--mstx` 为两组注入 `expert_offload_config.callback_mstx_enabled=true`，
并启用 profiler 的 MSTX 导出。此开关要求 `callback_timing_port` 已设置，支持现有单卡
Python/CPP LRC + TorchCopy 诊断模式。标记仅在计时窗口打开后生成，预热不打标。
不加 `--mstx` 即恢复原来的测试方式，不需改动 `test_graph2.sh`。

MSTX 的调用、字符串构造和采集本身有开销。子函数计时尽量避开自身起止标记开销，
但父函数时间仍会包含内部子标记的开销。因此这轮用于分析时间分布，
不要拿它直接替换 README 中关闭 MSTX 的性能数据。

蓝区可运行的回归测试：

```bash
python -B tests/ut/expert_offload/test_callback_mstx.py
python -B tests/ut/expert_offload/test_verified_cpp_tests.py
```

真实 NPU 导出、MindStudio 显示和时间轴对齐仍需绿区验证。
