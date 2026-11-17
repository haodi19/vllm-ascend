# 在 MindStudio 中查看专家卸载回调

## Host 与设备等待的时序核查

2026-11-17 的核查现象：PR 采集中，用户观察到 Stream 25 的短 wait 完全落在
预取 `synchronize` 的 MSTX 区间内；Stream 27 的长 wait 在预取 `synchronize`
尚未结束时开始，并在按需 `synchronize` 内结束。已有 trace 的只读结果确认这类
时间重叠普遍存在，不能将其归为个别截图误读。

本次四组采集的 Host 标记检查均无缺失、重复或嵌套错误。Stream 25 的
publish/wait 数量与预取回调相等，邻接结构为 publish → EVENT_WAIT → wait →
MEMCPY → EVENT_RECORD；Stream 27 的数量与按需回调相等，邻接结构为
MEMCPY → publish → wait → MEMCPY。这些证据支持本次的流角色判断，不能推广为
所有进程中固定的流号，也不能代替逐任务 slot/generation 身份匹配。

| 采集配置 | 预取 wait 数 | wait 中位数（μs） | 完全落在预取 synchronize 内 | 同序号 wait 早于 synchronize 结束 | 固定设备时间平移量的必要范围（μs） |
| --- | ---: | ---: | ---: | ---: | --- |
| token5，MTE 默认 | 84 | 2.50 | 69/84 | 73/84 | [130.75, 153.00]，可行 |
| token5，MTE 组合 | 84 | 2.75 | 66/84 | 73/84 | [141.00, 155.75]，可行 |
| token128，MTE 默认 | 2436 | 2.50 | 1519/2436 | 2266/2436 | 下界 318.25 > 上界 161.00，无解 |
| token128，MTE 组合 | 2436 | 2.50 | 1861/2436 | 2248/2436 | 下界 315.50 > 上界 145.50，无解 |

“完全落在”只按导出坐标比较重叠；“同序号”及平移范围依赖完整顺序配对候选。
token5 与固定偏移假设相容，但没有测出实际时钟偏移。token128 在该候选下不能
用全程同一个常数解释，后续需检查分段变化、任务匹配及时间戳语义。当前不能据此
认定设备提前完成，也不能宣布已经定位为固定时钟偏移。

Stream 27 的 wait 中位数分别为 1046.625、1042.250、1625.750、1596.250 μs。
第一版工具因为检测到 publish/wait 边界有正重叠而拒绝全部按需顺序配对，且未报告
重叠量。因此那次输出中 `CROSS_STREAM_SHIFT_HYPOTHESIS` 实际只包含 Stream 25，
不能作为两条流共同满足偏移约束的证据。

后续补齐边界检查后，四份采集的 Stream 27 正重叠分别为 6、5、152、216 处，
全部恰为 0.25 μs，未发现 wait 与下一次 publish 重叠。0.25 μs 与当前约
1.794×10^15 μs 时间戳的双精度浮点粒度一致，支持数值量化或导出边界误差的解释；
不能用这种亚微秒现象解释几百微秒的跨时间轴矛盾。保留严格拒绝结果，额外容差
候选使按需侧检查得以完成。

完整层序列验证通过：token5 观测到 2 次层循环，token128 为 58 次；预取每次
覆盖层 1～42，按需每次覆盖层 0～42。以下是已打印对应轮次的两流必要范围交集，
表示假设按序对应成立时，应给设备时间加上的偏移，不是独立校准值：

| 配置 | 首轮交集（μs） | 中间轮交集（μs） | 末轮交集（μs） | 全段固定偏移 |
| --- | --- | --- | --- | --- |
| token5，MTE 默认 | [130.00, 146.00] | — | [131.75, 147.25] | [131.75, 146.00]，可行 |
| token5，MTE 组合 | [140.25, 155.75] | — | [142.50, 157.25] | [142.50, 155.75]，可行 |
| token128，MTE 默认 | [132.75, 151.25] | [231.00, 250.50] | [318.25, 336.75] | 无解 |
| token128，MTE 组合 | [130.00, 145.25] | [229.25, 245.50] | [317.25, 331.50] | 无解 |

token128 的中间轮为从 0 编号的第 29 轮，末轮为第 57 轮。两组中，预取和按需
分别计算的 58 轮及 39 个局部窗口全部有解；同一层跨全段统计则全部无解。首末
范围中心在约 6.5 秒内增加约 185～187 μs，两组均约 29 μs/秒。这是两条时间轴
共同发生相对漂移的强证据，尚不能指定错误来自哪一种时钟或哪个转换实现。
摘要只列出部分轮次，没有验证全部 58 轮的跨流联合交集。

独立元数据核对没有支持“系统 wall/raw 时钟以约 29 μs/秒持续漂移”的解释。
按采集起止记录计算 `Δ(collectionTime × 1000 − clockMonotonicRaw)`，其中
collectionTime 单位为 μs、RAW 为 ns，得到以下结果。设备目录里的字段仍是
`clockMonotonicRaw`，不将其当作 NPU 硬件计数器。

| 配置 | Host wall/raw 偏移变化（μs/秒） | device 目录 wall/raw 偏移变化（μs/秒） |
| --- | ---: | ---: |
| token5，MTE 默认 | −2.863 | −3.866 |
| token5，MTE 组合 | −1.420 | −2.424 |
| token128，MTE 默认 | −2.441 | −2.304 |
| token128，MTE 组合 | −1.128 | −2.016 |

这些首末锚点的变化明显小于前述相对偏差变化，不能解释约 +29 μs/秒的趋势；
只有首末记录，不能排除中间变化。当前额外 5 秒采样中，MONOTONIC−RAW 和
REALTIME−RAW 的变化分别为 −0.047、+0.002 μs/秒，均与测量不确定度内稳定相容，
但这次采样不能还原 11 月 12 日的系统状态。

四组 Torch profiler 均保存 `syscnt_enable=false`。已读取的转换实现此时直接
返回输入时间，不使用该分支的 `freq=100` 换算，不能将这个数值认作根因。
这不等于 CANN 导出的 MSTX 也不需要频率：官方
[MSTX 导出实现](https://github.com/Ascend/msprof/blob/master/analysis/viewer/msproftx_viewer.py)
使用 Host counter 转换，
[时钟转换实现](https://github.com/Ascend/msprof/blob/master/analysis/common_func/info_conf_reader.py)
读取 CPU Frequency 与 Host 锚点；设备 counter 使用另一套频率和基准。
官方当前源码说明需检查的路径，服务器实际安装实现仍需逐项核对。

四次 `host_start.log` 还保存了 `cntvct` 与 `clock_monotonic_raw`。
在同一主机和启动时域、计数器连续且各 CPU 计数器同步的前提下，按相邻采集的
`Δcntvct / ΔRAW` 独立估计与 RAW 时间一致的有效换算频率：

| 相邻采集 | RAW 时间跨度（秒） | 估计 Host counter 频率（MHz） |
| --- | ---: | ---: |
| token5 默认 → token5 组合 | 396.529931 | 3544.671972 |
| token5 组合 → token128 默认 | 2119.366183 | 3544.671969 |
| token128 默认 → token128 组合 | 663.456336 | 3544.671968 |

三个估计接近，且独立于 MoE 握手。当时尚需对照 CANN 实际保存和使用的频率；
下文第三版核查确认该 MSTX 导出分支并未使用这一 CPU counter 频率。
RAW 自身可能也来自同一硬件 counter，这不是对物理振荡器的独立测量，
也不能把 CPU counter 频率直接当作 NPU 频率。
若只有 Host 频率换算出错、设备时间正确，给设备加的偏移应以
`(实测频率 / 配置频率 − 1) × 10^6 μs/秒` 增长。因此必须同时核对数值和方向，
不能仅因两种频率不同就定因；起始偏移还取决于各自基准。

只读工具 `script/inspect_moe_profiler_clocks.py` 原先遗漏标准编号文件名
`info.json.N`、`host_start.log.N`、`dev_start.log.N`，首次输出不代表这些文件不存在。
补齐识别后，一次读取四组的 Host/设备频率、原始锚点和当前安装的 CANN 转换源码。
工具不导入 Torch、不启动 NPU、不重新解析 profile，不修改原时间戳。
不能用握手约束拟合偏移后，再将同一批握手数据满足约束当作独立验证。

补读编号文件后，四组 `hwts_frequency` 均为 1000.000000 MHz，
Host 和设备目录的 `cntvct_diff` 均为 0。按 `dev_start.log.1` 的计数与 RAW
锚点跨采集计算，有效频率分别为 999.999997238、999.999999505、1000.000001278 MHz，
相对 1000 MHz 的速率差分别为 −0.002762、−0.000495、+0.001278 μs/秒。
这些跨采集平均值不支持约 29 μs/秒源于该标称频率误差的解释；仍不能独立验证
实际任务时间戳使用了同一个 counter，或保证采集内每一时刻的换算正确。

第二版输出里的 `CPU[0].Frequency` 仍为 `unparsed_string/value_omitted`，这是
工具隐藏了不符合数字格式的字符串，不能当作已取得实际频率，也不能据此判断
CANN 必然使用 CPU counter 频率。服务器返回的 `get_host_freq` 明确包含无效频率
回退分支；对应 `time_from_host_syscnt` 在默认频率下直接把输入按纳秒处理。
因此必须核对原值和实际采用的转换分支，不能默认使用 3544 MHz 左右的倍率。

第三版保留 `CPU[n].Frequency` 的有界原始字符串，补读默认频率常量和设备基准
解析辅助函数。可选 `--audit-mstx-export` 仅读取已有 `msproftx.db` 和最终
`trace_view.json`，按唯一的完整 Host 回调标记关联导出前后记录，报告原始单位/μs
比例、首末跨度差及代表样本。重复名称拒绝配对；数据库缺失时明确报告缺失，
不启动离线导出器。该核查用来定位导出是否改变时间跨度，不将数据库字段预先认定
为硬件 counter，也不将拟合残差小当作物理时钟或握手协议正确的证明。

第三版返回结果确认：四组 `CPU[0].Frequency` 和 `CPU[1].Frequency` 原值均为
空字符串。安装源码的 `HOST_DEFAULT_FREQ` 为 10^9，空字符串进入默认分支，
`time_from_host_syscnt` 直接按纳秒换算；不存在这一分支按约 3544 MHz 换算的情况。
token128 的两份已有数据库与最终 trace 按完整回调名称逐一匹配，结果如下：

| 配置 | 唯一匹配回调数 | 首末回调起点跨度（μs） | 数据库跨度 / 1000 与 trace 跨度之差（μs） | 起点仿射残差范围（μs） |
| --- | ---: | ---: | ---: | --- |
| token128，MTE 默认 | 4930 | 6485204.294 | 0.000 | [0.000, 0.000] |
| token128，MTE 组合 | 4930 | 6526681.096 | 0.000 | [0.000, 0.000] |

两组均没有重复名称或未匹配数据库回调。所有已匹配起点满足“数据库时间 / 1000
加固定偏移”，偏移也与各自 Host `collectionTimeBegin − clockMonotonicRaw / 1000`
一致。代表性首、中、末样本的结束点同样相符，但当前报告没有逐一汇总全部结束点残差。
因此，约 6.5 秒内增加 185～187 μs 的矛盾不能归因于这一 MSTX 数据库到最终 trace
的频率缩放步骤。数据库中的时间可能在更早的记录阶段已经偏离实际时钟，设备时间
也尚未独立校准；不能据此认定设备提前读到 done。token5 两组未在已探测路径找到
`msproftx.db`，不能将 token128 的逐条转换检查当作 token5 的验证结果。

独立校准工具 `script/calibrate_moe_profiler_clock.py` 运行 eager 和 graph 两组，
每组 61 次小张量 Add、间隔 0.2 秒；使用专用 Host 线程，在每次提交后同步，
不加载模型、专家权重或 MoE 握手算子。初始化、图捕获及预热均在采集前完成。
两组顺序使用同一张指定卡，分别保留 profile 和直接时钟记录，结果合并到一个
`summary.txt`。每组包含约 12 秒的采样跨度，连同初始化和导出有 240 秒超时限制。

- Host 检查：以 `CLOCK_MONOTONIC_RAW` 读取夹住 MSTX start/end API，检查导出
  标记是否落在对应上下界内，并报告首末偏移及偏移变化速率范围。
- 设备检查：仅在硬件轨道上唯一识别到每次一个 Add 且数量完整时，按串行提交顺序
  配对。真实执行应在 Host 提交前记录之后、同步返回后记录之前；工具报告当前
  导出坐标是否满足这一必要条件，以及是否存在统一偏移可满足所有样本。
- 辅助检查：比较设备 Event 的相对时间和 Host 提交/完成边界，独立于 trace 的
  绝对纪元转换。亚微秒残差需结合 Event 分辨率和 JSON 导出量化判断。

这些边界不使用 MoE 回调或 wait 关系来拟合。Host 标记越界而设备符合边界，优先
检查 Host 记录/换算；Host 正常而设备越界，优先检查设备记录、转换和区间语义；
如果两者均符合边界，只说明当前小负载没有复现，不能推翻旧采集中的矛盾，也不能
宣布 MoE 协议已经验证。工具不移动原 trace 的时间戳，缺失数据也不补成通过。

2026-11-17 15:50 的独立校准在物理卡 1 完成，两组均数值校验通过，唯一识别到
61 个同流 Add。Host MSTX 起止点分别全部落在直接 RAW 读取夹住的区间内，
设备导出时间则大量不满足“提交之后、同步返回之前”的必要边界：

| 项目 | eager | graph |
| --- | ---: | ---: |
| Host MSTX 开始 / 结束点越界数 | 0/61、0/61 | 0/61、0/61 |
| 设备 Add 时间边界越界数 | 58/61 | 59/61 |
| 最后一个 Add 导出起点早于 Host 提交前记录（μs） | 307.340 | 365.934 |
| 全程固定偏移约束交集（μs，导出减 RAW） | 下界 −164.113 > 上界 −336.704 | 下界 −178.870 > 上界 −365.934 |
| Event 相对时间超出 Host 必要范围的样本数 | 19/61 | 33/61 |
| 最后一个 Event 相对跨度至少短于 Host 允许跨度（μs） | 119.394125 | 192.4106875 |

这是不依赖 MoE slot/generation 配对的普通算子复现，eager 也有同类异常。
当前已确认设备导出坐标不能直接与 Host MSTX 横向比较，且单个固定平移不能使
所有样本满足因果边界。Host MSTX end 的首末偏移变化速率允许范围分别为
[−0.896, +0.414]、[−0.898, +0.398] μs/秒，与 0 相容，不支持 Host 记录以约
29 μs/秒漂移。start 首次打点较慢、上下界较宽，不宜用其宽范围估计实际漂移。

Event 相对时间不经过最终 trace 的绝对纪元转换，也出现偏短，因而排查重点进一步
转向设备计时及其换算链路，包括 Event 时间接口；不能仅解释为 JSON 的固定偏移。
这不等于已经确定物理硬件时钟损坏，也没有对历史 MoE 采集完成逐任务校准。
当前结果与原先设备坐标逐渐偏早的方向一致；摘要不足以估计本次准确漂移率。

本轮两组均记录停止 profiler 时仍处于 RECORD 的 schedule 警告。
[官方说明](https://www.hiascend.com/doc_center/source/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0133.html)
指出未完成采集周期可能导致数据不完整。此次目标 Add、MSTX 数量完整且结果正确，
但仍保留此采集质量限制。脚本随后改为显式 warmup=1、active=1、repeat=1，
在采样前推进到采集状态，采样后完成 active step 再 stop。
下一轮短校准用于验证正确收尾后异常是否仍在，不重跑模型，也不修改 MoE 同步协议。

工具补充所有样本的仿射必要约束：设设备时间 D = a × Host RAW 时间 H + b，
对每个 Add 使用提交前和同步返回后的边界求 a 的可行范围，不把 Host 区间中点
假定为真实执行时间。Event 的相对时间另行求解，负速率误差表示设备时间相对 RAW
推进较慢；若无可行解，明确报告单一倍率加偏移不足以解释。`--reference-run`
同时读取前次校准的已有数据，与新校准汇总到同一个新 `summary.txt`，不修改旧文件。
拟合范围仍是条件约束，不作为已验证的 trace 修正值。

16:00 的完整周期复测不再出现采集警告，eager / graph 的 Host MSTX 起止点仍全部
符合直接 RAW 边界，设备 Add 越界分别为 55/61、60/61。全部样本的设备计时相对
RAW 速率允许范围分别为 −39.733～−17.537、−36.912～−18.517 μs/秒，Event
相对计时也给出偏慢范围。由此确认当前环境存在独立于 MoE 的跨域计时不一致，
与旧 MoE 采集中约 29 μs/秒的相对偏差方向和量级一致。本问题停止重复校准排查，
后续转为可视化对齐；尚不指定错误属于物理时钟、驱动还是 CANN 的具体实现。

四组实验入口新增 `--align-profile-clocks`，显式开启同次采集校准和可视化导出。
完成无 profiler 请求计时后，控制线程预热一个专用小张量 Add 流；profiler 启动后、
正式采集请求前记录 24 个校准点，所有请求及回调计时结束后再记录 24 个。
两段均在同一 EngineCore、设备及 profile 内，不插入模型层或 MoE worker 的热路径。
校准原始记录保存为各组 `profile_clock_samples.json`。

离线工具 `script/align_moe_profile_clock.py` 使用偶数序号校准点拟合设备到 Host
的倍率和偏移，以未参与拟合的奇数序号点检验提交/完成边界，并检查 Host MSTX
与 RAW 是否相符。专用 Add 流不能唯一识别、模型不成立或留出样本不通过时，报告
`CLOCK_ALIGNMENT status=failed`，不输出标为成功的对齐文件。
拟合不使用 MoE wait、Host 回调位置或二者预期先后关系。

成功时生成同一 `ASCEND_PROFILER_OUTPUT` 下的 `trace_view_aligned.json`，
可直接导入 MindStudio；原 `trace_view.json` 保留。转换覆盖已识别的单设备
`Ascend Hardware` 进程下所有流、设备任务开始/结束及 flow 连线的设备端点，
Host MSTX 和 Host API 坐标保持原值，根元数据保留。未知时间域不猜测转换。
`clock_alignment.json` 记录倍率、偏移、校准范围及不确定度，摘要也打印相同结果。
校准范围之外的设备事件使用同一映射外推，并单独报告数量。

这提供一份经独立样本检查的可视化对齐估计，不能承诺微秒级边界零误差。
只有当前采集内的校准数据可用于该文件，不将另一轮小实验的倍率直接套到旧 profile。
现阶段原始实验统计继续保留原口径；对齐文件不自动替换历史空泡表，也不宣称已完成
底层计时修复或 MoE 协议精度验收。真实 PR 上的采集与 MindStudio 展示需要运行验证。

当前纯 MTE 严格等待路径要求同一次任务满足以下顺序：

```text
_synchronize_h2d 返回 → MSTX synchronize 结束 → Python Host 函数返回
→ C++ worker 发布本 slot / generation 的 done → 设备严格 wait 正常退出
```

消费者处等待还要求：预取严格 wait 退出、映射写回与交接事件完成后，主流才发布
同层按需请求。因此，如果完成配对后仍观察到上述关系，就不能将其解释为正常
的异步重叠。源码核查没有发现正常路径提前发布 done，也没有发现显式有界退出；
这不代替对服务器实际构建及设备运行的验证。

`CallbackTimings.wrap` 在同步函数返回的 `finally` 中结束 MSTX；
`host_sync.cpp` 在 Python 回调返回并释放 GIL 后才写入完成序号；
`host_sync_wait_strict_launch` 使用无限轮询，检查当前 slot 的准确完成代次。
MSTX 的 `backend=cpp/python` 表示 LRC 实现，不能据此辨认 callback/MTE 交接后端。

只读核查工具 `script/diagnose_mte_wait_timeline.py` 读取已有 `trace_view.json`，
不导入 Torch、不初始化 NPU、不重跑模型，也不修改原始时间戳或实验报告。
它先用 `Ascend Hardware` 元数据识别设备轨道，分别枚举每条轨道的 publish/wait，
再检查 Host 标记的线程、上下文和嵌套关系。不会按 Stream 25/27 的编号、wait
长短或横向重叠直接认定任务身份。

只有完整序列数量相等时，才检验同序号配对候选；该候选不是 slot/generation
身份映射的证明。给设备时间统一加上偏移 δ，正常顺序至少需要：

```text
publish_start_device + δ ≤ host_callback_start
host_callback_end ≤ wait_end_device + δ
```

对所有候选任务取约束交集，可判断一个共同时间平移是否能解释观测。
非零可行区间仅支持“系统性时间偏移”假设，不是独立时钟校准，不能据此平移数据后
重新宣称验证通过。publish 结束时间不作请求可见性的下界，因为 CPU 可能在
publish 算子退出之前就读到请求。

若不存在共同可行区间，继续区分任务配对错误、非恒定时间偏差、设备任务区间含义及
实际协议异常；已有采集不能辨别时，再补最小运行诊断，保持严格完成检查。
Host 内部嵌套完整性通过，只证明 Host 标记自洽，不证明 Host/设备时钟已对齐。
固定跨域偏移不会改变同一时间域内的区间长度，但可能改变 Host 窗口裁剪、空载回调
与设备空泡的重叠判定；这些跨域派生口径需在时间对齐核查后重新确认。

工具后续补充 `ORDER_AUDIT`：保留严格检查结果，打印全部正重叠的数量、最大值及
原始边界；仅当交替顺序正确且重叠不超过显式容差（默认 1 μs，可设为 0）时，额外
列出 `ORDER_TOLERATED_ORDINAL_CANDIDATE`。容差是诊断假设，不代表已证明采集误差，
不修改时间戳，也不将严格失败改成通过。报告同时列出真正参与偏移检查的流数。

`SHIFT_BINDING_SAMPLES` 显示决定全局上下界的层、call 和时间；`SHIFT_WINDOWS`
检查每 64 对任务的局部必要范围；`SHIFT_LAYERS` 检查各层的重复样本。
`SHIFT_LAYER_CYCLES` 仅在层序列可验证为完整重复的升序循环时，计算各层循环的范围，
不将不完整首尾默认为完整轮次。所有样本参与计算，摘要只显示少量代表行。
若局部范围成立而全程交集为空，才进一步检验随时间变化的偏差；这些范围仍不是
独立时钟校准。另统计同一个 wait 同时满足“起点在预取同步内、终点在按需同步内”
的数量，避免将两个独立计数误读为同一批任务。

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

### 相邻预取与按需回调之间的间隔

`--mstx` 现在还记录 `moe_offload/host_callback`，与内部 `update_weights` 使用相同的
backend/layer/kind/rows/call。当前层回调覆盖 guarded 函数；预取回调还覆盖映射快照与异常恢复，
分别显示为 `prefetch_snapshot` 和仅失败时出现的 `prefetch_failure_restore`。
这是 Python 包装范围，不覆盖进入 Python 之前的报告调度/GIL 获取，或返回运行时之后的收尾。
没有新增 host callback、设备同步或跨流等待来测量这些边界。

完整导出 profile 后，一键脚本自动分析 `trace_view.json`，在摘要中输出两个方向的 `GAP` 表，
并在本轮 `--output-dir` 中写入 `trace_with_gaps_<backend>_<序号>.json`。
该文件保留原始 traceEvents，新增三段派生时间块（不是设备上实测的新 MSTX 事件）：

| 派生时间块 | 区间含义 |
| --- | --- |
| callback_tail | 上一次内层 update_weights 结束，到它的 Python host_callback 范围结束 |
| between_callbacks_unattributed | 上一次 Python 回调范围结束，到下一次 Python 回调范围开始 |
| callback_head | 下一次 Python 回调范围开始，到其内部 update_weights 开始 |

三段之和等于相邻内层范围的间隔。回调头部可能包含预取快照；尾部包含 Python 收尾与计时器工作。
本工具按同一进程内的时间相邻事件配对，只比较 prefetch/reactive 切换；
不声称这些回调属于同一层或同一请求。每个派生块参数保留两端 layer/call，可自行核对。
并发重叠或不满足包含关系的样本跳过，并报告 skipped。

`memcpy_overlap_us` 统计间隔中名称含 MEMCPY_ASYNC 的完整事件的时间覆盖并集，
不是各 Stream 耗时简单相加。需在原图确认这些事件是否位于设备轨道、方向和归属；
此字段不区分路由、权重、映射，不代表因果，也不能与上述三段再次相加。
没有覆盖 MEMCPY 的时间不等于设备完全空闲，更不等于 GIL 等待。

新版时间线若支持在 MindStudio 中单独导入 JSON，可导入派生文件并搜索 `gap/`，
同时查看原有 CPU/NPU 轨道；若该版本只接受原始 profile 文件夹，则仍导入原始目录，
查看新增的真实 MSTX host_callback 标记，派生 JSON 可在兼容 Chrome Trace 的查看器中打开。
导入兼容性和实际阶段数据均待绿区验证。

首轮建议：

```bash
python -B script/run_verified_cpp_tests.py --only inference --mstx --npu 1 --requests 1 --profile-timeout 300 --output-dir bench_results/mstx_boundary_01
```

原始文件不改写。导出不完整时不自动分析，可在确认导出完成后单独执行：

```bash
python -B script/analyze_callback_gaps.py /实际路径/trace_view.json bench_results/trace_with_gaps_manual.json
```

分析器目前支持导出为 `ph=X` 完整范围的 MSTX 格式，其他格式会提示未验证，
不伪造配对结果。读取流式进行，但有最多 100 万个候选区间的保护限制；建议先采集一个请求。
返回摘要中的 GAP 表、pairs/skipped、marker_errors 即可，无需传输 profile 文件。

设备传输、预测算子和事件等待仍查看原始设备轨道。目前没有把这些阶段全部赋予独立 MSTX 名称：
当前流程在 profiler 开启前已经捕获图，torch_npu 2.10.0 的 MSTX 实现会在未开启采集时跳过打标，
因此不能简单在捕获期 Python copy_/wait_event 两侧打标就宣称测到了重放耗时。
依据见 [MSTX 实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/profiler/mstx_mgr.cpp)。
报告调度、GIL 获取及设备恢复仍包含在未归因区间，需要底层观测能力才能进一步独立测量。

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
