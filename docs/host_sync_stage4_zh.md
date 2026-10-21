# 本地 CPU/NPU 交接扩展：第四阶段

前三阶段已经验证 Host 控制内存、MTE 标记交接、图捕获与重放的正确性。
本阶段比较两种交接机制的完整小任务成本，并单独采集 profile。
本阶段代码尚待绿区运行，不预先给出加速结论。

## 两组实验

- `callback`：图内通过项目使用的 `torch_npu.npu._launch_host_func` 执行 CPU 回调。
- `mte`：图内执行发布和等待两个 AIV 算子，CPU 工作线程轮询 generation 并执行回调。

两组回调均只增加一次 Python 计数器，不做 LRC、日志输出、专家权重搬运或模拟繁重计算。
计时组在回调内额外读取两次 Host 时钟，记录回调主体耗时。该主体时间不包含进入 Python
前获取 GIL、框架包装、完成标记发布和 NPU 后续恢复的时间，也含时钟读取本身的测量扰动。

两组不是相同的设备算子：callback 组依赖运行时回调机制，MTE 组包含两个 AIV 算子及
标记读写。这正是两个完整交接方案的成本，但不能把差值直接称为“纯同步开销”，也不能
简单减去回调主体时间就得到 runtime 的固定成本。MTE 组还占用 CPU 工作线程及 AIV 资源，
后续需在实际推理并发下继续验证。

## 计时方法与字段

先普通执行一次，再捕获图。捕获前后检查回调数和设备结果不变。
默认图预热 20 次，测量 200 次；每轮同步并校验回调次数，MTE 组还检查 generation 和
设备成功标记。结果检查引起的 D2H 在计时结束后执行，不纳入该轮指标。

每种后端使用独立子进程，默认顺序为 callback、mte、mte、callback，每种各测两组。
这样减少固定先后顺序的影响，但不保证消除频率、系统负载或调度波动。

| 控制台字段 | 单位 | 含义 |
| --- | --- | --- |
| `event_interval` | 微秒 | 同一 NPU 流上，包围图重放的两个计时 Event 之间的时间；包含交接、CPU 回调与设备恢复，也可能包含提交间隙 |
| `replay_sync_host` | 微秒 | Host 从调用图重放前，到结束 Event 同步返回的墙钟时间，包含提交与 Host 等待 |
| `callback_body` | 微秒 | Python 回调内部计数操作及计时扰动，不是完整回调机制耗时 |
| `mte_minus_callback_us` | 微秒 | 两组均值之差，负数表示这个指标下 MTE 更低 |

每组输出 mean、p50、p95、min、max。最终 `COMPARISON` 汇总等样本数各组的均值。
不将 profiler 运行结果混入统计；不使用整个子进程的运行秒数衡量交接性能。

## 执行与 profile

更新代码后，在容器项目根目录执行：

```bash
python -B script/run_host_sync_stage4.py --npu 1
```

脚本自动在新目录构建，复跑普通执行与图正确性检查（正常场景各 10 轮），然后执行
无 profiler 的性能对照。两组计时数据和 `COMPARISON` 输出后，再分别启动 profile 子进程。
默认每个 profile 采集 5 次重放，预热不在采集范围内。

profile 默认保存至：

```text
moe_profile/host_sync_stage4/run_<唯一编号>/callback/
moe_profile/host_sync_stage4/run_<唯一编号>/mte/
```

精确路径在控制台输出。可用 `--profile-dir` 指定根目录，脚本仍追加唯一编号以避免覆盖。
成功时 `PROFILE_RESULT` 给出 `trace_view.json` 路径，可在 MindStudio 中打开对应目录。
时间线上可检查 `host_sync_publish`、`host_sync_wait`，以及两组 MSTX 标记：

- `host_sync/<后端>/callback_body`：CPU 回调主体范围，包含 profile 下的标记扰动。
- `host_sync/<后端>/replay_and_wait`：Host 提交和等待范围，并非设备同步机制的内部细分。

profile 子进程默认限时 180 秒，可用 `--profile-timeout` 修改。
导出失败或超时会标记 `INCOMPLETE`，仍尝试另一组 profile，前面已经输出的无 profiler
计时结果保留。只有导出返回且发现 `trace_view.json` 才报告 profile 成功。
这不能代替在 MindStudio 检查采集内容是否满足分析需要。

测试不生成汇总文件，直接复制控制台 `BENCH_RESULT`、`COMPARISON`、`PROFILE_RESULT`、
`STAGE 4 SUMMARY`；失败时复制对应报错即可。脚本不包含 Git 操作。

## 保持兼容与后续边界

本阶段只新增独立实验脚本，没有改动 MoE manager、LRC、原推理脚本、原有回调机制，
也没有修改两个参考仓库。仅在交接小任务中比较，不能直接推广为模型整体时延收益。
真实路由数据、权重搬运完成顺序、多槽位和并发接入都需要后续单独验证。

实现依据：[torch_npu 回调绑定](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/npu/Graph.cpp)、
[profiler MSTX 配置](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/experimental_config.py)。
