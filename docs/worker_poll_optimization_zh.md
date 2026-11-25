# 基于阻塞旧版的 worker 轮询改造

本段实验正文与数据已移至 [性能实验总记录](moe_experiment_results_zh.md)。

<a id="基线与阅读准备"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="一取消空闲-yield"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="原逻辑与修改内容"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="测试方法"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="测试结果关闭内部计时判断收益"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="测试结果开启内部计时分析原因与测量扰动"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="结论"></a>

[性能实验总记录](moe_experiment_results_zh.md)

本段实验正文与数据已移至 [性能实验总记录](moe_experiment_results_zh.md)。

<a id="二减少扫描范围静态集合的独立测试"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="原逻辑与修改内容-1"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="测试方法-1"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="当前结果"></a>

[性能实验总记录](moe_experiment_results_zh.md)

本段实验正文与数据已移至 [性能实验总记录](moe_experiment_results_zh.md)。

<a id="三减少空闲扫描-mutex独立测试已验证"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="原逻辑与修改内容-2"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="测试方法-2"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="当前结果-1"></a>

[性能实验总记录](moe_experiment_results_zh.md)

本段实验正文与数据已移至 [性能实验总记录](moe_experiment_results_zh.md)。

<a id="四工作线程绑核功能通过当前-cpu-未显示稳定收益"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="原逻辑与修改内容-3"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="测试方法-3"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="当前结果-2"></a>

[性能实验总记录](moe_experiment_results_zh.md)

本段实验正文与数据已移至 [性能实验总记录](moe_experiment_results_zh.md)。

<a id="五有效项组合对照已完成未观察到叠加收益"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="本次结果"></a>

[性能实验总记录](moe_experiment_results_zh.md)

<a id="阶段判断"></a>

[性能实验总记录](moe_experiment_results_zh.md)

## 六、MoE 可选接入：动态有效前缀与三个独立开关

### 调用与时序

独立实验的静态集合不能直接接入模型：实际任务槽位在工作线程启动后才逐次分配。
MoE 采用只增不减的有效前缀 `[0, count)`，初始 count=1，包含初始化握手的槽位 0。
槽位由 `MteHostDispatcher.launch()` 从 1 连续分配，无需扫描未分配的剩余容量。

1. 创建 Session，注册原有的全部包装函数。根据配置选择原子预检查、yield/spin；
   若选择 allocated，启动前调用 `configure_scan_prefix(1)`，然后启动工作线程并完成槽位 0 握手。
2. 外层卸载代码仍通过 dispatcher 的 `launch(stream, callback, args)` 提交任务。
   `launch()` 取 `next_slot`，先把回调函数和本次参数写入 `entries[slot]`。
3. 调用 C++ 方法 `expand_scan_prefix(slot + 1)`，在管理锁下验证 Session 正在运行、
   新槽位已经注册、上限不缩小且不超过容量，然后以 release 原子写发布新上限。
4. 工作线程每轮以 acquire 原子读获取上限，按原来的编号顺序扫描这个前缀。
   若当前一轮读到旧上限，下一轮会再次读取；请求代号会保留，不依赖一次性的唤醒通知。
5. `launch()` 在扩大上限成功后，才提交 NPU 的发布/等待算子。因此设备使用新槽位前，
   任务记录已经保存、扫描范围已经发布。失败时不提交设备请求。
6. 图重放仍使用捕获时的槽位地址，不重新调用 Python `launch()` 分配槽位。
   后续捕获新图会继续扩大上限；旧图槽位一直保留在前缀中，不收缩或复用。

这不是只扫描“当前这一轮有请求”的槽位，而是扫描“已经分配、今后可能收到请求”的所有槽位。
例如已使用槽位 0～85，就扫描 86 个，而不是固定 2048 个或只扫描其中一个。
CPU 与 NPU 的 request/done 协议、设备算子、权重拷贝、SyncLaunchStream、H2D 同步和 watchdog 未改。
Python 任务记录仍由 GIL 保护，前缀原子变量只控制 C++ 扫描范围，不替代任务记录本身的线程保护。

### 开启方式

三个键放入原 `--additional-config` 的 `expert_offload_config` 内：

| 配置键 | 默认值 | 可选值及作用 |
| --- | --- | --- |
| `host_sync_scan_mode` | `all` | `allocated`：扫描已分配任务槽位前缀 |
| `host_sync_idle_probe` | `locked` | `atomic`：无任务时先做原子检查，跳过空闲锁 |
| `host_sync_idle_mode` | `yield` | `spin`：无任务整轮结束后不主动 yield |

仅 `host_sync_backend="mte"` 使用这些配置。callback 路径保持原行为。
首轮建议 `allocated + locked + yield`；三个开关也可独立或组合启用，
例如 `allocated + atomic + spin`，但组合并非已证实的推荐性能配置。
现有扩展缺少所选接口会明确要求重新构建，不静默忽略开关。启动日志打印实际所选配置。

### 绿区一次执行

在 worker-poll 工作目录更新后运行下列命令。runtime-source 指向同一容器中原来可用的仓库，
只补齐本目录缺少的 `_build_info.py`、可选版本文件、动态库及完整 `_cann_ops_custom` 包；
不复制 Python 业务源码，不覆盖已有文件，不改变原目录。
test-script 使用原可运行脚本中的模型路径等配置，运行器把当前 worker-poll 目录置于导入路径前部。

```bash
python -B script/run_moe_host_sync.py --backend mte --npu 7 \
  --runtime-source /home/l00992607/vllm-ascend \
  --test-script /home/l00992607/vllm-ascend/script/test_graph2.sh \
  --worker-scan allocated --worker-probe locked --worker-idle yield \
  --requests 1 --warmup-requests 1 --request-timeout 180 --profile-timeout 120
```

脚本重装当前 Host 扩展、构建设备库、运行 CPU worker 与 MoE 调度测试后启动模型。
profile 保存在本轮结果目录的 `profile/mte`，控制台给出 summary 路径。
实际旧版 scale 阻塞仍可能在首次请求出现；本轮是接入验证，不应把单测通过当作它已经修复。
若阻塞，返回 summary 中的 watchdog 栈及配置；不要把超时结果当性能数据。
切换原子预检查用 `--worker-probe atomic`；切换忙轮询用 `--worker-idle spin`；
恢复原 worker 行为用 `--worker-scan all --worker-probe locked --worker-idle yield`。

### 当前验证状态

已新增动态前缀增长、旧/新槽位交替执行、缩小/越界/停止后增长拒绝等原生测试，
以及任务记录先于前缀发布、前缀发布先于设备提交、配置映射和默认兼容的 CPU 测试。
蓝区只验证 Python 调度逻辑；原生扩展及实际模型尚待绿区执行。

本段实验正文与数据已移至 [实验总记录](moe_experiment_results_zh.md#worker-model-20261031)。

<a id="7-mte普通流下逐项验证真实模型收益2026-10-31"></a>

[7. MTE普通流下逐项验证真实模型收益（2026-10-31）](moe_experiment_results_zh.md#worker-model-20261031)

<a id="当前基线与实验范围"></a>

[当前基线与实验范围](moe_experiment_results_zh.md#worker-model-20261031)

<a id="五组顺序实验"></a>

[五组顺序实验](moe_experiment_results_zh.md#worker-model-20261031)

<a id="绿区一次执行-1"></a>

[绿区一次执行](moe_experiment_results_zh.md#worker-model-20261031)

<a id="结果判断与当前状态"></a>

[结果判断与当前状态](moe_experiment_results_zh.md#worker-model-20261031)

<a id="首轮采集缺失与重测要求"></a>

[首轮采集缺失与重测要求](moe_experiment_results_zh.md#worker-model-20261031)

<a id="多请求整体时延与单请求profile分开测量"></a>

[多请求整体时延与单请求profile分开测量](moe_experiment_results_zh.md#worker-model-20261031)

<a id="atomic-独立服务交替复验"></a>

[atomic 独立服务交替复验](moe_experiment_results_zh.md#worker-model-20261031)

<a id="allocated-与-atomic-组合验证"></a>

[allocated 与 atomic 组合验证](moe_experiment_results_zh.md#worker-model-20261031)

<a id="组合测试结果2026-10-31"></a>

[组合测试结果（2026-10-31）](moe_experiment_results_zh.md#worker-model-20261031)

<a id="回调mte-默认与-mte-组合三组对照"></a>

[回调、MTE 默认与 MTE 组合三组对照](moe_experiment_results_zh.md#worker-model-20261031)

<a id="增加-python-lrc-与-debug-开启的早期配置对照"></a>

[增加 Python LRC 与 debug 开启的早期配置对照](moe_experiment_results_zh.md#worker-model-20261031)

## 独立测试的绿区执行

首次建立独立目录，不在原目录或阻塞复现目录 reset：

```bash
cd /home/l00992607/vllm-ascend
git fetch origin
git worktree add -b dev/moeoffload-worker-poll \
  /home/l00992607/vllm-ascend-worker-poll \
  origin/dev/moeoffload-worker-poll
cd /home/l00992607/vllm-ascend-worker-poll
python -B script/run_worker_poll_tests.py --npu 7
```

后续在该目录执行 `git pull --ff-only` 更新；NPU 编号按空闲设备修改。
脚本不含 Git 操作，结果写入新建的 `bench_results/worker_poll_<编号>/summary.txt`，
只返回汇总文本。单独补测多槽可加 `--slots 2048`。

独立握手不导入模型，无需复制公共算子包、热点 JSON 或安装整个插件。
脚本会安装本分支 Host 扩展，各工作目录共用容器 Python 环境，不能同时测试不同版本。
切回其他目录运行前，先在那个目录重装其扩展：

```bash
python -m pip install ./csrc/host_sync --no-build-isolation --no-deps --force-reinstall
```

## 五组真实模型后台复测（token5 与 token128）

在已配置环境的容器内，从项目目录执行：

```bash
bash script/run_moe_five_way_night.sh --work-root /data/w00919836/moeoffload --npu 1
```

脚本自动加载工作目录下的 `env.sh`，后台顺序运行 token5、token128，每轮包含五个独立服务：

| 配置名 | Host机制 | LRC | debug |
| --- | --- | --- | --- |
| callback_legacy | 回调 | Python | 开 |
| callback_python | 回调 | Python | 关 |
| callback | 回调 | C++ | 关 |
| mte_default | 内存共享 | C++ | 关 |
| mte_combined | 内存共享 | C++ | 关 |

两组 MTE 使用普通加载流、消费者处严格等待；worker 分别为
`all/locked/yield` 和 `allocated/atomic/yield`。
前两组只差 debug，第二、三组只差 LRC 实现。全部配置在当前代码上运行，不切换历史提交。

每组预热 5 次、独立计时 20 次，之后采集 1 次 profile。
token128 设置 `ignore_eos=true`，token5 保持原来的 EOS 行为。
默认服务端口为 8156、计时端口为 18156，可用 `--port`、`--timing-port` 修改；不会停止占用端口的其他服务。

主时延为无 profiler 的 Decode 时间：最后非空文本块到达减去首个非空文本块到达。
完整端到端、TTFT、TPOT 同时保留。计时窗口不启动 MoE MSTX，前后检查 collector
未激活、无进行中的标记回调，且累计 MSTX 标记调用数没有增加；检查不通过时不作为无插桩结果。
随后单独开启 profiler、MSTX 和同次采集时钟校准，保留原始与对齐后的可视化。
miracle TPOT 使用 MTE 组合组同次 profile 请求估算全部 Host 间隔消失的情形，
只与同次带采集 TPOT 比较，不作为无采集请求的实测加速。

原 Host 间隔、加载流空泡和分项计时继续保留；MTE 两组额外自动统计：

- 预取→按需方向，计算推进早于、晚于预取回调结束的数量及比例；不确定、无法识别分别列出。
- 晚完成样本中预取零加载、任一端零加载的数量及比例，未知加载量不当作零。
- 去掉晚完成样本、扣除晚完成样本中剩余计算等待时间两种统计，原始值保留。
  同时列出直接判定和带校准保守余量的判定；全样本统计保留无法判定样本的原值，另列仅确定样本。

上述边界分析只用于已核查的 MTE publish/wait 流程，回调组不套用该配对规则。
分析读取本轮对齐采集，不更改原始时间戳，不用旧实验填补缺失数据。

输出目录默认为工作目录下的 `bench_results/night_five_way_<时间>_<PID>`，也可用
`--output-root` 指定一个尚不存在的目录。启动命令会打印 PID 和 `tail -f` 命令。
两轮结束后，该目录的 `summary.txt` 汇总 token5、token128 全部结果，复制这一份即可；
每轮自己的 `summary.txt`、`experiment_tables.md` 和详细日志仍保留。
可视化导入对应轮次的 `profile_aligned/<配置>/<python或cpp>/<采集目录>`。
