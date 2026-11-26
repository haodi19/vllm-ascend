# MoE 原生 C++ 权重更新（实验扩展）

有效性能结果统一见 [MoE 卸载实验总记录](moe_experiment_results_zh.md#native-20261124)；本页不再记录独立实验。

<a id="moe-原生-c-权重更新实验扩展"></a>
<a id="实现范围"></a>

## 实现与兼容范围

2026-11-26 增加 `native_pipeline`，在 native 的共享最终 scale 布局和 C++ 任务基础上，
参考同事的提前预测、规划与 DMA 提交分离思路：

- 当前层按需请求 publish 后，立即计算下一层预测专家并暂存、发布预取请求，再等待当前层权重。
  预测使用原有 MoE 输入和 FATE/哈希路由逻辑，没有改专家数、缓存容量、预测数量或 MTP。
- 原 C++ worker 负责规划，另一条 C++ worker 负责 DMA 提交。各捕获位置拥有独立的路由和映射暂存缓冲，
  预取任务必须在对应同代按需任务提交后才能提交；零加载任务也经过相同队列。
- 每批不再调用 Host stream synchronize。加载流先搬权重/scale、再搬映射，最后用完成算子写入
  NPU 上的代际标记；原严格 wait 等待这个标记，保证完成前不能消费权重。
  Host 的“已入队”标记与设备的“可消费”标记分开，不能混用。
- eager/prefill 使用 `native_full` 的已注册批量提交；上述提前预测和异步流水仅用于捕获的 decode 图。
  原 `python`、`native`、`native_full` 路径和 `--case all` 的原三组选择保留。

实现参考 [同事的调度代码](https://github.com/yxk-fdu/vllm-ascend/tree/361c929b6b5e7034a7dbddce9a269b0ede082945)。
这里保留现有 MTE publish/wait，新增加载流完成算子，没有直接替换为同事的 ACL value wait/write。
它减少串行等待并提供重叠机会；不能仅凭代码宣称已经消除所有空泡。

新三组入口为 `--case pipeline`：Python MTE → 现有 native → native_pipeline。
每组仍在同一服务内先完成 5 次预热、20 次无 profiler/MSTX 计时，再采 profile，
默认依次运行 token5 和 token128。先运行小张量检查：同一图反复切换专家，验证
权重/scale、映射、消费位置快照、完成代际和持有 GIL 时的独立推进；失败不启动模型。
本地 CPU 回归不能代替目标 NPU 验收，目前没有本版本的服务器性能结果。

在容器内运行：

```bash
source /data/libvirt/images/l00992607/moeoffload/env.sh
cd /data/libvirt/images/l00992607/moeoffload/vllm-ascend
git pull --ff-only origin dev/moeoffload-worker-poll
bash script/run_native_moe_suite.sh --case pipeline --npu 6 \
  --output-root "$PWD/bench_results/native_pipeline_01"
```

脚本自行转入后台；查看 `native_pipeline_01/run.log`，完成后返回
`native_pipeline_01/summary.txt`。三组各自目录位于 `token5/` 和 `token128/` 下，
可视化仍集中在各 token 目录的 `profile/` 下，自动对齐并识别加载流。
分开执行时用同一新目录，每条结束后再执行下一条；默认每条都运行 token5、token128：

```bash
bash script/run_native_moe_suite.sh --case python --npu 6 --output-root "$PWD/bench_results/native_pipeline_02"
bash script/run_native_moe_suite.sh --case native --npu 6 --output-root "$PWD/bench_results/native_pipeline_02"
bash script/run_native_moe_suite.sh --case native_pipeline --npu 6 --output-root "$PWD/bench_results/native_pipeline_02"
```

仅测一种长度可在对应命令后加 `--tokens 5` 或 `--tokens 128`。编号已存在时使用下一未占用编号。

统计中 `native_pipeline` 的 `host_callback/update_weights` 只包含规划与入队；
`load_submit/copy_submit` 在另一条 C++ 线程，按相同 call ID 统计，`synchronize` 不存在，不能补零。
原始 Host 间隔、计算早晚比例及两种重统计继续输出，但此时“预取 Host 结束”是规划入队结束，
不是 DMA 完成。完成顺序检查使用 publish 与入队开始的必要约束，不再错误要求 Host 返回一定早于设备放行。
这些重统计仍是基于对齐边界的条件分析，不是新测 Decode/TPOT 收益。
加载流空泡保留全窗口的 10–5000 μs 口径；异步 DMA 已离开 Host 规划窗口，
因此不再用该窗口归因预取→按需方向，摘要会明确说明不可用。以实际 Decode/TPOT 和全窗口空泡验证收益。
全窗口仍按加载流全部设备任务计算占用，包括新增的完成算子和监控 D2H；空泡缩短不能单独证明权重带宽提高。

当前实现用于单卡 MTE、C++ LRC、普通加载流和消费者处严格等待的 decode 图重放。`host_update_backend` 默认仍为 `python`；选择 `native` 后，专家规划、ACL 搬运提交、同步与映射更新由原生线程执行，执行期间不调用 Python，也不获取 GIL。初始化和图注册仍由 Python 完成。

2026-11-25 增加 `native_full`，补齐运行时 eager/prefill 的 C++ 批量搬运提交，保留当前 `native` 作独立基线。三组含义如下：

| 配置组 | Decode 图重放 | eager/prefill 搬运提交 | scale CPU 缓冲 |
| --- | --- | --- | --- |
| `python`：Python MTE | Python 更新，C++ LRC | Python/Torch 逐项提交 | 原 Python 布局 |
| `native`：当前 native | C++ 规划、ACL 提交与同步 | Python 构造任务、Storage 字节拷贝 | 共用最终布局缓冲 |
| `native_full`：完整提交优化组 | 沿用当前 native | 初始化登记地址，C++ 整批 ACL 提交 | 复用同一份最终布局缓冲 |

新增路径参考同事仓库 `csrc/lrc_policy/native_dma.h` 的常驻地址表和 `native_dispatch.h` 的逐专家、逐字段 ACL 提交方式。prefill 的完整专家列表也在 C++ 中预建；每次只按层选择表，不重新切 Tensor/Storage 或构造每个字段的 Python 任务。eager 一次传入 `(slot, expert)` 计划，进入 C++ 后释放 GIL，直到整批提交结束。所有字段先验证，现有 stream 同步位置继续保留，mapping 与消费者严格等待的协议不变。

scale 注册允许 ND（2）与 NCHW（0）这两种基础格式编号不同，但必须通过 dtype、逐专家 shape/stride、无重叠密集布局及存储范围检查；打包格式仍不接受。权重的格式一致性检查保持严格。NPU 小张量测试同时覆盖真实的 Prefill 地址登记和字节搬运，以检查“Decode 三维分配后 reshape/transpose”与“Prefill 直接四维分配”产生的格式差异。

这里的“完整”限定为本轮权重/scale 搬运提交优化。模型初始化、eager 专家规划和图注册仍使用 Python；Decode 回调与当前 native 相同。预期可减少 eager/prefill 的提交开销，Decode 是否改善由实测决定。同事 CPU MoE 的计算准入策略、独立 DMA 队列和设备事件发布属于另一套调度协议，本轮没有移植，不能据此声称按需与预取之间的空泡已消除。

截至 2026-11-25，本次实现包含以下配套修改：

- MXFP scale 在 checkpoint 加载时直接写入最终 dtype/stride 的 pinned CPU 缓冲。native 注册地址时必须复用该缓冲，不再保留第二份 scale；不满足布局条件就报错，禁止静默分配副本。
- native 的 eager/热专家预加载与 prefill 共用这份 CPU 缓冲。prefill scale 目标池保留对应的物理布局；布局、类型一致时使用带准确槽位偏移的 Storage 字节拷贝，避免非连续 CPU scale 到连续 NPU scale 的隐式转换。
- 其他已支持量化格式若需要最终转换，逐专家替换管理器中的原缓冲；FP32 派生 scale 单独计数。它是另一种必要表示，与同一表示的布局副本分开报告。当前 DeepSeek-V4-Flash MXFP 路径没有这项派生分配。
- 原 Python 配置继续使用原分配与搬运路径。注册后的地址受现有重加载保护约束，任务完成和错误传播沿用严格 MTE 握手。

上述修改不改变专家数量、缓存容量、预取数量或 MTP 配置。性能结论需来自新旧版本实测，不能将源码完成视为收益验收。

<a id="运行初次验证"></a>

## 验证与结果读取

继续使用 `script/run_native_moe_compare.py`，默认顺序运行 Python MTE、当前 native、native_full 三组。它先运行 CPU 回归、NPU scale 字节一致性、整批 eager/prefill 提交、严格图重放及 GIL 独立推进检查，再运行性能请求和同服务内的 profile。结果统一汇总到指定目录的 `summary.txt`，保留 Decode、HTTP E2E、TTFT、TPOT、Host 函数耗时、Host 间隔及加载流空泡。

每组只启动一次模型服务：预热 → 正式计时 → 同一服务开启 profile/MSTX → 采集 → 关闭服务。计时前后检查采集状态及 Python/C++ 标记计数，要求采集关闭、标记计数不增长；原有包装与开关检查仍保留。采集阶段不重新加载权重、不重新捕获模型图，也不做第二轮预热。NPU 前置检查包含同一张图上关闭→开启→关闭原生标记的验证。显式使用 `--profile-only-from` 补采旧结果时，才单独启动采集服务。

构建自动复用：首次生成并通过检查后，在仓库的 `build/native_moe_handshake/verified_build.json` 保存源码、Python/依赖、CANN/工具链和 SoC 的指纹，以及已安装扩展和握手库的哈希。后续组、token5/token128 及新实验目录复用这套产物，打印 `BUILD_CACHE_HIT`，不再执行 pip 重装和 CMake 构建；CPU/NPU 前置检查仍会执行并写入 `build.log`。缓存位于持久构建目录，不依赖任何旧实验目录。

指纹变化、库缺失或已安装扩展被替换时才重新构建，仍使用新的 `build_XX`，避免重复链接被 AscendC 改写过的中间文件。失败构建或未通过检查的产物不会登记为可复用缓存。更新到此版本后，尚无校验记录的旧产物会触发一次构建；之后测试命令不用增加参数。

需要分开运行时，分别传 `--case python`、`--case native`、`--case native_full`，三条命令可以使用同一结果目录，逐组追加。每条命令均独立完成该组计时和 profile；`--case both` 保留原 Python/native 两组入口。单组计时验证该组请求一致性，汇总保留各组测量来源，不能把分开启动称为同次配对测量。单组计时完成后如需补采，可用 `--profile-only-from` 配合对应的 `--profile-case` 和新输出目录；三组补采传 `--profile-case all`。

完整测试使用 `script/run_native_moe_suite.sh`：后台依次运行 token 5、128，每轮均比较三组，每组 5 次预热、20 次正式请求并独立采 profile。它继承当前终端环境；在容器内执行，下面的 `--npu 6` 为物理卡号，结果目录已存在时改为下一个编号：

```bash
source /data/w00919836/moeoffload/env.sh
cd /data/w00919836/moeoffload/vllm-ascend
bash script/run_native_moe_suite.sh --case all --npu 6 \
  --output-root /home/w00919836/results/native_three_way_01
```

启动后立即返回 PID，关闭终端仍继续运行。查看实时输出及完成后的合并摘要：

```bash
tail -f /home/w00919836/results/native_three_way_01/run.log
# 全部结束后复制这个文件即可，包含 token 5、128 两轮摘要。
cat /home/w00919836/results/native_three_way_01/summary.txt
```

独立运行某一组也用此入口，默认仍测 token 5、128。以下三条共用 `native_three_way_02`，等上一条后台任务结束再执行下一条：

```bash
bash script/run_native_moe_suite.sh --case python --npu 6 \
  --output-root /home/w00919836/results/native_three_way_02

bash script/run_native_moe_suite.sh --case native --npu 6 \
  --output-root /home/w00919836/results/native_three_way_02

bash script/run_native_moe_suite.sh --case native_full --npu 6 \
  --output-root /home/w00919836/results/native_three_way_02
```

仅测一种输出长度时，在相应命令追加 `--tokens 5` 或 `--tokens 128`。单组入口允许补上同一总目录中尚未运行的组或 token 长度；重复组（包括失败后留下的目录）会在启动前被拒绝。完整 `all/both` 入口仍要求新总目录。同一总目录用 `.suite.lock` 防止并发写入，后台结束时释放锁。运行输出追加到 `run.log` / `run_details.log`；各组、各 token 和根目录的 `summary.txt` 从已保存结果重新生成，每项只列一次，不拼接历史控制台输出。

默认服务端口为 8156、独立计时端口为 18156；占用时传入 `--port` 和 `--timing-port`。批量入口沿用 5 GiB 剩余空间检查，profile 导出最多等 600 秒。结果位于根目录的 `token5/`、`token128/`；`status.txt` 逐组逐轮记录退出码。`BATCH_EXIT` 保留历史失败，`run.log` 中的 `LAST_JOB_EXIT` 表示刚结束的任务；摘要分别标记计时、profile 和重统计的完整性，没有测完三组时不能作为完整对照。第一种长度失败后仍尝试第二种长度。

每个 token 目录内，三组日志、命令与数据分别归档，三组可视化集中放在 `profile/`。例如：

```text
native_three_way_01/
├── run.log                         # 后台实时输出
├── summary.txt                     # token5、token128 完整汇总
├── status.txt
├── token5/
│   ├── summary.txt                 # 三组计时、Host 间隔、空泡及两种重统计方案
│   ├── run_details.log             # 逐请求JSON、前置检查与失败详情
│   ├── build.log
│   ├── python/                     # 本组summary.txt、inference.log、command.json、timing_result.json
│   ├── native/                     # 同上，另含 profile.log、profile_counters.json 等
│   ├── native_full/                # 每组均保留独立 profile 报告及 bubble_details.log
│   ├── profile/
│   │   ├── python/<采集目录>/
│   │   ├── native/<采集目录>/
│   │   └── native_full/<采集目录>/
│   └── prefetch_overlap.json       # 三组计算早/晚完成与重统计的逐样本数据
└── token128/                       # 结构与 token5 相同
```

导入对应组的 `<采集目录>` 即可查看已对齐的可视化。单组测试也使用相同结构，只生成所选组。旧版平铺的日志目录仍支持补采及离线分析，不需要手动搬文件。

常规测试的服务输出全程写入本组 `inference.log`，`profile.log` 记录采集阶段的控制与分析信息；它不再表示第二次模型启动。返回结果仍只需根目录的 `summary.txt`。

原 `run_native_moe_compare.py` 入口保留，可按 `--case`、`--max-tokens` 和 `--output-dir` 前台独立执行。计时阶段关闭 profiler/MSTX；profile 使用同次时钟对齐并自动识别加载流。`native_full` 必须同时出现原生图任务完成、共享 scale 校验和 `FULL_SUBMIT executed=` 记录，防止只注册但未执行新路径的测试被接受。

每次任务结束后，脚本默认重新生成 `summary.txt`，Python/native 分开运行时也一样，无需额外开关。摘要包括以下已完成的分析；失败或缺失阶段单独标记，不补零，也不删除已完成的计时数据：

- Host 函数与分项耗时，以及原始预取→按需间隔。
- 计算相对于预取 Host 结束的早/晚、不确定、无法识别样本数量、占比和间隔均值。
- 方案 1：删除明确晚完成样本；方案 2：扣除这些样本从预取结束至计算到达的经过时间。
- 两方案的保留样本数、均值及相对原始均值变化，分别保留直接判定和保守判定。早、不确定、无法识别样本保留原值，不另加 IQR 或时长截断。

逐样本结果保存为 `prefetch_overlap.json`，详细解析输出为 `prefetch_overlap_details.log`。计算边界不足时明确输出 `OVERLAP_FAILED` 或部分覆盖状态，已有计时、间隔和 profile 保留；条件坐标估算继续标明顺序检查未通过，不能当成精确测量或新的 Decode 收益。`--skip-profile` 不产生这两种方案的数据，会明确标记缺失。

旧摘要混入重复输出或失败堆栈时，等后台任务结束后执行下面一条命令即可整理，不重跑实验、不解析大 trace、不查询 NPU：

```bash
python -u -B script/summarize_native_moe_results.py /home/w00919836/results/native_three_way_02
```

命令直接打印核心表格，并更新根目录、各 token 和各组的 `summary.txt`。旧版原始摘要自动归档到同目录 `run_details.log`，各组 JSON、profile 和原日志不删除。返回根目录 `summary.txt` 即可；若只返回一组，可用对应的 `token5/native/summary.txt` 等文件。整理仅重排已有统计，缺失的 profile 或重统计不会自动补采。

已完成的旧结果可直接离线补算，无需重新加载模型或采集：

```bash
python -u -B script/reanalyze_native_prefetch_overlap.py \
  /home/w00919836/results/native_shared_scale_python_01 \
  /home/w00919836/results/native_shared_scale_native_01 \
  --output-dir /home/w00919836/results/native_shared_scale_overlap_01
```

该命令把核心表格直接打印到终端，并保存到新分析目录的 `summary.txt`；不修改原 profile 和旧摘要。目录已存在时改用下一个未占用编号。

新版 `NATIVE_SCALE_MEMORY` 记录会出现在计时与 profile 的摘要中：

| 字段 | 含义 |
| --- | --- |
| `policy=shared_final_layout_v1` | 本次共用最终布局缓冲的实现标识 |
| `duplicate_bytes` | 相同 scale 表示的额外布局副本；必须为 0 |
| `reused_bytes` | native 直接复用的 CPU scale 数据字节数 |
| `replaced_bytes` | 其他量化格式替换原缓冲的数据字节数；不与旧缓冲长期并存 |
| `derived_bytes` | 必要的 FP32 派生表示占用；MXFP 本次应为 0 |

缺少该记录或重复副本不为零时，脚本停止验收。`scale_buffer_bytes` 旧字段保留用于兼容，当前只计额外派生表示；不能据此推算进程 RSS 或 pinned allocator 的缓存占用。

<a id="验证状态"></a>

本地 CPU 测试覆盖加载分片顺序、原缓冲地址复用、prefill/eager 非零槽位、布局不匹配拒绝、替换后旧缓冲释放和原生任务执行。NPU 检查已集成到原测试入口；需要在目标服务器上运行后才能确认新版本的模型输出与性能。既有实验数据仍对应当时版本，不回填为本次改动的效果。
