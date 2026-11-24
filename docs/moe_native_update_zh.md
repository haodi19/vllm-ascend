# MoE 原生 C++ 权重更新（实验扩展）

新增 `host_update_backend="native"`，默认仍为 `"python"`。原回调版、MTE 默认版、MTE 组合版及原五组实验命令继续使用 Python 权重更新路径。

## 实现范围

本版面向单卡、C++ LRC、普通加载流、消费者处严格等待的 MTE 图重放。MTE 工作线程不再获取 GIL 调用 Python 回调，而是直接执行原生任务：读取已经完成 D2H 的专家 ID/分数/映射，计算命中与淘汰，提交 ACL H2D，等待加载流完成，提交映射更新，发布完成代数。设备 publish/wait 与延后预取等待的位置不变。

初始化、图构建、eager 预热和 prefill 仍使用原实现。本版不是整个推理引擎的 C++ 重写。专家 trace 和 debug 不支持新路径，配置时明确拒绝。原生诊断直接调用 MSTX C 接口，标记名沿用 `host_callback` 以便比较，实际标记的是 C++ 更新任务；不为采集而重新包装 Python 回调。原五组实验仍保持原路径。

权重沿用原来已准备的 Storage 字节布局。CPU scale 加载缓冲保持原布局，供 eager/prefill 原路径使用；native decode 注册描述符时，dtype 和物理排列相同的 pinned CPU 缓冲直接复用，其余 scale/offset/scale_bias 单独整理为目标布局。此前让原 CPU scale 缓冲直接采用设备 stride 的实现已撤回，以隔离 prefill 的搬运行为；这不是已证明的性能根因。派生 FP32 scale 也提前准备，保留原路径先转换到目标 scale dtype 再转 FP32 的舍入语义。新增内存由 `scale_buffer_bytes` 报告，复用字节数由 `scale_reused_bytes` 报告，不预设为 8 GiB，也不承诺所有量化格式都零额外副本。

目标必须是基础格式、每个专家槽位物理连续且无重叠；不支持的格式直接报错。描述符计入目标 storage offset 并检查地址范围。注册后禁止通过权重加载入口重新加载或重排已注册层；创建图任务时检查目标 tensor 身份、地址和布局，防止复用失效的描述符。更新权重需重启并重新注册；不支持外部代码绕过加载入口修改源缓冲。

异步搬运后必须同步完成再发布成功；超时或错误不提交新映射、不发布成功，原 MTE watchdog 终止服务。注册任务及其源、目标、请求缓冲一直保留到设备同步和工作线程停止之后。使用的运行时接口为 [aclrtMemcpyAsync](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0106.html) 和 [aclrtSynchronizeStreamWithTimeout](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0077.html)。

原生规划采用专家 ID 升序分配本次 miss，淘汰优先级、热度更新与预取候选选择复用原 C++ LRC 语义。它不复刻 CPython set 的物理遍历顺序，因此专家到槽位的配对可以不同；检查驻留专家集合、热度状态和槽位内容，而不能要求物理映射逐字节相同。槽位不足会报错，不继续执行缺少专家的计算。

这次参考同事实现中的提前整理 scale、复用 pinned 缓冲、冻结搬运描述符三个做法。原来的 Torch 异步 H2D 和本版 ACL H2D 都使用设备搬运能力；改变的是主机提交路径。本版保留每批同步完成后发布成功的协议，没有迁入对方的独立 DMA 队列、设备完成通知写入或提前下一层预取调度。这些改动影响依赖关系，不能直接删除现有同步来替代。

## 运行初次验证

在原来能够跑通实验的容器、已激活原环境、项目根目录执行：

```bash
git pull --ff-only origin dev/moeoffload-worker-poll
python -u -B script/run_native_moe_compare.py \
  --npu 7 --port 8156 --timing-port 18156 \
  --test-script script/test_graph2.sh \
  --max-tokens 5 --warmup-requests 5 --requests 20 \
  --output-dir /data/w00919836/moeoffload/bench_results/native_update_06
```

脚本继承当前环境，只读取 `test_graph2.sh` 的 `vllm serve` 参数，不执行其清日志、sysctl 等前置命令，不修改原脚本。模型路径、MTP、图捕获尺寸、缓存容量等继续使用该文件的设置。输出目录必须是新目录。命令解析在构建和 NPU 检查前完成，兼容末尾续行符后没有换行的脚本；仍拒绝未闭合引号等语法错误。

握手核显式使用 `CMAKE_BUILD_TYPE=Release`，CMake 工程也为未指定构建类型的调用提供 Release 默认值。构建开启 verbose；失败时，本步骤最后至多 60 行/32 KiB 日志自动进入 `summary.txt`，保留失败命令，完整日志仍在 `build.log`。

依次执行：构建两个 CPU 扩展；读取当前逻辑卡 0 的实际型号并构建匹配的 MTE 核；运行 CPU 回归；用小张量验证连续/转置/MXFP4 布局 scale 的字节拷贝和 pinned 复用；运行真实 NPU 图的 D2H→publish→原生搬运→严格 wait→映射 H2D 测试；原 MTE 组合配置与原生更新配置各启动一个服务，5 次预热、20 次正式请求。两组只有 `host_update_backend` 不同，计时服务不启用 profiler/MSTX。随后分别另起诊断服务，各预热 5 次、采集 1 次请求，统计 Host 间隔和加载流空泡。共启动四个串行服务；可用 `--skip-profile` 只测时延。

握手库默认构建在项目的 `build/native_moe_handshake/build_01/lib/libhost_sync_device.so`，之后构建使用未占用的 `build_02` 等子目录，不依赖临时结果目录。`summary.txt` 的 `HANDSHAKE_LIBRARY` 打印本轮精确路径。旧产物全部保留，避免 CANN 重用已合并的对象文件；`--build-dir` 指定这些新子目录的父目录。修改 C++ 后需让脚本重编译，不使用 `--skip-build`；此步骤只编译两个小扩展和握手核，不重新安装整个 Ascend 插件。

小图使用禁止调用的 Python 回调作为兜底哨兵，验证重复输入、miss、淘汰后的权重/scale/映射内容；Python 持有 GIL 忙等期间，必须观察到至少一次尚未完成的原生任务完成。无法形成该观测则报验证不充分，不假报通过。该测试在独立子进程运行，严格等待失败后终止子进程，不继续模型测试。

正式请求通过流式响应比较输出文本摘要、token 用量；未发现原生图任务实际完成标记时拒绝给出收益。任一步失败即停止，服务退出时只清理自己启动的进程组。`summary.txt` 汇总全部逐请求数据、均值表、原生准备内存、输出一致性与失败位置，详细构建和服务日志在同目录。

主时延为首个非空输出至最后非空输出的 decode 时延；同时保留 TTFT、完整 HTTP 请求时延和 TPOT（decode 时延除以实际输出 token 数减一）。沿用既有流式计时函数，SSE 块数不当作 token 数。所有请求保留 raw 和同规则 IQR 统计，主表列 raw 均值。这仍是单对服务的初次对照，不能代替多次交错实验和完整精度验收。不要把本轮结果混入原五组历史表格。

诊断的两组双向 Host 间隔、加载流空泡及其表格都进入 `summary.txt`。Host 间隔是相邻任务结束到下一任务开始，原生范围从 `NativeUpdate::execute()` 入口覆盖到映射提交后返回、早于 worker 发布完成标志；Python 范围沿用外层回调。保留零加载任务及 raw/3×IQR，不把短回调直接判为空载。加载流按 `--bubble-stream-name` 精确选择，默认 `Stream 56`，缺失则报错，不凭拷贝数改选 Stream 27。MEMCPY、Transpose 及其他流内任务的并集均算占用，只统计 `10 ≤ gap < 5000 μs`，保留全采集窗口与 callback 连续窗口，并输出能按相邻 Host 窗口关联的方向统计；关联不完整会明确标记。该表不等同于按专家记录排除空载后的历史表，也不能证明整卡空闲或空泡已消除。

每份采集使用同次请求前后的 Add 校准，留出样本验证通过后发布对齐的 JSON 和数据库。MindStudio 导入 `profile/python/` 或 `profile/native/` 下的 `*_ascend_pt` 文件夹；`summary.txt` 会打印精确路径。MSTX 通过 [C 接口](https://www.hiascend.com/document/detail/en/mindstudio/2610/SDK/mstxAPIReference/docs/en/api_reference/Common/mstxRangeStartA.md) 在工作线程打点，无 Python/GIL；控制 HTTP 线程和冷启动校准仍是 Python。原生小图额外验证持有 GIL 时启用 C MSTX 的任务仍能完成。诊断插桩开销只存在于独立采集服务，不能用 profile 请求时延替换无插桩主时延。

若两组时延已完成、profile 因空间不足等原因未取得，可只续采。默认要求新输出目录所在文件系统至少有 20 GiB 可用空间，再执行：

```bash
python -u -B script/run_native_moe_compare.py \
  --npu 7 --port 8156 --timing-port 18156 \
  --profile-only-from /data/w00919836/moeoffload/bench_results/native_update_06 \
  --output-dir /data/w00919836/moeoffload/bench_results/native_profile_01
```

续采读取原保存的两组启动命令、已完成请求及库路径，不重新编译、不重新做 20 次时延测试，也不重新解析当前 `test_graph2.sh`；每组仍启动模型、预热后采集。`max_tokens` 从原请求恢复，使用原环境和原物理卡执行。原报告不变，新 `summary.txt` 以 `TIMING_SOURCE` 标明沿用的历史时延，合并新采的 Host 间隔、空泡和可视化路径。续采与原计时不是同时进行，服务器状态可能不同；原数据缺组、输出不一致或库已丢失会停止，不以部分结果拼出对照。可把新输出目录改到另一个有空间的挂载盘，不能仅在同一满盘上换个目录名。

只有约 7 GiB 可用时，可增加 `--min-profile-free-gib 5`，将本次门槛降到 5 GiB；默认值和其他实验入口不变。续采入口及每组采集服务启动前均使用此门槛，实际设置写入 `summary.txt`。这只是启动前剩余空间门槛，不限制 profile 大小，也不保证两组采集和对齐导出都能放下；第一组后剩余空间低于 5 GiB 会停止下一组并保留已有结果。

每组计时服务和采集服务启动前、退出后都会把 `numactl -H`、全机与各 NUMA 节点的 free/文件缓存/匿名内存等、允许的内存节点以及可见 cgroup 限制写入 `summary.txt`。缺少 `numactl` 时仍使用 `/proc` 和 `/sys`。服务退出后再检查本组进程组是否还有非僵尸进程；没有退出则停止，不用清缓存掩盖残留进程。服务失败时额外记录清理前现场，保留原错误，不自动回收缓存。

需要自动回收时增加 `--auto-drop-caches`。全机 MemFree 低于 `--min-host-free-gib`，或任一允许的 NUMA 节点 MemFree 低于 `--min-numa-free-gib`，在服务之外执行一次 `sync` 后写 `drop_caches=1`，打印回收前后差值。两个触发阈值默认均为 160 GiB，按当前约 137 GiB 专家缓冲加 8.06 GiB scale 缓冲留出余量；这是可调的回收触发值，不是已证实的驱动分配要求。只清文件页缓存，不修改大页、NUMA 绑定或 memlock，不反复清理。权限不足会明确停止。回收后若全机 MemAvailable 仍低于配置阈值，下一组不启动；仅某节点 free 仍低则记录提示，不能在实际落点未知时据此断言该节点导致 OOM。

该动作影响宿主机文件缓存，不能释放仍被进程或驱动占用的 pinned 内存，也不保证解决分配失败。第一组启动前同样检查，所有回收均在正式计时、预热和 profile 窗口之外；各组保留原预热次数。是否执行过回收及前后状态进入报告，冷启动环境可能变化。参考 [Linux drop_caches 说明](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#drop-caches)。

复核同事的 `vllm-ascend` 提交 `c441874` 中 `prepared_h2d.py`：布局相同复用，否则一次性整理，再导出固定地址表；`native_submission.py` 还使用独立 DMA 通道、线程配置和依赖调度。我们参考前者，保留现有普通加载流与严格完成协议，没有把后者混入本次对照。对方 CPU MoE 的收益不能直接作为本项目的收益。

2026-11-24 复核 `native_update_05`：两组各 20 次正式请求、输出一致，但未采 profile。Python/native 的 decode raw 均值为 151.937/162.576 ms，TPOT 为 37.984/40.644 ms/token；IQR 后 decode 为 148.445/153.739 ms。HTTP 均值为 3569.273/12416.579 ms，TTFT 为 3417.158/12253.835 ms。主要退化在首 token 之前，不能归为已取得优化收益，也不能从这些数值推断 Host 间隔和空泡。后续修正保留原 prefill scale 布局，并补齐原生采集；复测结果见下文。

## 验证状态

随后返回的完整时延日志仍标记为 `native_update_06`，使用新构建目录 `build_01`，构建及 NPU 小图检查全部通过。每组 5 次预热、20 次正式请求，输出均为 5 token，摘要与用量一致。无 profiler/MSTX 的结果如下；对照是 MTE Python 更新与 MTE 原生 C++ 更新，不是早期 launch-host-function 回调。

| 指标 | MTE Python 更新 | MTE native 更新 | 降低 |
| --- | ---: | ---: | ---: |
| Decode 均值，ms | 149.154 | 123.746 | 17.03% |
| TPOT 均值，ms/token | 37.289 | 30.936 | 17.03% |
| TTFT 均值，ms | 3454.910 | 3421.619 | 0.96% |
| HTTP E2E 均值，ms | 3604.243 | 3545.555 | 1.63% |
| Decode 3×IQR 后均值，ms | 145.185（18/20） | 123.746（20/20） | 14.77% |

Python/native 的 decode 标准差分别为 14.574/1.828 ms，原始范围分别为 137.530～194.444 / 121.275～127.241 ms。两组原始样本范围未重叠，过滤 Python 两个长尾后仍有收益；但仍是顺序运行的单对服务，不能代替交错复测或把全部收益单独归于 GIL。上一轮 TTFT 大幅退化本次未复现，与恢复 prefill 源布局的修正方向一致，尚不构成单因素根因证明。

原生实际执行标记出现，43 层额外 scale 缓冲为 8657043456 字节（8.0625 GiB），复用为 0。本轮 profile 在启动第一组采集服务前因结果盘仅剩 0.70 GiB、低于 20 GiB 门槛而停止，所以没有本轮 Host 间隔或 memcpy 空泡数值，不能声称空泡已消除。无采集时延与输出一致性数据保留有效。

此前的 `native_update_06` 尝试在构建 MTE 核时停止，`aic_obj/aiv_obj` 未重新编译，后续 `ld.lld` 对已有 `handshake.cpp.o` 报 `unknown file type`。这与早期阶段记录的复用目录失败同类，当次尚未执行模型或性能测试；仅凭日志不能确定对象文件为何不可识别。脚本已改为每次保留新的构建子目录，避免复用中间对象，新增目录隔离与旧文件保留回归；上方复测已通过构建。

2026-11-24：本地重编译原生 C++ 扩展，相关八组 CPU/离线回归共 120 项，117 项通过、3 项平台相关跳过。新增覆盖原 prefill 缓冲布局保持、独立采集配置不污染主计时命令、原生计数器、采集失败清理、Host 标记覆盖核验、精确选流及 Transpose 占用空泡统计。定向 Ruff 通过；完整 `format.sh ci` 因本地缺少 pre-commit 未完成。上方复测通过 NPU 小图检查并取得时延结果，真实模型的 Host 间隔和加载流空泡仍待续采。

本地 Windows CPU 环境可以编译、加载两个 C++ 扩展，并验证原生线程在 Python 持有 GIL 时独立完成、错误不发布成功、重复请求读取新输入、预取上限和 LRC 状态一致性。随机序列测试比较 200 次更新的驻留集合与实际槽位字节。已有工作线程、生命周期、C++ LRC 和延后等待回归保留。

2026-11-20：重编译 C++ LRC 扩展，8 组本地测试共 150 项通过、2 项平台相关跳过。新增验证覆盖真实 MXFP4 布局函数、单缓冲加载、旧 eager/prefill 数值一致性、dtype 转换回退、带 storage offset 的视图、重载保护及地址溢出。改动文件的定向 Ruff 检查通过；manager 原有 12 条 lint 问题未扩大。完整 `format.sh ci` 因本地缺少 pre-commit 未执行成功。

绿区 Ascend950PR_957c 已通过：握手核构建、16 项原生 CPU 测试、30 项原工作线程测试、6 种 uint8/float32 连续与转置/MXFP4 布局的真实 NPU scale 拷贝检查。严格握手图重放 8 次，权重/scale/映射一致；其中 7 次在 Python 持有 GIL 的观测范围内由未完成推进到完成。

该轮随后在解析 `test_graph2.sh` 末尾续行符时失败，尚未启动完整模型。解析器已按原测试脚本的处理补齐输入末尾换行，并通过 8 项脚本回归。本地没有 Ascend NPU；完整模型输出、性能和空泡收益仍待绿区模型对照，不能从小图通过推断。
