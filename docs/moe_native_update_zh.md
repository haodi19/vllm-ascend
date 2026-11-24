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

加载流现在默认自动识别，Python/native 对照、原四/五组对照和五组夜间脚本均不再固定 54/56。采集服务初始化时，对 `manager.load_stream` 调用 [CANN 的 aclrtStreamGetId](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1038.html) 取得实际流 ID；每个采集窗口开始时，由控制线程将该 ID 写入一个 `moe_offload/loading_stream` Host 标记。分析器读取同次采集、同一 Host 进程的标记，再匹配设备轨道的 `thread_name`，不按 tid 或最多拷贝数选流。查询、标记均在计时请求和权重更新热路径之外；不创建新设备流、不向加载流提交额外任务、不改变 GIL-free 更新路径。`summary.txt` 的 `LOADING_STREAM_RUNTIME` 和 `LOADING_STREAM_SELECTION` 会记录查询结果、所选轨道及依据。

新测试删除旧命令中的 `--bubble-stream-name "Stream 54"` 或 `"Stream 56"` 即可，默认值为 `auto`，无需增加其他参数。手动流名与运行时记录冲突时会明确报错；缺少记录或存在多条同名候选时保留采集，不猜测。此前生成的旧 profile 没有这个标记，离线重统计仍需指定实际流名；自动识别不能凭空补出旧采集未记录的身份。

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

也可分两次执行续采：上述命令加 `--profile-case python`，输出目录用 `native_profile_python_01`；结束后再加 `--profile-case native`，输出目录用 `native_profile_native_01`。两条命令均保留同一个 `--profile-only-from` 来源，每次只启动所选组，默认仍预热 5 次、采集 1 次。单组 `summary.txt` 只列该组的历史时延和本次采集结果；两组分别返回各自摘要即可。成功的 Python 单组仍在服务退出后执行固定清缓存，native 单组不会再启动 Python。省略 `--profile-case` 时，原来的两组顺序执行流程不变。

采集服务失败时，摘要自动加入服务日志末尾及错误上下文（最多读取末尾 64 KiB），并保留内存现场；无需另外查找子进程日志。完整服务日志仍保留，摘要摘录不能保证覆盖更早的错误。

若已打印 `PROFILE SAVED`、`ALIGNED_PROFILE`，随后因 callback/update 嵌套检查失败，不要重新启动模型。此时采集和对齐文件已经保存，可执行 `python -B script/inspect_native_profile_nesting.py <本次结果目录>`，只读输出 callback/update 数量、重复或缺失编号、线程不一致，以及最多 6 个异常区间的起止时间。检查同时比较现有浮点算法与直接解析 JSON 数值的十进制算法，以区分大时间戳运算误差和导出文件中实际存在的越界；不修改时间线、不增加容差、不自动接受失败采集。后续在线测试遇到此类失败也自动把这些诊断写入摘要。

2026-11-24，`native_profile_native_02` 已确认是统计脚本的浮点误报：170 对 Host 区间全部配对、同线程，无重复或缺失编号；第 138 次调用内层结束为 `1795507048074899.799 us`，外层结束为 `1795507048074899.899 us`，实际提前 0.100 us，浮点相加却算成内层越界 0.25 us。嵌套检查及 Host 间隔相减现改用直接从 JSON 读取的十进制数，完成相减后才转为统计数值；真实越界仍拒绝，没有新增容差，也没有改动原可视化时间戳。已有采集可运行以下离线重统计，原测试入口也自动使用修复后的计算：

```bash
python -B script/reanalyze_native_moe_profiles.py \
  /data/w00919836/moeoffload/bench_results/native_profile_python_01 \
  /data/w00919836/moeoffload/bench_results/native_profile_native_02 \
  --bubble-stream-name "Stream 56" \
  --output-dir /data/w00919836/moeoffload/bench_results/native_profile_analysis_01
```

该命令先验证已有对齐元数据、Host 计数及保存的输出一致性，再复用原统计入口生成双向 Host 间隔和指定加载流的空泡表；结果仅写入新目录的 `summary.txt` 及详细报告，不启动模型、不重新对齐、不覆盖原摘要。Host 间隔可能有亚微秒数值修正，加载流空泡的算法及筛选阈值保持不变。上述两个输入均来自物理卡 6；输出会列出读取到的用卡记录，不把离线重统计当成新一轮时延测试。

加载流显示名不是跨采集固定编号。2026-11-24 的另一次 native 采集已完成请求和时钟对齐，也取得两个方向的 Host 间隔，但因实际轨道中没有默认的 `Stream 56`，空泡选流失败；无需重跑模型。在线统计现在会将 `COPY_TRACK` 列表同时写入控制台和摘要，即使选流失败也保留。对已经保存的采集，可只读列出轨道：

```bash
python -B script/reanalyze_native_moe_profiles.py \
  /data/w00919836/moeoffload/results/native_profile_native_02 \
  --list-streams
```

列表给出可视化显示名、设备 pid/tid、MEMCPY 数量，以及 trace 中可用的方向和字节数；字段缺失时明确报告未知。它不按 tid 或最大拷贝数自动认定加载流。确认加载流后，在上面的离线重统计命令中使用 `--bubble-stream-name "实际显示名"`；在线测试入口也支持同名参数。如果两份采集的显示名不同，应分别指定各自的流做离线重统计。阈值仍为 `10<=gap<5000 us`，Transpose 等同流任务仍计为占用。

同日下午再次采集时，`Stream 54` 只有 2 次 MEMCPY、0 个匹配加载批次，45696 次 MEMCPY 出现在 `Stream 56`。这说明上一次采集的流名不能直接沿用到重新启动的服务。本次请求、采集和时钟对齐成功，Host 间隔 raw 均值为预取→按需 257.544 us、按需→下一层预取 34.281 us；3×IQR 后分别为 257.544/34.791 us。所选 Stream 54 的 4574.25 us 间隔不能作为专家加载空泡结果。汇总表此前对空的方向统计 `kept=null` 取下标，导致 `TypeError`；现已修复，空样本显示数量 0、均值“—”，不当作零时延。选中轨道没有匹配加载批次时，明确标记 `BUBBLES_UNVERIFIED`，保留轨道诊断但不纳入加载空泡对比表。已有 profile 可离线重新选流，无需重采。

只有约 7 GiB 可用时，可增加 `--min-profile-free-gib 5`，将本次门槛降到 5 GiB；默认值和其他实验入口不变。续采入口及每组采集服务启动前均使用此门槛，实际设置写入 `summary.txt`。这只是启动前剩余空间门槛，不限制 profile 大小，也不保证两组采集和对齐导出都能放下；第一组后剩余空间低于 5 GiB 会停止下一组并保留已有结果。

每组计时服务和采集服务启动前、退出后都会把 `numactl -H`、全机与各 NUMA 节点的 free/文件缓存/匿名内存等、允许的内存节点以及可见 cgroup 限制写入 `summary.txt`。缺少 `numactl` 时仍使用 `/proc` 和 `/sys`。服务退出后再检查本组进程组是否还有非僵尸进程；没有退出则停止，不用清缓存掩盖残留进程。服务失败时额外记录清理前现场，保留原错误，不自动回收缓存。

按固定顺序执行，无需额外参数：Python 组成功完成并退出后，确认本组进程组已退出，执行一次 `echo 1 > /proc/sys/vm/drop_caches`，成功后才允许进入 native 组。完整测试的计时对照和独立 profile 对照都执行此顺序。内存读数不再决定是否回收，也不按 160 GiB 阈值阻止下一组；native 结束后不额外回收。保留回收前后的 NUMA/Host 读数及差值，命令成功明确打印 `HOST_CACHE_RECLAIM_PASS`；读取内存信息失败不会跳过固定回收，命令本身失败则停止下一组。

该动作影响宿主机文件缓存，不能释放仍被进程或驱动占用的 pinned 内存，也不保证解决分配失败。第一组启动前只记录内存；固定回收在 Python 组结束之后、下一组启动之前，各组保留原预热次数。回收及前后状态进入报告，后组的冷启动缓存条件可能变化。参考 [Linux drop_caches 说明](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#drop-caches)。

复核同事的 `vllm-ascend` 提交 `c441874` 中 `prepared_h2d.py`：布局相同复用，否则一次性整理，再导出固定地址表；`native_submission.py` 还使用独立 DMA 通道、线程配置和依赖调度。我们参考前者，保留现有普通加载流与严格完成协议，没有把后者混入本次对照。对方 CPU MoE 的收益不能直接作为本项目的收益。

2026-11-24 复核 `native_update_05`：两组各 20 次正式请求、输出一致，但未采 profile。Python/native 的 decode raw 均值为 151.937/162.576 ms，TPOT 为 37.984/40.644 ms/token；IQR 后 decode 为 148.445/153.739 ms。HTTP 均值为 3569.273/12416.579 ms，TTFT 为 3417.158/12253.835 ms。主要退化在首 token 之前，不能归为已取得优化收益，也不能从这些数值推断 Host 间隔和空泡。后续修正保留原 prefill scale 布局，并补齐原生采集；复测结果见下文。

## 验证状态

2026-11-24 的两组续采均在物理卡 6 上完成，运行时 ACL 流 ID 和导出的加载轨道均为 Stream 56。Python/native 的预取→按需 Host 原始间隔为 104.876/257.082 μs，按需→下一层预取为 64.623/34.316 μs；全窗口加载流空泡为 444.669/273.944 μs，回调连续窗口为 389.865/201.577 μs。两方向加载批次空泡分别为 399.486/263.371 μs、330.873/71.108 μs。每组仅一个采集请求；这些是新采的 profile 数据，无采集时延仍引用 `native_update_06`。

预取→按需 Host 间隔变长，可能因为预取任务提前结束而暴露出剩余计算等待，不能单凭间隔均值定因。可复用这两份已对齐 profile，执行：

```bash
python -B script/reanalyze_native_prefetch_overlap.py \
  /data/w00919836/moeoffload/results/native_profile_python_01 \
  /data/w00919836/moeoffload/results/native_profile_native_01 \
  --output-dir /data/w00919836/moeoffload/results/native_overlap_01
```

此命令不运行模型、不导入 Torch、不重采或重对齐。只向新目录写入 `summary.txt`、逐样本 `prefetch_overlap.json` 和 `details.log`，核心表格同时打印到终端。它检查同次采集的输出、计数器与校准，再沿用原 MTE 消费边界和两种方案：方案 1 删除明确计算晚于预取结束的样本；方案 2 扣除这些样本从预取结束到计算到达消费位置的经过时间，采用边界区间中点。全样本口径下，不确定和无法识别样本保留原间隔；不做 IQR 或长尾截断。直接判定与原保守余量判定同时输出，所有相对原均值的降低都是描述性重统计，不是新测推理加速收益。

摘要中的 Host 函数时延表覆盖全部任务，按全部、预取、按需三种范围对比 `host_callback`、`update_weights`、`load_submit`、`copy_submit`、`synchronize`，保留原始与 3×IQR 后的均值、有效/应有样本数及两组原均值变化。这里包括首尾和零加载任务，不局限于可配对的间隔样本；缺失分项不按零时延处理，嵌套分项不能相加。

原因分析在同层同次序、边界算子一致的样本上拆分计算相对到达、Host 提前结束及后续按需开始的变化；另以上一按需 Host 结束作流程锚点，避免把预取启动时刻变化直接当作设备计算变慢。native 旧采集未记录逐专家加载计数，缺少 Python 的准备标记不当作零加载，跨组配对也不声称专家集合相同。时间戳先以十进制减去同次采集纪元再转浮点，避免紧邻原生标记在大时间戳上发生虚假的嵌套越界。当前只完成本地合成采集回归，实际原因及重统计数值需读取服务器上上述 profile 后确认。

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
