# 基于阻塞旧版的 worker 轮询改造

## 代码基线与分支

从 `942868dd8cec9019dee04bee038bfcd678b6d66a` 新建
`dev/moeoffload-worker-poll`，不合入原开发分支中的布局适配、CPU 重排、活跃槽位前缀扫描。
原 `dev/moeoffload` 和旧版阻塞复现目录保留。

第一阶段增加可选的 `Session.configure_polling(idle_mode, timing)`，必须在 start 前调用。
默认 `yield`、计时关闭；MoE dispatcher 未调用此接口，因此原阻塞路径保持旧行为。
`spin` 仅取消空闲整轮扫描后的主动 yield，不删除 mutex，不更改 GIL、共享内存协议或设备算子。
后续去锁、直接定位槽位、绑核分别作为独立改动，不在本轮混入。

## 计时范围

worker 内使用 steady_clock，记录有界样本，停止后读取；不逐轮打印，也不在运行中读样本。
每次任务记录领取锁等待、发现任务到获取 GIL 前、GIL 获取、回调、GIL 释放、完成锁等待及完成发布。
空闲扫描每 1024 轮抽样一次，记录整轮扫描、其中的锁获取时间，以及 yield 调用前后时间。
样本容量各 8192；任务样本溢出禁止性能比较；空闲样本截断会显式报告。

计时开启时每槽位读取时钟会扰动扫描，必须与关闭计时的两组对照。
空闲样本包含初始化、图捕获、预热和逐次正确性校验间隙，不能作为 NPU 发布到 CPU 发现的时延。
spin 模式的 idle_action 主要是空计时开销。原始样本使用纳秒，输出统计统一转换为微秒，
指标名以 `_us` 结尾并标记 `unit=us`。

## 一次执行的实验

```bash
python -B script/run_worker_poll_tests.py --npu 7
```

脚本构建 Host 扩展和原设备算子，先执行 CPU 生命周期及新策略测试，再比较：

- 槽位数 1 与 2048，活动任务均放在最后一个槽位，每组内部只有 yield/spin 不同。
- 计时关闭与开启，每组交替顺序运行两轮，每轮预热 20 次、测量 200 次。
- 每次图重放均核对回调次数和 generation，失败立即停止，不重试或忽略失败样本。

单次完整交接包含 NPU 发布、CPU 空 Python 回调和 NPU 有界等待，不搬专家权重，
不加载模型、不采集 profiler，因此不触发已知 scale 搬运阻塞。
NPU Event 区间是包含图执行的完整机制区间，不是纯共享内存通知成本。
Host replay 到同步返回的时间还包含提交与等待；两者不与 worker 子项直接相加。
分项用于定位，收益以计时关闭组为主；不据此宣称真实 MoE 获得同等收益。
当前尚未加入 worker CPU 占用、上下文切换统计或严格的 NPU 发布到 CPU 发现计时。

输出保存在新建的 `bench_results/worker_poll_<编号>/summary.txt`，详细子进程日志留在同目录。
只返回 summary 内容即可。脚本不含 Git 操作，也不修改生产默认配置。

## 绿区首次切换

在原可运行仓库执行以下 Git 命令，建立第三个独立目录。不要在阻塞复现目录 reset。

```bash
cd /home/l00992607/vllm-ascend
git fetch origin
git worktree add -b dev/moeoffload-worker-poll \
  /home/l00992607/vllm-ascend-worker-poll \
  origin/dev/moeoffload-worker-poll
cd /home/l00992607/vllm-ascend-worker-poll
python -B script/run_worker_poll_tests.py --npu 7
```

独立握手不导入 vllm_ascend 模型，无需复制公共算子包、热点 JSON 或重新安装整个插件。
本脚本会安装本分支 Host 扩展，三个目录仍共用容器 Python 环境，不能同时测试不同版本。
切回任一目录运行之前，先在那个目录执行：

```bash
python -m pip install ./csrc/host_sync --no-build-isolation --no-deps --force-reinstall
```

NPU 编号按空闲设备修改。后续更新只在新目录 git pull --ff-only，再运行同一脚本。
