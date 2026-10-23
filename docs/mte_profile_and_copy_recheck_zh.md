# MTE 可视化与量化拷贝复核

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
