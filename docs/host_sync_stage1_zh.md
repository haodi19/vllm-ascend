# 本地 CPU/NPU 交接扩展：第一阶段

本阶段为后续以 generation 标记替代 `_launch_host_func` 的交接方式准备
CPU 侧基础设施：独立 C++ 扩展、常驻工作线程、控制内存注册以及生命周期管理。
**当前没有替换 MoE 的任何执行路径，也没有实现 AIV 发布、等待算子。**

## 本阶段实现的内容

扩展独立安装，导入名称是 `vllm_ascend_host_sync`。它不要求导入 vLLM 或
PyTorch，构建也不与现有 LRC C++ 扩展绑定。

一个 `Session` 包含若干任务槽和一个常驻工作线程。每个任务槽在启动前登记
一个无参数 Python 回调。工作线程发现新 generation 后执行对应回调，成功后
记录完成 generation；回调失败时记录错误，不能按成功完成处理。

工作线程空闲轮询时不持有 Python GIL，调用 Python 回调时获取 GIL。
多个任务槽由同一个工作线程依次处理，回调不会在这个 Session 内并发执行。
本阶段不预设任务槽分别对应哪一层、按需还是预取，MoE 接入留待后续阶段。

有两种控制内存后端：

- `cpu`：普通 CPU 测试内存，用于检验工作线程和控制逻辑，不需要 NPU。
- `acl`：申请 CPU 内存并通过 ACL 注册映射，获得 CPU 地址和供设备使用的地址。
  注册或设备初始化失败时直接报错，不自动降级为 `cpu`。

获得设备地址只表明注册成功，**不代表已验证 NPU 算子可以正确访问、跨设备缓存
可见性成立，或交接时延已经下降**。这些需要下一阶段的 MTE 算子和真实设备测试。

## 控制协议与接口

每个任务槽维护以下主要状态：

| 字段 | 含义 |
| --- | --- |
| `request_generation` | 已发布的任务轮次 |
| `done_generation` | 成功完成的任务轮次 |
| `error_generation` | 出错的任务轮次 |
| `status` | 0：空闲，1：执行中，2：成功，3：错误，4：取消 |
| `error` | 用于排查问题的 CPU 侧错误描述 |

generation 从 1 开始，同一槽每次必须增加 1。上一任务尚未结束时不得覆盖其请求。
当前 `submit_for_test()` 由 Host 模拟发布请求，只用于本阶段验证，不能与未来
设备发布算子同时写同一个任务槽。

主要接口如下：

```python
import vllm_ascend_host_sync as hs

with hs.Session(slots=1, backend="cpu") as session:
    session.register_task(0, lambda: print("任务已执行"))
    session.start()
    session.submit_for_test(0, 1)
    result = session.wait(0, 1, timeout_ms=5000)
    print(result)
```

`with` 负责退出时关闭 Session，不会自动启动工作线程。任务只能在 `start()` 前登记，
一个 Session 只能启动一次。

扩展使用 Python 包装层保留 Session 的退出清理引用，防止用户丢失引用时工作线程
仍在访问已经释放的对象。建议始终使用 `with` 或显式 `close()`，并在 PyTorch/NPU
环境销毁之前关闭。进程正常退出时，兜底清理会等待正在执行的回调结束；它无法安全
强杀一个永不返回的 Python 回调，因此测试脚本另有子进程超时保护。

- `info()`：查看后端、任务槽数量、内存布局、地址、工作线程和生命周期状态。
- `snapshot(slot)`：查看一个任务槽的当前状态。
- `wait()`：等待指定轮次；超时抛出 `TimeoutError`，回调错误或任务取消抛出
  `RuntimeError`。
- `stop()`：停止接受新任务，取消尚未执行的任务，等待正在执行的任务结束；
  不立即释放控制内存。
- `close()`：停止后释放资源，重复关闭不产生额外操作。

`stop()` 或 `close()` 超时后，不能提前释放仍被工作线程使用的内存，调用者可以
等回调结束后重试。回调不能在自己的工作线程中调用该 Session 的
`wait()`、`stop()` 或 `close()`，以免等待自己完成。

后续接入设备图时，还必须先确认图和设备操作已不再使用控制内存，才能关闭
Session。当前 CPU 生命周期测试不等价于验证设备仍在访问时可以安全释放资源。

当前控制协议为实验性版本 1，请求区与完成区分别按 512 字节对齐，每槽占用
1024 字节。CPU 测试使用 64 位 acquire/release 原子读写，但这本身不能证明
NPU MTE 与 CPU 间的缓存可见性。`info()` 暴露地址和偏移供后续开发核对，
当前不要把这些地址交给设备算子或图使用。

## 绿区执行方式

更新代码后，在项目根目录、原有 Python 环境中执行：

```bash
python -B script/run_host_sync_stage1.py --npu 4
```

将 `4` 改成实际要使用的物理 NPU 编号。脚本只在 ACL 测试子进程中设置
`ASCEND_RT_VISIBLE_DEVICES`，该子进程通过逻辑编号 `0` 使用所选设备。
测试不会启动模型，也不会占用整卡模型权重内存。

脚本依次完成：

1. 检查 `setuptools`、`wheel`、`pybind11` 是否可用。
2. 用当前 Python 安装 `csrc/host_sync` 独立扩展，使用
   `--no-build-isolation --no-deps`，不自动安装其他依赖。
3. 执行 CPU 单测。
4. 在独立子进程初始化 `torch_npu`，测试 ACL 注册、设备地址返回、
   Host 模拟请求、停止和释放。
5. 在控制台打印摘要。

缺少构建依赖时，按公司允许的包源补齐后重试。脚本不会自行切换包源。
构建和测试均有限时保护；失败时打印该步骤输出末尾，便于复制排查。
脚本不包含 Git 操作，不写测试日志或结果文件；安装扩展本身会产生正常构建产物。

仅运行无需 NPU 的检查：

```bash
python -B script/run_host_sync_stage1.py --cpu-only
```

已构建且本轮没有修改扩展代码时，可跳过构建：

```bash
python -B script/run_host_sync_stage1.py --npu 4 --skip-build
```

`--build-timeout`、`--test-timeout`、`--acl-timeout` 分别控制三个步骤的超时秒数，
默认是 600、120、120。不要因为超时直接认定发生死锁，应结合失败步骤输出判断。

## 测试结果如何理解

CPU 单测检查多槽、多轮 generation、单工作线程执行、Python 线程继续运行、
回调异常、等待超时、停止取消、关闭超时后重试、非法生命周期操作和进程正常退出。
扩展未安装时测试直接失败，不会以跳过测试冒充通过。

ACL 测试在注册的 CPU 内存上运行 40 个 Host 模拟请求。其通过表示：

- 当前环境可以加载所需 ACL 接口并注册控制内存。
- 接口返回非零设备地址。
- 工作线程、回调和资源释放可以在该环境完成。

**它不包含 NPU 发起的 MTE 访问、CPU/NPU 真实交接、图捕获重放、MoE 推理、
时延对比或 profile。** `--cpu-only` 会明确把 ACL 部分标记为 `NOT TESTED`。

执行后检查控制台末尾的 `STAGE 1 SUMMARY`。请先审查输出内容，再复制摘要；
如果失败，请同时复制失败步骤及其错误末尾。下一阶段再实现设备发布和等待算子，
开展真正的 CPU/NPU 交接与图重放实验。
