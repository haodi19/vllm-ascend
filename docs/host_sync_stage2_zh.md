# 本地 CPU/NPU 交接扩展：第二阶段

第一阶段已在绿区验证通过：17 项 CPU 单测、ACL 注册以及 40 次 Host 模拟请求。
设备为 Ascend950PR_9579，PyTorch 为 2.10.0+cpu，torch_npu 为
2.10.0.post4.dev20260715。

第二阶段增加独立的 AIV 发布、等待算子，验证真正由 NPU 发起的交接。
本阶段已在上述设备完成绿区编译和上板正确性验证，不预先宣称性能收益。

## 已验证结果

18 项 Host 单测、5 项结果与构建参数校验全部通过，AIV 动态库编译成功。
普通执行连续 100 轮，CPU 回调次数为 100，最终 NPU 结果为 `[100, 1, 100, 7]`；
轮询次数范围为 1～134。回调异常返回 `[1, 2, 0, 339]`；受控超时返回
`[1, 3, 0, 8]`，均符合预期，设备同步后完成内存释放。

这证明所测环境下的标记交接可用。轮询次数不是微秒；子进程耗时包含运行时初始化等开销，
不能作为同步时延。下一步见[第三阶段：图捕获与重放](host_sync_stage3_zh.md)。

## 与现有推理版本的关系

新增算子通过独立 CMake 工程构建为 `libhost_sync_device.so`，只由测试子进程加载。
不安装系统 OPP 算子包，不注册 Torch 全局算子，不修改 MoE manager、LRC、
主工程构建入口或 `script/test_graph2.sh`。原推理路径继续使用原来的实现。

Host 扩展升级到 API 2，原有接口保留；新增设备访问保护接口，防止设备仍可能使用
控制内存时调用 Host 模拟提交、停止或释放。参考仓库不作修改。

## 实际执行流程

```text
NPU 发布算子：读取并递增 NPU 上的 generation
    ↓ MTE3 / DataCopyPad 写入注册的 Host 请求区
CPU 常驻线程：发现新 generation，执行测试回调
    ↓ 成功写 done_generation；异常写 error_generation
NPU 等待算子：MTE2 / DataCopyPad 反复读取完成区
    ↓ 完成、错误或达到轮询上限后退出
测试程序：同步设备流，检查 NPU 输出及 CPU 回调次数
    ↓ 确认没有设备访问后释放控制内存
```

发布和等待算子都只启动一个 block。请求与完成区域使用映射设备地址，
两者禁用 L2 缓存提示，算子内使用 MTE/Scalar 事件保证搬运与核内标量操作的顺序。
这些处理不能代替跨 CPU/NPU 内存可见性的真实验证；测试检查实际读取结果。

每轮 generation 在 NPU 算子中递增，Host 不逐轮写入该数值。
当前测试是多轮普通执行，尚未捕获成图。图重放及与 Torch 异步任务队列混合提交，
留到闭环验证通过后再单独实现。

NPU 输出四个整数：`[generation, code, observed_generation, polls]`。

| code | 含义 |
| --- | --- |
| 1 | 观察到本轮成功完成标记 |
| 2 | 观察到本轮回调错误标记 |
| 3 | 达到最大轮询次数 |
| 4 | 非法 generation，例如溢出到 0 |

达到轮询上限只会结束独立测试算子，不表示任务成功，也不能据此放行真实专家计算。
本阶段没有后续模型计算，不存在失败后继续使用未就绪专家权重的路径。

## 绿区执行

更新代码后，在容器的项目根目录执行：

```bash
python -B script/run_host_sync_stage2.py --npu 1
```

物理 NPU 编号可替换。脚本依次重新构建独立 Host 扩展、执行 CPU 回归检查、
构建 AIV 动态库，再启动三个隔离子进程进行测试：

1. 正常回调：默认连续 100 轮，检查 generation、成功标记和回调次数一致。
2. 回调异常：刻意让回调抛错，确认设备看到错误而不是成功。
3. 轮询上限：暂时阻止 CPU 回调完成，确认设备可以退出等待并报告 code 3。

设备测试还检查访问保护：持有设备访问权限时不能关闭 Session、再次获取权限或
由 Host 提交测试任务。测试结束后必须先成功同步设备，再释放访问权限和内存。

默认 CANN 路径是 `/usr/local/Ascend/ascend-toolkit/latest`。如果实际路径不同：

```bash
python -B script/run_host_sync_stage2.py --npu 1 --cann-root /usr/local/Ascend/cann-9.1.0
```

使用现有 CANN 中的编译器与 `ascendc.cmake`。`SOC_VERSION` 默认使用已确认的具体
型号 `ascend950pr_9579`，`ASCEND_COMPUTE_UNIT` 使用架构名 `ascend950`，两者不能混用。
其他 Ascend950 型号可用 `--soc-version` 指定具体名称，并需在当前 CANN 支持列表中。
首次绿区第二阶段测试通过 18 项 Host 单测与 3 项结果校验，但因把架构名误作
`SOC_VERSION` 而停在 CMake 配置；修正后已通过上述真实 NPU 交接测试。
本次修复直接重新执行上述命令，不加 `--skip-build`，脚本会覆盖旧缓存中的型号参数。
算子编译产物保存在 `build/host_sync_device_stage2`，不修改 CANN 安装目录。
未修改代码且已经成功构建时，可以加 `--skip-build`。

`--max-polls` 默认 100000，控制正常/异常测试中设备轮询次数上限，它不是微秒单位。
`--test-timeout` 默认 120 秒，控制每个设备测试子进程的最大运行时间。
设备同步失败时测试子进程直接以失败退出，避免执行可能提前释放设备使用中内存的
Python 清理函数；外层还有进程组超时保护。不要把这个测试退出策略直接用于服务进程。

## 返回内容与判断边界

请返回控制台的 `RESULT` 行和 `STAGE 2 SUMMARY`；失败时连同失败步骤及错误末尾返回。
脚本不包含 Git 操作，不写实验日志和结果文件，构建产物除外。

全部通过只证明所测环境和次数下的交接正确性，包括预期错误与退出路径。
`polls` 是轮询次数，不能直接换算成同步时延。本阶段不测推理、不比较 `_launch_host_func`
性能、不保存 profile；图重放、性能与可视化对照放在下一阶段。

## 接口依据

- [Ascend C Kernel Launch 工程说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0056.html)
- [GlobalTensor 缓存提示接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/data_structures/GlobalTensor/SetL2CacheHint.md)
