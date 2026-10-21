# 本地 CPU/NPU 交接扩展：第五阶段

目前已实现独立 C++ 工作线程、控制内存和 AIV/MTE 交接算子，并通过普通执行、图重放、
最小性能与 profile 检查。测试脚本用于验证这些实现；真实 MoE manager 尚未使用新后端。

第五阶段从“回调只做计数”推进到实际数据搬运。权重和路由数据是人工构造的，
没有加载模型，也没有调用 LRC 或 `_update_weights`。代码已完成，待绿区验证。

## 本次验证流程

1. 在图外改变 NPU 路由输入，每轮使用不同整数，确保不会误用上轮数据。
2. 图内在计算流上将 64 字节路由数据 D2H 到固定的 pinned Host 缓冲区。
3. callback 组通过原 `_launch_host_func` 进入报告线程；MTE 组通过发布算子进入新的
   C++ 工作线程。两组使用相同的 Python 数据处理函数。
4. CPU 从路由数据确定四份人工专家权重中的一份，在独立传输流上提交实际 H2D。
   默认每次传输 65536 个 int64，共 512 KiB。随后同步传输流，确认完成后更新 Host 映射。
5. CPU 函数返回。callback 组由原机制恢复计算流；MTE 组由完成标记放行等待算子。
6. 计算流将 64 字节映射 H2D，再执行一个新增 AIV 消费算子，读取权重首尾、映射和路由，
   生成可校验结果。MTE 等待未成功时，消费算子返回零结果，不读取权重。
7. 图执行结束后逐轮检查回调次序、消费结果、全部权重内容；任一项失败就停止。

捕获一次后默认重放 20 次，输入和所选专家每次变化。预热一次，因此每组正常回调数为 21。
完整权重检查发生在图执行完毕后，包含额外 D2H；本阶段是正确性测试，不报告性能收益。

测试中的数据搬运使用 `aclrtMemcpyAsync`，工作线程通过 `aclrtSynchronizeStream`
等待独立传输流，使用的是实际设备传输。图内不混用 Torch 的异步任务提交队列。
因此通过后仍需验证项目中的 `Tensor.copy_`、Torch 队列、真实 `_update_weights`、
多槽位和预取/按需任务组合，不能把本阶段直接当作完整 MoE 接入。

## 绿区执行

更新代码后执行：

```bash
python -B script/run_host_sync_stage5.py --npu 1
```

脚本自动使用新目录编译（新增消费算子需要重编），复跑基础交接/图检查，随后分别运行
callback 和 MTE 数据流。没有修改原推理入口和默认行为，不需要重装主项目。
输出直接显示在控制台；本轮不采 profile、不写实验结果文件。
返回 `DATAFLOW_RESULT`、`STAGE 5 SUMMARY`；如失败，返回对应步骤的报错。

可用 `--rounds` 改变图重放次数，用 `--elements` 改变每次权重传输的 int64 元素数，
后者必须是 8 的倍数，范围为 8～1048576。默认单个子进程限时 120 秒。

参考：[aclrtMemcpyAsync 接口与异步完成要求](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/runtimeapi/aclcppdevg_03_0106.html)。
