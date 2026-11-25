# MTE 旧版权重搬运阻塞定位

实验记录已统一迁入 [MoE 卸载实验总记录](moe_experiment_results_zh.md#block-history-20261028)。本页只保留旧链接入口，后续结果请更新总记录。

<a id="mte-旧版权重搬运阻塞定位"></a>

[MTE 旧版权重搬运阻塞定位](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-1)

<a id="已确认事实与目标"></a>

[已确认事实与目标](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-3)

<a id="正常交接如何完成"></a>

[正常交接如何完成](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-18)

<a id="哪些改动可能形成循环等待"></a>

[哪些改动可能形成循环等待](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-33)

<a id="方向一实际提交到了错误的流或者夹带了跨流依赖"></a>

[方向一：实际提交到了错误的流，或者夹带了跨流依赖](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-47)

<a id="方向二量化参数-tensor-拷贝包含额外设备操作"></a>

[方向二：量化参数 Tensor 拷贝包含额外设备操作](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-57)

<a id="方向三host-提交队列等待而非已经提交的设备任务等待"></a>

[方向三：Host 提交队列等待，而非已经提交的设备任务等待](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-70)

<a id="暂不优先的解释"></a>

[暂不优先的解释](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-80)

<a id="本轮诊断在同一次模型运行中拆分等待阶段"></a>

[本轮诊断：在同一次模型运行中拆分等待阶段](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-86)

<a id="执行命令与下一步边界"></a>

[执行命令与下一步边界](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-116)

<a id="原始-acl-等待结果与-scale-分段诊断"></a>

[原始 ACL 等待结果与 scale 分段诊断](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-132)

<a id="分段结果h2d-完成设备侧布局写入未完成"></a>

[分段结果：H2D 完成，设备侧布局写入未完成](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-162)

<a id="独立-d2d-与等待并行实验结果及图内依赖对照"></a>

[独立 D2D 与等待并行实验结果及图内依赖对照](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-183)

<a id="分叉汇合实验通过返回真实模型取证"></a>

[分叉汇合实验通过，返回真实模型取证](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-205)

<a id="真实上下文中连续目标也失败转向-runtime-任务证据"></a>

[真实上下文中连续目标也失败：转向 Runtime 任务证据](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-242)

<a id="runtime-超时快照与流身份核对"></a>

[Runtime 超时快照与流身份核对](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-272)

<a id="流映射结果与预取等待方式对照"></a>

[流映射结果与预取等待方式对照](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-296)

<a id="预取报告线程等待对照完成推理"></a>

[预取报告线程等待对照完成推理](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-332)

<a id="撤掉逐项诊断同步后的连续请求结果"></a>

[撤掉逐项诊断同步后的连续请求结果](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-357)

<a id="aiv-退出前后完成状态实验"></a>

[AIV 退出前后完成状态实验](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-382)

<a id="百万次预算实验结果与设备退出原因取证"></a>

[百万次预算实验结果与设备退出原因取证](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-412)

<a id="十万次预算结果与-runtime-任务级定位"></a>

[十万次预算结果与 Runtime 任务级定位](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-435)

<a id="runtime-debug-已确认原始-scale-拷贝对应超时-aivec-任务"></a>

[Runtime DEBUG 已确认原始 scale 拷贝对应超时 AIVEC 任务](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-461)

<a id="已确认转置任务身份用加载流分核检验资源需求假设"></a>

[已确认转置任务身份；用加载流分核检验资源需求假设](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-483)

<a id="加载流-28-核诊断结果仍在相同预取位置超时"></a>

[加载流 28 核诊断结果：仍在相同预取位置超时](moe_experiment_results_zh.md#block-history-20261028-moe_mte_block_diagnosis_zh-518)
