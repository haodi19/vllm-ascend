# MoE MTE 完整复核与等待位置实验（2026-10-29）

实验记录已统一迁入 [MoE 卸载实验总记录](moe_experiment_results_zh.md#wait-placement-20261030)。本页只保留旧链接入口，后续结果请更新总记录。

<a id="moe-mte-完整复核与等待位置实验2026-10-29"></a>

[MoE MTE 完整复核与等待位置实验（2026-10-29）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-1)

<a id="1-原项目与当前改造的完整顺序"></a>

[1. 原项目与当前改造的完整顺序](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-15)

<a id="2-dsa-迁移中需要补充的三个差异"></a>

[2. DSA 迁移中需要补充的三个差异](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-56)

<a id="3-历史证据重新分类"></a>

[3. 历史证据重新分类](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-76)

<a id="4-候选原因与可区分的新方向"></a>

[4. 候选原因与可区分的新方向](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-99)

<a id="5-本次实现只延后严格预取等待及其后的映射写回"></a>

[5. 本次实现：只延后严格预取等待及其后的映射写回](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-118)

<a id="6-绿区命令与回传判据"></a>

[6. 绿区命令与回传判据](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-147)

<a id="7-首次等待位置实验捕获前退出尚未形成机制对照"></a>

[7. 首次等待位置实验：捕获前退出，尚未形成机制对照](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-177)

<a id="8-首次离线取证修正误筛补齐同一运行的日志覆盖"></a>

[8. 首次离线取证：修正误筛，补齐同一运行的日志覆盖](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-233)

<a id="9-离线取证结束一次带启动观测的模型运行"></a>

[9. 离线取证结束：一次带启动观测的模型运行](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-287)

<a id="10-等待位置有效通过下一步去掉最初8批加载诊断"></a>

[10. 等待位置有效通过：下一步去掉最初8批加载诊断](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-352)

<a id="101-当前判断与优先级"></a>

[10.1 当前判断与优先级](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-383)

<a id="102-下一次只运行这一组"></a>

[10.2 下一次只运行这一组](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-401)

<a id="11-普通加载通过转入有限输出对照与更长执行覆盖"></a>

[11. 普通加载通过：转入有限输出对照与更长执行覆盖](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-424)

<a id="111-为什么下一步做输出对照"></a>

[11.1 为什么下一步做输出对照](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-447)

<a id="112-绿区命令与判读"></a>

[11.2 绿区命令与判读](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-468)

<a id="12-三种输入输出一致验证是否依赖图debug2026-10-30"></a>

[12. 三种输入输出一致：验证是否依赖图debug（2026-10-30）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-493)

<a id="121-两项尚未解释的观测"></a>

[12.1 两项尚未解释的观测](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-505)

<a id="122-下一步的代码依据与边界"></a>

[12.2 下一步的代码依据与边界](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-521)

<a id="123-绿区命令"></a>

[12.3 绿区命令](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-539)

<a id="13-回到阻塞主线固定晚映射只改变严格wait位置2026-10-30"></a>

[13. 回到阻塞主线：固定晚映射，只改变严格wait位置（2026-10-30）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-560)

<a id="131-本轮阻塞排查进展"></a>

[13.1 本轮阻塞排查进展](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-568)

<a id="132-下一实验具体定位什么"></a>

[13.2 下一实验具体定位什么](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-582)

<a id="133-判据及证据边界"></a>

[13.3 判据及证据边界](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-606)

<a id="134-绿区命令"></a>

[13.4 绿区命令](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-621)

<a id="14-合并两边证据一次等待顺序对照后转向候选方案的成本验收2026-10-30"></a>

[14. 合并两边证据：一次等待顺序对照后，转向候选方案的成本验收（2026-10-30）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-639)

<a id="141-补入此前未同步的辅助流限核结果"></a>

[14.1 补入此前未同步的辅助流限核结果](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-645)

<a id="142-对另一对话修改的审阅结论"></a>

[14.2 对另一对话修改的审阅结论](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-665)

<a id="143-为什么优先保留延后等待而不是把限核当作最终方案"></a>

[14.3 为什么优先保留延后等待，而不是把限核当作最终方案](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-684)

<a id="144-保留现成对照设置明确的停止点"></a>

[14.4 保留现成对照，设置明确的停止点](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-695)

<a id="15-固定晚映射的早晚wait结果开始评估候选方案的成本2026-10-30"></a>

[15. 固定晚映射的早/晚wait结果：开始评估候选方案的成本（2026-10-30）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-725)

<a id="151-本轮阻塞结论"></a>

[15.1 本轮阻塞结论](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-727)

<a id="152-下一轮同一入口完成原callback与晚wait-mte的成本对照"></a>

[15.2 下一轮：同一入口完成原callback与晚wait MTE的成本对照](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-757)

<a id="16-短profile两组通过分开判断加载阻塞搬运成本与采集可用性2026-10-30"></a>

[16. 短profile两组通过：分开判断加载阻塞、搬运成本与采集可用性（2026-10-30）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-801)

<a id="161-两个分析脚本问题及修正"></a>

[16.1 两个分析脚本问题及修正](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-830)

<a id="162-先检查已有文件不重跑模型"></a>

[16.2 先检查已有文件，不重跑模型](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-849)

<a id="163-当前正确性范围请求计时与间隔统计口径"></a>

[16.3 当前正确性范围、请求计时与间隔统计口径](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-861)

<a id="17-开启cpp-lrc后的回调与晚等待mte对照2026-10-30"></a>

[17. 开启cpp LRC后的回调与晚等待MTE对照（2026-10-30）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-892)

<a id="171-实验目的与固定条件"></a>

[17.1 实验目的与固定条件](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-894)

<a id="172-一次执行"></a>

[17.2 一次执行](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-920)

<a id="173-大间隔筛选与数据判读"></a>

[17.3 大间隔筛选与数据判读](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-947)

<a id="174-绿区结果cpp_mte_join_profile_01"></a>

[17.4 绿区结果：cpp_mte_join_profile_01](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-971)

<a id="18-mte加载流单变量对照"></a>

[18. MTE加载流单变量对照](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-1056)

<a id="19-回调与两种mte加载流三组对照待绿区执行"></a>

[19. 回调与两种MTE加载流三组对照（待绿区执行）](moe_experiment_results_zh.md#wait-placement-20261030-moe_mte_reassessment_20261029_zh-1091)
