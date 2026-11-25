# DeepSeek-V4 单卡 MoE 卸载测试基线与专家加载记录

实验记录已统一迁入 [MoE 卸载实验总记录](moe_experiment_results_zh.md#dt-20261105)。本页只保留旧链接入口，后续结果请更新总记录。

<a id="deepseek-v4-单卡-moe-卸载测试基线与专家加载记录"></a>

[DeepSeek-V4 单卡 MoE 卸载测试基线与专家加载记录](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-1)

<a id="1-四组配置的含义"></a>

[1. 四组配置的含义](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-11)

<a id="2-测试条件"></a>

[2. 测试条件](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-33)

<a id="3-内存与专家缓存配置"></a>

[3. 内存与专家缓存配置](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-59)

<a id="31-hbm-占用"></a>

[3.1 HBM 占用](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-61)

<a id="32-专家数量缓存和预取"></a>

[3.2 专家数量、缓存和预取](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-81)

<a id="4-四组性能结果"></a>

[4. 四组性能结果](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-104)

<a id="41-整体请求时延"></a>

[4.1 整体请求时延](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-106)

<a id="42-策略计算与权重加载分项"></a>

[4.2 策略计算与权重加载分项](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-121)

<a id="43-host-回调间隔与设备加载流空泡"></a>

[4.3 Host 回调间隔与设备加载流空泡](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-148)

<a id="44-tpot"></a>

[4.4 TPOT](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-183)

<a id="5-实际专家-h2d-加载行为"></a>

[5. 实际专家 H2D 加载行为](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-219)

<a id="51-单专家传输组成与总量"></a>

[5.1 单专家传输组成与总量](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-226)

<a id="52-各轮加载量"></a>

[5.2 各轮加载量](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-255)

<a id="53-各轮各层加载次数"></a>

[5.3 各轮各层加载次数](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-264)

<a id="54-专家-id-示例"></a>

[5.4 专家 ID 示例](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-315)

<a id="6-用于-hhm-对照的基线结论"></a>

[6. 用于 HHM 对照的基线结论](moe_experiment_results_zh.md#dt-20261105-moe_decode_baseline_zh-327)
