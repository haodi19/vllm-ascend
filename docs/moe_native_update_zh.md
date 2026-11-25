# MoE 原生 C++ 权重更新（实验扩展）

有效性能结果统一见 [MoE 卸载实验总记录](moe_experiment_results_zh.md#native-20261124)；本页不再记录独立实验。

<a id="moe-原生-c-权重更新实验扩展"></a>
<a id="实现范围"></a>

## 实现与兼容范围

当前实现用于单卡 MTE、C++ LRC、普通加载流和消费者处严格等待的 decode 图重放。`host_update_backend` 默认仍为 `python`；选择 `native` 后，专家规划、ACL 搬运提交、同步与映射更新由原生线程执行，执行期间不调用 Python，也不获取 GIL。初始化和图注册仍由 Python 完成。

截至 2026-11-25，本次实现包含以下配套修改：

- MXFP scale 在 checkpoint 加载时直接写入最终 dtype/stride 的 pinned CPU 缓冲。native 注册地址时必须复用该缓冲，不再保留第二份 scale；不满足布局条件就报错，禁止静默分配副本。
- native 的 eager/热专家预加载与 prefill 共用这份 CPU 缓冲。prefill scale 目标池保留对应的物理布局；布局、类型一致时使用带准确槽位偏移的 Storage 字节拷贝，避免非连续 CPU scale 到连续 NPU scale 的隐式转换。
- 其他已支持量化格式若需要最终转换，逐专家替换管理器中的原缓冲；FP32 派生 scale 单独计数。它是另一种必要表示，与同一表示的布局副本分开报告。当前 DeepSeek-V4-Flash MXFP 路径没有这项派生分配。
- 原 Python 配置继续使用原分配与搬运路径。注册后的地址受现有重加载保护约束，任务完成和错误传播沿用严格 MTE 握手。

上述修改不改变专家数量、缓存容量、预取数量或 MTP 配置。性能结论需来自新旧版本实测，不能将源码完成视为收益验收。

<a id="运行初次验证"></a>

## 验证与结果读取

继续使用 `script/run_native_moe_compare.py`，无需新增开关。它先运行 CPU 回归、NPU scale 字节一致性、严格图重放及 GIL 独立推进检查，再运行 Python/native 两组性能请求和独立 profile。结果统一汇总到指定目录的 `summary.txt`，保留 Decode、HTTP E2E、TTFT、TPOT、Host 间隔及加载流空泡。

新版 `NATIVE_SCALE_MEMORY` 记录会出现在计时与 profile 的摘要中：

| 字段 | 含义 |
| --- | --- |
| `policy=shared_final_layout_v1` | 本次共用最终布局缓冲的实现标识 |
| `duplicate_bytes` | 相同 scale 表示的额外布局副本；必须为 0 |
| `reused_bytes` | native 直接复用的 CPU scale 数据字节数 |
| `replaced_bytes` | 其他量化格式替换原缓冲的数据字节数；不与旧缓冲长期并存 |
| `derived_bytes` | 必要的 FP32 派生表示占用；MXFP 本次应为 0 |

缺少该记录或重复副本不为零时，脚本停止验收。`scale_buffer_bytes` 旧字段保留用于兼容，当前只计额外派生表示；不能据此推算进程 RSS 或 pinned allocator 的缓存占用。

<a id="验证状态"></a>

本地 CPU 测试覆盖加载分片顺序、原缓冲地址复用、prefill/eager 非零槽位、布局不匹配拒绝、替换后旧缓冲释放和原生任务执行。NPU 检查已集成到原测试入口；需要在目标服务器上运行后才能确认新版本的模型输出与性能。既有实验数据仍对应当时版本，不回填为本次改动的效果。
