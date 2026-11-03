# 新服务器容器配置

## 1. 先确认新容器基础环境

当前工作分支是`dev/moeoffload-worker-poll`。旧实验环境为Ascend950PR_9579、CANN 9.1.0、
Python 3.11、torch 2.10.0+cpu、torch-npu 2.10.0.post4.dev20260715；
vLLM源码曾固定在`568afb3a13806beb53bb2e6bd518269357b237c0`。
这些是历史实测信息，不表示新容器已经具备相同环境，也不能用普通最新版依赖自动替换。

新容器尚未安装项目时，先将`script/inspect_new_moe_container.py`的内容复制到容器内，
例如保存为`/tmp/inspect_new_moe_container.py`，再执行：

```bash
python3 -B /tmp/inspect_new_moe_container.py
```

脚本只依赖Python标准库，读取系统、Python/SSL、包元数据、CANN路径、设备节点及npu-smi状态。
不安装依赖、不修改系统配置、不导入Torch/torch_npu、不运行NPU计算，也不写结果文件。
代理只显示是否设置，不打印代理地址或凭据；控制台输出须人工检查脱敏后再返回。

项目、模型和热点文件搜索默认限制为`/home`、`/mnt`、`/data`、`/workspace`、`/root`，
不递归进入结果目录、构建目录或目录符号链接。找不到不表示服务器上绝对不存在。
如果文件放在其他挂载目录，直接指定实际目录，可重复传参：

```bash
python3 -B /tmp/inspect_new_moe_container.py --search-root /实际挂载目录
```

## 2. 根据结果准备安装配置

需根据快照确认匹配的镜像/框架、源码位置、模型挂载、原`expert_rank_gsm8k.json`和算子构建产物。
若SSL仍因OpenSSL版本失败，先核对容器系统库挂载；不通过覆盖libcrypto或关闭证书验证处理。
若CANN、设备或torch-npu缺失，先补齐匹配的基础环境，再安装项目。
不能伪造热点JSON或_build_info来绕过检查，也不能把旧服务器的网卡/IP和绝对路径直接沿用。

完成上述确认后，再生成针对该容器的安装与启动脚本；当前环境快照不代表模型已经验收通过。
