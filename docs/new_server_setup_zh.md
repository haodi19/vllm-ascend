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

## 3. 新容器首轮快照与需要先处理的差异（2026-11-03）

新容器为aarch64，旧实验宿主CPU为x86_64。Python 3.11.10的SSL导入成功，
CANN 9.1.0的头文件/AscendC CMake存在；torch-npu元数据版本为2.10.0.post4.dev20260715。
但npu-smi由于缺少liburma.so.0退出，尚未确认NPU型号及可用性。
设备列表只发现davinci_manager、hisi_hdc、ummu；是否缺少计算设备映射需要与宿主机对照，不能仅据此判定卡坏了。

镜像自带vLLM 0.23和vllm-ascend 0.23，transformers为5.5.4，缺少fastokens、setuptools-rust及本项目两个Host扩展。
这些与原实验软件配置不同。后续需安装固定的vLLM源码、本开发分支及对应依赖，
并在aarch64容器重新构建扩展，不能复制旧x86_64的.so，也不能复用其他人的0.23分支算子包来充当匹配构建。

已发现自己的目录`/home/l00992607/vllm-ascend`和原名热点JSON，但尚未确认该目录分支/提交。
`/data/model`下发现的是DeepSeek-V4.1-Flash及w8a8变体，不能仅改路径就认为复现了旧V4-Flash负载。
需要核对architecture、量化配置及模型来源；更换CPU架构后也要在新机上重新做回调/MTE配对，不能直接比较两台机器绝对时延。
首次搜索还遍历到了`/data/docker/overlay2`中的镜像层，这些不是开发目录；后续脚本已排除Docker存储目录。

将`script/inspect_moe_migration.py`复制到宿主机的`/tmp/inspect_moe_migration.py`，在宿主机执行：

```bash
docker ps --format '{{.Names}}'
python3 -B /tmp/inspect_moe_migration.py --container 实际容器名
```

脚本读取宿主机npu-smi及其动态依赖，查找liburma，并读取指定容器的设备/挂载配置。
随后通过docker exec标准输入在同一个容器内运行读取部分，一次返回宿主机与容器的对照。
不会创建、停止或改动容器，不安装库，不导入Torch或运行模型，不读取权重内容，不写输出文件。
不会打印docker inspect中的环境变量；只读取本人的指定源码目录和模型配置中的结构/量化字段。

这一步用于决定如何补齐匹配的URMA依赖和必要设备映射。
不要为解决liburma缺失重新挂载整个宿主机/usr/lib64，以免再次覆盖容器OpenSSL等系统库。
如果实际源码或模型目录不同，可追加`--repo 目录 --model-root 目录`。
