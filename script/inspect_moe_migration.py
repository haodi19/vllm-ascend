# SPDX-License-Identifier: Apache-2.0
"""在宿主机一次检查设备/URMA依赖及目标容器；只读、无安装、无模型启动。"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_FIELDS = (
    "model_type",
    "architectures",
    "torch_dtype",
    "dtype",
    "hidden_size",
    "num_hidden_layers",
    "n_routed_experts",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "quantization_config",
    "quantization",
    "compression_config",
)
LIB_ROOTS = (
    "/usr/lib64",
    "/usr/lib",
    "/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/usr/local/Ascend/driver",
    "/opt",
)
MAX_DIRS = 500
MAX_ENTRIES = 2000
SEARCH_SECONDS = 20


def emit(message):
    print(re.sub(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@", r"\1<credentials>@", str(message)), flush=True)


def capture(args, timeout=30, input_text=None):
    try:
        result = subprocess.run(
            args,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT（不表示硬件故障）"
    except OSError as exc:
        return 127, "", str(exc)


def show(label, args, timeout=30):
    code, out, err = capture(args, timeout)
    emit(f"\n{label}: exit={code}")
    emit((out + err)[-20000:].strip() or "（无输出）")
    return code


def elf_machine(path):
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(20)
        if len(header) < 20 or header[:4] != b"\x7fELF" or header[5] not in (1, 2):
            return "not_ELF"
        code = int.from_bytes(header[18:20], "little" if header[5] == 1 else "big")
        return {62: "x86_64", 183: "aarch64"}.get(code, str(code))
    except OSError:
        return "unreadable"


def find_urma(roots):
    queue = [(Path(root), 0) for root in roots if Path(root).is_dir()]
    visited, found = set(), set()
    start = time.monotonic()
    limited = False
    while queue:
        if len(visited) >= MAX_DIRS or time.monotonic() - start >= SEARCH_SECONDS:
            limited = True
            break
        root, depth = queue.pop(0)
        resolved = str(root.resolve())
        if resolved in visited:
            continue
        visited.add(resolved)
        try:
            with os.scandir(root) as items:
                for index, entry in enumerate(items):
                    if index >= MAX_ENTRIES:
                        limited = True
                        break
                    if entry.name.startswith("liburma.so") and entry.is_file():
                        found.add(str(Path(entry.path).absolute()))
                    elif depth < 4 and entry.is_dir(follow_symlinks=False):
                        if entry.name not in ("docker", "overlay2", "containerd", "site-packages", "opp", ".git"):
                            queue.append((Path(entry.path), depth + 1))
        except OSError:
            limited = True
    return sorted(found), limited


def local_platform():
    emit(f"SYSTEM arch={platform.machine()}; python={sys.executable}")
    devices = sorted(
        {
            p
            for pattern in ("/dev/davinci*", "/dev/hisi*", "/dev/ummu*", "/dev/uburma*", "/dev/ubcore*", "/dev/udma*")
            for p in glob.glob(pattern)
        }
    )
    for value in devices:
        path = Path(value)
        emit(f"DEVICE {value}; char={path.is_char_device()}; dir={path.is_dir()}")
        if path.is_dir():
            emit(f"  children={sorted(p.name for p in path.iterdir())[:30]}")
    if not devices:
        emit("DEVICE: 未找到指定前缀的设备节点")
    smi = shutil.which("npu-smi")
    if smi:
        emit(f"NPU_SMI path={smi}; resolved={Path(smi).resolve()}; ELF={elf_machine(smi)}")
        show("NPU状态", [smi, "info"])
        if shutil.which("ldd"):
            show("npu-smi动态依赖", ["ldd", smi])
    else:
        emit("NPU_SMI: PATH中未找到")
    ldconfig = shutil.which("ldconfig") or ("/sbin/ldconfig" if Path("/sbin/ldconfig").is_file() else None)
    if ldconfig:
        code, out, _ = capture([ldconfig, "-p"])
        emit(f"LDCONFIG_READ_ONLY exit={code}")
        emit("\n".join(line for line in out.splitlines() if "urma" in line.lower()) or "缓存中没有URMA条目")
    libraries, limited = find_urma(LIB_ROOTS)
    for library in libraries[:12]:
        emit(f"URMA {library} -> {Path(library).resolve()}; ELF={elf_machine(library)}")
        if shutil.which("ldd"):
            show("liburma动态依赖", ["ldd", library])
    emit(f"URMA_SEARCH found={len(libraries)} limited={limited}；限范围搜索，未发现不等于不存在")
    if shutil.which("ip"):
        show("IPv4接口", ["ip", "-br", "-4", "addr"])


def source_and_models(repo, model_root):
    emit(f"\nSOURCE {repo}; exists={repo.is_dir()}")
    if (repo / ".git").exists() and shutil.which("git"):
        # Read-only Git queries; no fetch/checkout/commit/push.
        show("源码HEAD", ["git", "-C", str(repo), "rev-parse", "HEAD"])
        show("源码分支", ["git", "-C", str(repo), "branch", "--show-current"])
    for relative in (
        "script/run_moe_callback_worker_compare.py",
        "script/moe_outlier_stats.py",
        "vllm_ascend/_build_info.py",
        "vllm_ascend/_cann_ops_custom/vendors/custom_transformer",
        "vllm_ascend/expert_offload/expert_rank_gsm8k.json",
    ):
        emit(f"SOURCE_FILE {relative} exists={(repo / relative).exists()}")
    emit(f"\nMODEL_ROOT {model_root}; exists={model_root.is_dir()}")
    if not model_root.is_dir():
        return
    for folder in sorted(model_root.iterdir()):
        path = folder / "config.json"
        if "deepseek" not in folder.name.lower() or not path.is_file():
            continue
        emit(f"MODEL {folder}")
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("config.json超过1MiB，不自动读取")
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("config.json不是对象")
            for section, value in (("root", config), ("text_config", config.get("text_config", {}))):
                if not isinstance(value, dict):
                    continue
                selected = {key: value[key] for key in MODEL_FIELDS if key in value}
                emit(f"  {section}=" + json.dumps(selected, ensure_ascii=False)[:5000])
            index = folder / "model.safetensors.index.json"
            emit(f"  weight_index={index.is_file()}; tokenizer_config={(folder / 'tokenizer_config.json').is_file()}")
        except (OSError, ValueError, TypeError) as exc:
            emit(f"  CONFIG_READ_FAILED {exc}")


def docker_summary(record):
    host = record.get("HostConfig", {})
    return dict(
        name=record.get("Name"),
        image=record.get("Config", {}).get("Image"),
        status=record.get("State", {}).get("Status"),
        settings={
            key: host.get(key)
            for key in ("Privileged", "Runtime", "NetworkMode", "IpcMode", "ShmSize", "Devices", "DeviceRequests")
        },
        mounts=[{key: item.get(key) for key in ("Source", "Destination", "RW")} for item in record.get("Mounts", [])],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("host", "container"), default="host")
    parser.add_argument("--container", help="host模式必填，只读取指定容器")
    parser.add_argument("--repo", type=Path, default=Path("/home/l00992607/vllm-ascend"))
    parser.add_argument("--model-root", type=Path, default=Path("/data/model"))
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("在Linux宿主机或容器运行")
    if args.mode == "host" and (not args.container or args.container.startswith("-")):
        parser.error("host模式需要 --container 容器名（用docker ps查看）；只查询该容器")
    emit(f"=== MIGRATION CHECK mode={args.mode} ===")
    emit("只读；不安装、不修改挂载、不创建或停止容器、不导入Torch、不读取权重数据、不写结果文件。")
    local_platform()
    if args.mode == "container":
        source_and_models(args.repo, args.model_root)
        return 0
    if not shutil.which("docker"):
        emit("FAILED: 宿主机未找到docker命令")
        return 1
    code, out, err = capture(["docker", "inspect", args.container])
    if code:
        emit(f"DOCKER_INSPECT_FAILED {err}")
        return 1
    data = json.loads(out)
    if len(data) != 1:
        raise ValueError("无法唯一识别容器")
    emit("CONTAINER_CONFIG " + json.dumps(docker_summary(data[0]), ensure_ascii=False))
    if data[0].get("State", {}).get("Running") is not True:
        emit("容器未运行；没有自动启动。请运行容器后重试。")
        return 1
    emit("\n=== 同一脚本通过标准输入在目标容器内运行，无文件复制 ===")
    code, out, err = capture(
        [
            "docker",
            "exec",
            "-i",
            args.container,
            "python3",
            "-B",
            "-",
            "--mode",
            "container",
            "--repo",
            str(args.repo),
            "--model-root",
            str(args.model_root),
        ],
        timeout=180,
        input_text=Path(__file__).read_text(encoding="utf-8"),
    )
    emit(out)
    emit(err)
    emit(f"CONTAINER_CHECK exit={code}；请检查脱敏后返回以上输出。")
    return int(code != 0)


if __name__ == "__main__":
    raise SystemExit(main())
