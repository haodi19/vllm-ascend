# SPDX-License-Identifier: Apache-2.0
"""读取新容器的MoE调试环境；不安装、不写文件、不导入Torch或初始化NPU。"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

PACKAGES = (
    "pip",
    "setuptools",
    "setuptools-scm",
    "setuptools-rust",
    "wheel",
    "packaging",
    "cmake",
    "ninja",
    "pybind11",
    "torch",
    "torch-npu",
    "torchvision",
    "torchaudio",
    "vllm",
    "vllm-ascend",
    "transformers",
    "fastokens",
    "fastapi",
    "starlette",
    "pydantic",
    "numpy",
    "triton-ascend",
    "xgrammar",
    "compressed-tensors",
    "memfabric-hybrid",
    "memcache-hybrid",
    "arctic-inference",
    "vllm-ascend-host-sync",
    "vllm-ascend-lrc",
)
SEARCH_ROOTS = ("/home", "/mnt", "/data", "/workspace", "/root")
SKIP_DIRS = frozenset(
    (
        ".git",
        ".cache",
        "__pycache__",
        "node_modules",
        "site-packages",
        "dist-packages",
        "build",
        ".deps",
        ".venv",
        "venv",
        "ASCEND_PROFILER_OUTPUT",
        "FRAMEWORK",
        "logs",
        "log",
        "moe_profile",
        "bench_results",
        "bench_results2",
        "bench_results_tmp",
    )
)
MAX_TEXT_BYTES = 64 * 1024
MAX_COMMAND_BYTES = 256 * 1024
MAX_RESULTS = 40
MAX_ENTRIES_PER_DIR = 1000
DEFAULT_MAX_DIRS = 3000
DEFAULT_DEPTH = 5
SEARCH_SECONDS = 30
SECRET_URL = re.compile(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@", re.IGNORECASE)


def emit(text):
    print(SECRET_URL.sub(r"\1<credentials>@", str(text)), flush=True)


def read_text(path, lines=12):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_TEXT_BYTES)
        return "\n".join(data.decode("utf-8", errors="replace").splitlines()[:lines])
    except OSError as exc:
        return f"不可读取：{type(exc).__name__}"


def command(label, args, timeout=20):
    """只运行固定的只读子命令；不执行shell、不读取pip代理配置。"""
    emit(f"\n{label}")
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
        output = (result.stdout + result.stderr)[:MAX_COMMAND_BYTES].decode("utf-8", errors="replace")
        emit(output.strip() or "（无输出）")
        emit(f"exit={result.returncode}")
    except subprocess.TimeoutExpired:
        emit(f"TIMEOUT: {timeout}s；未取得该项结果")
    except OSError as exc:
        emit(f"无法执行：{type(exc).__name__}")


def package_info(name):
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return f"{name}: MISSING"
    result = f"{name}={dist.version}; location={dist.locate_file('')}"
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        url = urlsplit(direct.get("url", ""))
        if url.scheme == "file":
            result += f"; source={unquote(url.path)}; editable={direct.get('dir_info', {}).get('editable', False)}"
    except (OSError, ValueError, TypeError):
        result += "; source_metadata_unreadable"
    return result


def mount_paths(text):
    """仅输出容器内挂载点及类型，省略宿主机来源和挂载选项。"""
    result = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 7 or "-" not in fields:
            continue
        index = fields.index("-")
        if index + 1 >= len(fields):
            continue
        path = fields[4].replace(r"\040", " ").replace(r"\134", "\\")
        if path in ("/", "/home", "/mnt", "/data", "/dev/shm", "/lib64", "/usr/lib64") or path.startswith(
            ("/usr/local/Ascend", "/dev/davinci", "/dev/uburma", "/dev/ummu")
        ):
            result.append((path, fields[index + 1]))
    return result


def scan_paths(roots, depth_limit, dir_limit):
    """有界搜索已知项目/模型/热点文件；不跟随目录符号链接，不读取权重。"""
    pending = [(Path(root), 0) for root in roots if Path(root).is_dir()]
    visited, found, limits = set(), set(), set()
    start = time.monotonic()
    while pending:
        if len(visited) >= dir_limit or time.monotonic() - start > SEARCH_SECONDS:
            limits.add("directory/time limit")
            break
        directory, depth = pending.pop()
        real = str(directory.resolve())
        if real in visited:
            continue
        visited.add(real)
        for kind, relative in (("ASCEND_REPO", "vllm_ascend/__init__.py"), ("VLLM_REPO", "vllm/__init__.py")):
            if (directory / relative).is_file():
                found.add((kind, str(directory)))
        try:
            with os.scandir(directory) as entries:
                for count, entry in enumerate(entries):
                    if count >= MAX_ENTRIES_PER_DIR:
                        limits.add("entries per directory limit")
                        break
                    path = Path(entry.path)
                    if entry.is_file(follow_symlinks=True):
                        if entry.name == "expert_rank_gsm8k.json":
                            found.add(("HOT_EXPERTS", str(path)))
                        elif entry.name == "config.json" and "deepseek" in directory.name.lower():
                            found.add(("MODEL_CONFIG", str(path)))
                    elif entry.is_dir(follow_symlinks=True):
                        if (path / "vllm_ascend/__init__.py").is_file():
                            found.add(("ASCEND_REPO", str(path)))
                            for relative in (
                                "expert_rank_gsm8k.json",
                                "vllm_ascend/expert_offload/expert_rank_gsm8k.json",
                            ):
                                if (path / relative).is_file():
                                    found.add(("HOT_EXPERTS", str(path / relative)))
                        if (path / "vllm/__init__.py").is_file():
                            found.add(("VLLM_REPO", str(path)))
                        if depth < depth_limit and entry.name not in SKIP_DIRS and not entry.is_symlink():
                            pending.append((path, depth + 1))
                    if len(found) >= MAX_RESULTS:
                        limits.add("result limit")
                        return sorted(found), len(visited), sorted(limits)
        except OSError:
            limits.add("unreadable directories")
    return sorted(found), len(visited), sorted(limits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", action="append", type=Path, help="替换默认搜索范围，可重复指定")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--max-dirs", type=int, default=DEFAULT_MAX_DIRS)
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.depth <= 10 or not 1 <= args.max_dirs <= 20000:
        parser.error("depth应为0～10，max-dirs应为1～20000")
    if sys.platform != "linux":
        parser.error("请在新服务器的Linux容器内执行")
    emit("=== 新MoE容器环境快照：只读、控制台输出 ===")
    emit("不安装、不修改系统配置、不写结果文件、不导入Torch/torch_npu、不启动模型或NPU计算。")
    emit("仅查询npu-smi状态；未找到文件不等于服务器上不存在，限深度搜索不会扫描所有挂载。")
    emit("\n[1] 系统 / Python / SSL")
    emit(read_text("/etc/os-release"))
    emit(f"arch={platform.machine()}; kernel={platform.release()}")
    emit(f"python={sys.executable}; version={sys.version.splitlines()[0]}")
    for name in ("python", "python3", "pip", "git", "gcc", "g++", "make", "cmake", "ninja", "npu-smi"):
        emit(f"PATH {name}={shutil.which(name) or 'MISSING'}")
    command("SSL导入", [sys.executable, "-B", "-c", "import ssl; print(ssl.OPENSSL_VERSION)"])
    command("pip所属环境", [sys.executable, "-B", "-m", "pip", "--version"])
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        emit(f"{name}: {'SET (value hidden)' if os.environ.get(name) else 'UNSET'}")
    for name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "PYTHONHOME",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
    ):
        emit(f"{name}={os.environ.get(name, '<unset>')}")

    emit("\n[2] Python包元数据（不加载框架）")
    for name in PACKAGES:
        emit(package_info(name))

    emit("\n[3] CANN / ATB / 驱动 / NPU")
    toolkit = Path("/usr/local/Ascend/ascend-toolkit/latest")
    ascend = Path("/usr/local/Ascend")
    if ascend.is_dir():
        emit(f"ASCEND directories={sorted(p.name for p in ascend.iterdir() if p.is_dir())[:20]}")
    emit(f"CANN latest={toolkit.resolve()}; exists={toolkit.exists()}")
    for path in (
        toolkit / "set_env.sh",
        toolkit.parent / "set_env.sh",
        Path("/usr/local/Ascend/nnal/atb/set_env.sh"),
        Path("/usr/local/Ascend/nnal/atb/latest/atb/set_env.sh"),
        toolkit / "include/acl/acl_rt.h",
        toolkit / "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    ):
        emit(f"FILE {path}: {'FOUND' if path.is_file() else 'MISSING'}")
    for root in (
        toolkit,
        toolkit / platform.machine() / "lib64",
        toolkit / f"{platform.machine()}-linux",
        Path("/usr/local/Ascend/driver"),
    ):
        for name in ("version.info", "version.cfg", "ascend_toolkit_install.info"):
            path = root / name
            if path.is_file():
                emit(f"VERSION {path}\n{read_text(path, 8)}")
    devices = sorted(str(p) for p in Path("/dev").glob("davinci*"))
    devices += [str(p) for p in (Path("/dev/hisi_hdc"), Path("/dev/ummu"), Path("/dev/uburma")) if p.exists()]
    emit(f"DEVICE_NODES={devices}")
    smi = shutil.which("npu-smi")
    if smi:
        command("NPU状态", [smi, "info"], timeout=30)
    else:
        emit("npu-smi不在PATH；尚不能确定设备型号或驱动挂载是否正常。")

    emit("\n[4] 容器挂载 / 内存 / 磁盘")
    mounts = mount_paths(read_text("/proc/self/mountinfo", 10000))
    for path, filesystem in mounts:
        emit(f"MOUNT {path} type={filesystem}")
    if any(path in ("/lib64", "/usr/lib64") for path, _ in mounts):
        emit("CHECK: 整个系统库目录是单独挂载点；若SSL报OpenSSL版本缺失，优先核对docker run。")
    for line in read_text("/proc/meminfo", 80).splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
            emit(line)
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        if Path(path).is_file():
            emit(f"CGROUP {path}={read_text(path, 1)}")
    for root in ("/", "/dev/shm", "/home", "/mnt", "/data"):
        if Path(root).exists():
            usage = shutil.disk_usage(root)
            emit(f"SPACE {root}: total={usage.total / 1024**3:.2f} GiB free={usage.free / 1024**3:.2f} GiB")
    ip = shutil.which("ip")
    if ip:
        command("容器IPv4接口（用于替换旧脚本网卡/IP）", [ip, "-br", "-4", "addr"])

    emit("\n[5] 项目 / 模型 / 热点专家文件")
    if args.skip_search:
        emit("搜索已跳过")
    else:
        roots = args.search_root or [Path(p) for p in SEARCH_ROOTS]
        emit(f"SEARCH roots={list(map(str, roots))}; depth={args.depth}; max_dirs={args.max_dirs}")
        found, visited, limits = scan_paths(roots, args.depth, args.max_dirs)
        for kind, value in found:
            emit(f"{kind}: {value}")
            if kind == "ASCEND_REPO":
                package = Path(value) / "vllm_ascend"
                emit(
                    f"  _build_info={(package / '_build_info.py').is_file()}; "
                    f"custom_ops={(package / '_cann_ops_custom/vendors/custom_transformer').is_dir()}"
                )
        emit(f"SEARCH_RESULT found={len(found)} visited_dirs={visited} limits={limits}")
        if not found:
            emit("搜索范围内未发现目标；后续需要准备源码、模型挂载及原热点文件，不能用空文件替代。")
    emit("\n=== 检查结束；这不是模型/算子运行验收。请检查脱敏后返回控制台输出。 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
