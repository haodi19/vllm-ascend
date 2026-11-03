# SPDX-License-Identifier: Apache-2.0
"""宿主机执行：向指定ARM容器补齐两项URMA库，再验证链接和设备查询。"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LIBRARIES = ("liburma.so.0", "liburma_common.so.0")
CONF = "/etc/ld.so.conf.d/90-moe-urma.conf"
CONTAINER_PREPARE = r"""
import hashlib, json, os, platform, sys
from pathlib import Path
target, manifest = Path(sys.argv[1]), json.loads(sys.argv[2])
assert platform.machine() == "aarch64", "target container must be aarch64"
assert os.geteuid() == 0, "container exec must run as root"
assert str(target).startswith("/opt/moe-urma/"), "invalid destination"
assert target.resolve() == target, "destination must not traverse symlinks"
target.mkdir(parents=True, exist_ok=True)
copy = []
for name, digest in manifest.items():
    path = target / name
    assert not path.is_symlink(), "existing library is a symlink"
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "existing file differs"
    else:
        copy.append(name)
print(json.dumps(copy))
"""
CONTAINER_VALIDATE = r"""
import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
target, manifest = Path(sys.argv[1]), json.loads(sys.argv[2])
for name, digest in manifest.items():
    assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest, "copied file checksum mismatch"
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = str(target) + ":" + env.get("LD_LIBRARY_PATH", "")
smi, ldd = shutil.which("npu-smi"), shutil.which("ldd")
assert smi and ldd, "npu-smi/ldd missing"
result = subprocess.run([ldd, smi], env=env, capture_output=True, text=True, timeout=20)
output = result.stdout + result.stderr
print(output, flush=True)
assert result.returncode == 0 and not re.search(r"not found", output, re.I), "dynamic dependencies still unresolved"
for name in manifest:
    assert str(target / name) in output, "expected URMA library was not selected"
print("URMA temporary link check PASS", flush=True)
"""
CONTAINER_REGISTER = r"""
import shutil, subprocess, sys
from pathlib import Path
target, conf = Path(sys.argv[1]), Path(sys.argv[2])
assert target.resolve() == target and conf.parent.resolve() == conf.parent, "symlink in managed paths"
assert not conf.is_symlink(), "configuration is a symlink"
expected = str(target) + "\n"
if conf.exists():
    assert conf.read_text() == expected, "existing URMA config differs; not overwritten"
else:
    with conf.open("x") as f:
        f.write(expected)
ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
subprocess.run([ldconfig], check=True, timeout=30)
print("URMA registered: " + str(conf), flush=True)
"""
CONTAINER_CHECK = r"""
import re, shutil, ssl, subprocess
result = subprocess.run(["ldd", shutil.which("npu-smi")], capture_output=True, text=True, timeout=20)
output = result.stdout + result.stderr
print(output, flush=True)
assert result.returncode == 0 and not re.search(r"not found", output, re.I), "persistent link check failed"
assert "/opt/moe-urma/" in output, "managed libraries not loaded"
print("SSL: " + ssl.OPENSSL_VERSION, flush=True)
assert shutil.which("timeout"), "coreutils timeout missing"
print("LIBRARY_FIX=PASS; NPU health not yet verified", flush=True)
"""


def arm_library(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as source:
        header = source.read(20)
    if len(header) < 20 or header[:4] != b"\x7fELF" or header[5] not in (1, 2):
        raise ValueError(f"不是有效ELF库：{path}")
    machine = int.from_bytes(header[18:20], "little" if header[5] == 1 else "big")
    if machine != 183:
        raise ValueError(f"不是aarch64库：{path}")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def execute(label, argv, *, source=None, timeout=60, required=True, quiet=False):
    print(f"START {label}", flush=True)
    try:
        result = subprocess.run(
            argv,
            input=source,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label}超出外层等待预算；未确认成功") from exc
    if not quiet or result.returncode:
        print((result.stdout + result.stderr)[-24000:].strip(), flush=True)
    print(f"END {label}: exit={result.returncode}", flush=True)
    if required and result.returncode:
        raise RuntimeError(f"{label}失败；停止后续修改")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="moeoffload_ljn")
    parser.add_argument("--apply", action="store_true", help="实际复制两库并在容器注册；不重建容器")
    args = parser.parse_args(argv)
    if sys.platform != "linux" or platform.machine() != "aarch64":
        parser.error("在这台aarch64宿主机执行")
    if not args.container or args.container.startswith("-"):
        parser.error("无效容器名")
    for tool in ("docker", "timeout", "npu-smi"):
        if not shutil.which(tool):
            parser.error(f"宿主机缺少命令：{tool}")
    libraries = {name: arm_library(Path("/usr/lib64") / name) for name in LIBRARIES}
    manifest = {name: data[1] for name, data in libraries.items()}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    target = f"/opt/moe-urma/{fingerprint}"
    inspected = execute("读取目标容器", ["docker", "inspect", args.container], quiet=True)
    records = json.loads(inspected.stdout)
    if len(records) != 1 or not records[0].get("State", {}).get("Running"):
        raise ValueError("目标容器必须唯一且正在运行")
    for name, (path, digest) in libraries.items():
        print(f"COPY PLAN {path} -> {args.container}:{target}/{name}; sha256={digest}", flush=True)
    print(f"CONFIG PLAN {CONF} -> {target}", flush=True)
    print("只复制这两个ARM库；不覆盖/lib64或/usr/lib64，不更改设备映射，不停止容器。", flush=True)
    if not args.apply:
        print("未修改；执行时添加--apply。", flush=True)
        return 0
    prefix = ["docker", "exec", "-i", args.container, "python3", "-B", "-"]
    parameters = [target, json.dumps(manifest)]
    prepared = execute("创建独立库目录", [*prefix, *parameters], source=CONTAINER_PREPARE, quiet=True)
    for name in json.loads(prepared.stdout):
        execute(f"复制{name}", ["docker", "cp", str(libraries[name][0]), f"{args.container}:{target}/{name}"])
    execute("先验证两库及完整动态依赖", [*prefix, *parameters], source=CONTAINER_VALIDATE)
    execute("注册容器动态库搜索路径", [*prefix, target, CONF], source=CONTAINER_REGISTER)
    execute("验证持久链接和SSL", prefix, source=CONTAINER_CHECK)
    host = execute(
        "宿主机NPU查询（最多约50秒）", ["timeout", "-k", "5s", "45s", "npu-smi", "info"], required=False, timeout=65
    )
    container = execute(
        "容器NPU查询（最多约50秒）",
        ["docker", "exec", args.container, "timeout", "-k", "5s", "45s", "npu-smi", "info"],
        required=False,
        timeout=70,
    )
    print(
        f"SUMMARY library_fix=PASS host_npu_exit={host.returncode} container_npu_exit={container.returncode}",
        flush=True,
    )
    if host.returncode or container.returncode:
        print("缺库已修复，但设备查询未同时通过；不要据此开始模型测试。124/137表示超时或被超时工具终止。", flush=True)
        return 2
    print("两端NPU查询通过；下一步安装固定源码与ARM扩展，尚未启动模型或验证计算。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAILED: {exc}", flush=True)
        raise SystemExit(1) from None
