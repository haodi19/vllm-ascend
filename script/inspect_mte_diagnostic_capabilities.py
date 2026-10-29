# SPDX-License-Identifier: Apache-2.0
"""只读检查本机 CANN 诊断接口与工具；不初始化 NPU、不启动模型、不写文件。"""

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess
from pathlib import Path

MAX_HEADERS = 2500
MAX_HEADER_BYTES = 2 * 1024 * 1024
MAX_APIS = 35
MAX_TOOLS = 25
API_PATTERN = re.compile(
    r"\b(?:aclrt|rt)[A-Za-z0-9_]*(?:Debug|Dump|Snapshot|TaskInfo|TaskId|TaskName|"
    r"ModelInfo|ModelGet|CaptureGet|GraphGet|ExceptionInfo)[A-Za-z0-9_]*\s*\("
)
TOOL_PATTERN = re.compile(
    r"^(?:msprof|msdebug|msopprof|msopgen|msaccucmp|npu-smi|"
    r"ascend-dmi|msobjdump|llvm-objdump|ccec|bisheng)(?:[.-].*)?$"
)


def declarations(text):
    """Select declarations, not unrelated API mentions in documentation comments."""
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    result = []
    for match in API_PATTERN.finditer(clean):
        end = clean.find(";", match.start())
        if end != -1 and end - match.start() < 1200:
            result.append(" ".join(clean[match.start() : end + 1].split()))
    return list(dict.fromkeys(result))


def walk_files(root, max_depth):
    for directory, dirs, names in os.walk(root, followlinks=False):
        if len(Path(directory).relative_to(root).parts) >= max_depth:
            dirs[:] = []
        for name in names:
            yield Path(directory) / name


def inspect(root):
    print("READ ONLY: 不导入 torch/torch_npu，不初始化设备，不启动诊断工具，不写文件。")
    print(f"CANN_ROOT={root}; resolved={root.resolve()}; exists={root.is_dir()}")
    for package in ("torch", "torch-npu"):
        try:
            print(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}=NOT_INSTALLED")
    version_paths = [root / "version.cfg", root / "version.info", root / "ascend_toolkit_install.info"]
    for path in version_paths:
        if path.is_file():
            print(f"VERSION {path}")
            print(path.read_text(encoding="utf-8", errors="replace")[:1200])
    headers = []
    seen = set()
    for folder in (
        root / "include",
        root / "runtime/include",
        root / "x86_64-linux/include",
        root / "aarch64-linux/include",
    ):
        if not folder.is_dir():
            continue
        for path in walk_files(folder, 6):
            resolved = path.resolve()
            if path.suffix in (".h", ".hpp") and resolved not in seen:
                seen.add(resolved)
                headers.append(path)
    # ACL/Runtime public headers first; unrelated operator headers must not crowd them out.
    headers.sort(key=lambda path: (not any(x in str(path).lower() for x in ("acl", "runtime", "rt_")), str(path)))
    found = {}
    for path in headers[:MAX_HEADERS]:
        if path.stat().st_size > MAX_HEADER_BYTES:
            continue
        for declaration in declarations(path.read_text(encoding="utf-8", errors="replace")):
            found.setdefault(declaration, path)
    ordered = sorted(found, key=lambda value: (not any(x in value for x in ("Debug", "Dump", "Snapshot")), value))
    print(f"HEADERS discovered={len(headers)} scanned_at_most={MAX_HEADERS}; candidate_declarations={len(found)}")
    for declaration in ordered[:MAX_APIS]:
        print(f"API {found[declaration]}: {declaration}")
    print("头文件声明不等于当前硬件支持；没有匹配也不证明不存在其他接口。")
    tools = set()
    for name in ("msprof", "msdebug", "npu-smi", "ascend-dmi", "msobjdump", "llvm-objdump", "ccec", "bisheng"):
        path = shutil.which(name)
        if path:
            tools.add(str(Path(path).resolve()))
    for folder in (root / "tools", root / "bin", root / "compiler"):
        if folder.is_dir():
            for path in walk_files(folder, 5):
                if TOOL_PATTERN.fullmatch(path.name):
                    tools.add(str(path.resolve()))
    for path in sorted(tools)[:MAX_TOOLS]:
        print(f"TOOL {path}")
    print(f"TOOLS found={len(tools)} printed_at_most={MAX_TOOLS}; 仅发现路径，未验证功能或执行。")
    nm = shutil.which("nm")
    libraries = [
        root / "lib64/libascendcl.so",
        root / "runtime/lib64/libascendcl.so",
        root / "x86_64-linux/lib64/libascendcl.so",
        root / "aarch64-linux/lib64/libascendcl.so",
    ]
    library = next((path for path in libraries if path.is_file()), None)
    if nm and library:
        try:
            completed = subprocess.run(
                [nm, "-D", "--defined-only", str(library)], capture_output=True, text=True, timeout=15, check=False
            )
            print(f"EXPORTED SYMBOLS library={library.resolve()} nm_exit={completed.returncode}")
            names = {decl.split("(", 1)[0] for decl in ordered[:MAX_APIS]}
            for name in sorted(names):
                present = bool(re.search(rf"\b{re.escape(name)}(?:@[^\s]+)?$", completed.stdout, re.M))
                print(f"SYMBOL {name}: {'EXPORTED' if present else 'NOT_FOUND_IN_THIS_LIBRARY'}")
            if completed.returncode:
                print(completed.stderr[:500])
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"SYMBOL_CHECK_UNAVAILABLE {exc}")
    else:
        print("SYMBOL_CHECK_UNAVAILABLE: nm 或 libascendcl.so 未在候选位置找到")
    print("结束。请检查脱敏后返回控制台输出；本次没有重跑模型。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    args = parser.parse_args()
    inspect(args.cann_root)


if __name__ == "__main__":
    main()
