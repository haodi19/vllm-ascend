"""检查设备占用及基础 NPU 操作；仅控制台输出，不运行模型或清理设备。"""

import argparse
import contextlib
import importlib.metadata
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import torch
import torch_npu
print("[1/6] import OK; torch=" + torch.__version__ + "; torch_npu=" + torch_npu.__version__, flush=True)
torch.npu.set_device(0)
print("[2/6] logical device 0 selected", flush=True)
with torch.device("npu:0"):
    scale = torch.tensor(1.0, dtype=torch.float32)
torch.npu.synchronize()
print("[3/6] scalar creation + synchronize OK", flush=True)
expected = torch.arange(1024, dtype=torch.float32)
x = expected.to("npu:0")
torch.npu.synchronize()
print("[4/6] H2D + synchronize OK", flush=True)
y = x + 1
torch.npu.synchronize()
print("[5/6] vector add + synchronize OK", flush=True)
actual = y.cpu()
assert scale.cpu().item() == 1.0
assert torch.equal(actual, expected + 1), "vector result mismatch"
print("[6/6] D2H + result check OK", flush=True)
"""


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return result


def run_probe(command, env, timeout):
    # Inherit stdout/stderr so the last completed stage survives a timeout.
    with subprocess.Popen(command, env=env, start_new_session=os.name == "posix") as child:
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT：基础测试超过 {timeout}s；只终止本脚本创建的测试进程。", flush=True)
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait()
            return 124


def status(command, keywords=None):
    print("\n$ " + " ".join(command), flush=True)
    if not shutil.which(command[0]):
        print("命令不可用；需要时到宿主机执行。", flush=True)
        return
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=15)
        lines = (result.stdout + result.stderr).splitlines()
        if keywords:
            lines = [line for line in lines if any(word in line for word in keywords)]
        print("\n".join(lines[:100]) or "没有匹配记录", flush=True)
        print(f"exit_code={result.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print("状态查询超过 15s，跳过。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True, help="与 ASCEND_RT_VISIBLE_DEVICES 使用同一编号")
    parser.add_argument("--timeout", type=positive, default=60)
    parser.add_argument("--status-only", action="store_true", help="只看状态，不提交 NPU 操作")
    parser.add_argument("--test-script", type=Path, default=Path(__file__).with_name("test_graph2.sh"))
    args = parser.parse_args()
    if args.npu < 0:
        parser.error("NPU 编号不能为负数")
    print(f"Python={sys.executable}\n测试设备映射：ASCEND_RT_VISIBLE_DEVICES={args.npu} → logical NPU 0", flush=True)
    for name in ("torch", "torch-npu"):
        try:
            print(f"{name}={importlib.metadata.version(name)}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}=not installed", flush=True)
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING"):
        print(f"原环境 {name}={os.environ.get(name, '<unset>')}", flush=True)
    if args.test_script.is_file():
        print(f"脚本设备配置（只读）：{args.test_script}", flush=True)
        for line in args.test_script.read_text(encoding="utf-8").splitlines():
            if "ASCEND_RT_VISIBLE_DEVICES" in line:
                print(line, flush=True)
    status(["npu-smi", "info"])
    status(["ps", "-eo", "pid,ppid,stat,etime,args"], ("EngineCore", "vllm", "host_sync", "profiler"))
    if args.status_only:
        return 0
    print("\n开始基础测试：不导入 vLLM、不启动 MoE/MTE、不重置卡、不删除日志。", flush=True)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    code = run_probe([sys.executable, "-B", "-u", "-c", PROBE], env, args.timeout)
    print(
        f"\n基础测试 exit_code={code}；{'通过（不代表完整模型已验证）' if code == 0 else '失败，请返回以上阶段及错误'}",
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
