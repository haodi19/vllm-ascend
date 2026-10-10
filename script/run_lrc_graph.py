#!/usr/bin/env python3
"""Run test_graph2.sh with an explicit LRC backend, without editing the file."""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def select_backend(source, backend):
    if backend not in ("python", "cpp", "cpp_fused"):
        raise ValueError("backend must be python, cpp or cpp_fused")

    def replace(match):
        config = json.loads(match.group(2))
        offload = config.setdefault("expert_offload_config", {})
        if not offload.get("cache_policy_enabled"):
            raise ValueError("Launch script must enable cache_policy_enabled")
        offload["cache_policy_backend"] = backend
        return match.group(1) + shlex.quote(json.dumps(config))

    updated, count = re.subn(r"(?m)^(\s*--additional-config\s+)'([^'\n]*)'", replace, source)
    if count != 1:
        raise ValueError("Expected one active, single-quoted --additional-config line")
    return updated if updated.endswith("\n") else updated + "\n"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp", "cpp_fused"), default="python")
    parser.add_argument("--test-script", type=Path, default=repo / "script/test_graph2.sh")
    args = parser.parse_args()
    try:
        source = select_backend(args.test_script.read_text(encoding="utf-8"), args.backend)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Launching {args.test_script.name}: cache_policy_backend={args.backend}", flush=True)
    # Execute the existing user's launch configuration, including its system tuning.
    # No temporary script or result file; the shell inherits stdout/stderr.
    raise SystemExit(subprocess.run(["bash"], input=source, text=True, cwd=repo).returncode)


if __name__ == "__main__":
    main()
