# SPDX-License-Identifier: Apache-2.0
"""Exercise the batch shell with a fake runner, never importing Torch/NPU."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class NativeSuiteTests(unittest.TestCase):
    def setUp(self):
        self.bash = shutil.which("bash")
        if self.bash is None:
            self.skipTest("bash required")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "script").mkdir()
        (self.root / "bin").mkdir()
        shutil.copyfile(ROOT / "script/run_native_moe_suite.sh", self.root / "script/run_native_moe_suite.sh")
        for path in (ROOT / "script").glob("*.py"):
            shutil.copyfile(path, self.root / "script" / path.name)
        self.output = self.root / "results_01"
        fake = self.root / "bin/python"
        fake.write_text(
            """#!/usr/bin/env bash
set -eu
if [[ $1 == -c ]]; then echo PYTHON=fixture; exit 0; fi
if [[ ${3:-} == */summarize_native_moe_results.py ]]; then exec "ACTUAL_PYTHON" "$@"; fi
case_name=; tokens=; out=; npu=
while (($#)); do
  case "$1" in
    --case) case_name=$2; shift 2 ;;
    --max-tokens) tokens=$2; shift 2 ;;
    --output-dir) out=$2; shift 2 ;;
    --npu) npu=$2; shift 2 ;;
    *) shift ;;
  esac
done
case "$case_name" in
  all) mkdir "$out"; cases=(python native native_full) ;;
  both) mkdir "$out"; cases=(python native) ;;
  *) mkdir -p "$out"; cases=("$case_name") ;;
esac
for name in "${cases[@]}"; do
  mkdir "$out/$name"
  mkdir -p "$out/profile/$name"
  cat > "$out/$name/timing_result.json" <<EOF
{"case":"$name","requests":1,"output_match":true,"stream_samples":[
{"digest":"same","usage":{"completion_tokens":$tokens},"completion_tokens":$tokens,
"content_chunks":3,"decode_us":0,"ttft_us":3000,"output_latency_us":3000,"tpot_us":0,"http_wall_us":3100}]}
EOF
  echo "trace $name $tokens" > "$out/profile/$name/trace.json"
done
printf 'case=%s tokens=%s npu=%s\n' "$case_name" "$tokens" "$npu" >> calls.txt
echo "FAKE_RESULT case=$case_name tokens=$tokens" >> "$out/summary.txt"
echo "LIVE tokens=$tokens"
if [[ -f fail_token$tokens ]]; then exit 3; fi
""".replace("ACTUAL_PYTHON", Path(sys.executable).as_posix()),
            encoding="utf-8",
            newline="\n",
        )
        fake.chmod(0o755)
        git = self.root / "bin/git"
        git.write_text("#!/usr/bin/env bash\necho fixture-commit\n", encoding="utf-8", newline="\n")
        git.chmod(0o755)

    def invoke(self, *args):
        # Positional arguments preserve spaces; fake python applies only here.
        if "--worker" in args:
            (self.output / ".suite.lock").mkdir()
        return subprocess.run(
            [
                self.bash,
                "-c",
                'cd "$1" || exit; export PATH="$(pwd -P)/bin:$PATH"; '
                'exec bash "$1/script/run_native_moe_suite.sh" "${@:2}"',
                "suite-test",
                self.root.as_posix(),
                "--output-root",
                self.output.as_posix(),
                *args,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )

    def calls(self):
        return (self.root / "calls.txt").read_text(encoding="utf-8").splitlines()

    def wait_background(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (self.output / ".suite.lock").exists():
            time.sleep(0.05)
        self.assertFalse((self.output / ".suite.lock").exists())
        self.assertTrue((self.output / "summary.txt").exists())

    def test_separate_cases_share_both_token_directories_and_keep_previous_artifacts(self):
        saved = {}
        for case in ("python", "native", "native_full"):
            result = self.invoke("--case", case)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.wait_background()
            for path, content in saved.items():
                self.assertEqual(path.read_bytes(), content)
            for tokens in (5, 128):
                paths = (
                    self.output / f"token{tokens}/{case}/timing_result.json",
                    self.output / f"token{tokens}/profile/{case}/trace.json",
                )
                saved.update({path: path.read_bytes() for path in paths})
        summary = (self.output / "summary.txt").read_text(encoding="utf-8")
        for case in ("python", "native", "native_full"):
            for tokens in (5, 128):
                self.assertIn(f"| {case} |", summary)
                self.assertIn(f"[{tokens}]", summary)
        self.assertNotIn("FAKE_RESULT", summary)
        self.assertEqual(len(self.calls()), 6)
        self.assertIn("BATCH_EXIT=0", summary)
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertNotEqual(self.invoke("--case", "python").returncode, 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()})

    def test_locked_root_refuses_another_job_before_writing(self):
        self.output.mkdir()
        (self.output / ".suite.lock").mkdir()
        (self.output / "run.log").write_text("running")
        result = self.invoke("--case", "native")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("locked", result.stderr)
        self.assertEqual((self.output / "run.log").read_text(), "running")
        self.assertFalse((self.root / "calls.txt").exists())

    def test_later_token_job_preserves_earlier_summary_and_failure_status(self):
        (self.root / "fail_token5").touch()
        self.assertEqual(self.invoke("--case", "python", "--tokens", "5").returncode, 0)
        self.wait_background()
        self.assertEqual(self.invoke("--case", "python", "--tokens", "128").returncode, 0)
        self.wait_background()
        summary = (self.output / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("[5]", summary)
        self.assertIn("[128]", summary)
        self.assertIn("BATCH_EXIT=1", summary)

    def test_all_cases_run_both_token_counts_serially_and_combine_results(self):
        self.output.mkdir()
        result = self.invoke("--worker", "--case", "all", "--npu", "7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), ["case=all tokens=5 npu=7", "case=all tokens=128 npu=7"])
        summary = (self.output / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("BATCH_EXIT=0", summary)
        self.assertIn("[5]", summary)
        self.assertIn("[128]", summary)

    def test_single_case_and_token_selection_do_not_run_other_cases(self):
        for case in ("python", "native", "native_full"):
            with self.subTest(case=case):
                self.output = self.root / (case + "_01")
                self.output.mkdir()
                result = self.invoke("--worker", "--case", case, "--tokens", "128")
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.calls(), [f"case={case} tokens=128 npu=6" for case in ("python", "native", "native_full")]
        )

    def test_failure_remains_explicit_and_later_token_count_is_still_attempted(self):
        self.output.mkdir()
        (self.root / "fail_token5").touch()
        result = self.invoke("--worker")
        self.assertEqual(result.returncode, 1)
        summary = (self.output / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("BATCH_EXIT=1", summary)
        self.assertIn("max_tokens=5; exit=3", summary)
        self.assertIn("max_tokens=128; exit=0", summary)
        self.assertEqual(len(self.calls()), 2)

    def test_background_launch_produces_live_log_pid_and_refuses_reuse(self):
        result = self.invoke("--case", "native_full", "--tokens", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Live output:", result.stdout)
        self.wait_background()
        self.assertTrue((self.output / "background.pid").read_text().strip().isdigit())
        self.assertIn("LIVE tokens=5", (self.output / "run.log").read_text(encoding="utf-8"))
        again = self.invoke()
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(self.calls(), ["case=native_full tokens=5 npu=6"])

    def test_invalid_parameters_fail_before_creating_results(self):
        for args in (("--case", "unknown"), ("--tokens", "32"), ("--npu", "-1"), ("--port", "18156")):
            with self.subTest(args=args):
                self.assertNotEqual(self.invoke(*args).returncode, 0)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
