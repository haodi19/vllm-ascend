#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run inside the configured container with the project Python environment active.
set -euo pipefail

case_name=all
npu=6
port=8156
timing_port=18156
output_root=
worker=0
tokens=(5 128)
while (($#)); do
  case "$1" in
    --case) case_name=${2:?Missing case}; shift 2 ;;
    --npu) npu=${2:?Missing NPU}; shift 2 ;;
    --port) port=${2:?Missing port}; shift 2 ;;
    --timing-port) timing_port=${2:?Missing timing port}; shift 2 ;;
    --output-root) output_root=${2:?Missing output root}; shift 2 ;;
    --tokens) tokens=("${2:?Missing token count}"); shift 2 ;;
    --worker) worker=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$case_name" in all|both|pipeline|python|native|native_full|native_pipeline) ;; *) echo "Invalid case: $case_name" >&2; exit 2 ;; esac
for token_count in "${tokens[@]}"; do
  case "$token_count" in 5|128) ;; *) echo "--tokens must be 5 or 128" >&2; exit 2 ;; esac
done
for value in "$npu" "$port" "$timing_port"; do
  [[ $value =~ ^[0-9]+$ ]] || { echo "NPU and ports must be non-negative integers" >&2; exit 2; }
done
(( port >= 1 && port <= 65535 && timing_port >= 1 && timing_port <= 65535 && port != timing_port )) || {
  echo "Use two different ports in [1, 65535]" >&2; exit 2;
}
[[ -n $output_root ]] || { echo "--output-root is required; use a fresh _01, _02 directory" >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
script_path="$repo/script/run_native_moe_suite.sh"
runner="$repo/script/run_native_moe_compare.py"
[[ -f $runner ]] || { echo "Missing runner: $runner" >&2; exit 2; }
command -v python >/dev/null || { echo "Activate the project env.sh first" >&2; exit 2; }

if (( ! worker )); then
  mkdir -p "$output_root"
  output_root=$(cd "$output_root" && pwd -P)
  mkdir "$output_root/.suite.lock" 2>/dev/null || {
    echo "This output root is locked by another suite: $output_root/.suite.lock" >&2
    exit 2
  }
  trap 'rmdir "$output_root/.suite.lock"' EXIT
  # Check both token directories before starting any work. Failed cases are
  # retained too; rerunning a case requires a new experiment root.
  for token_count in "${tokens[@]}"; do
    case "$case_name" in
      all|both|pipeline)
        # Combined comparisons require a fresh token directory, not a fresh
        # root: --tokens 128 may follow --tokens 5 in the same experiment.
        token_dir="$output_root/token${token_count}"
        if [[ -e $token_dir || -L $token_dir ]]; then
          echo "Token directory already exists; no files overwritten: $token_dir" >&2
          exit 2
        fi
        continue
        ;;
    esac
    for previous in "$output_root/token${token_count}/$case_name" \
      "$output_root/token${token_count}/profile/$case_name" \
      "$output_root/token${token_count}/${case_name}_command.json"; do
      if [[ -e $previous ]]; then
        echo "Case already exists; no files overwritten: $previous" >&2
        exit 2
      fi
    done
  done
  worker_args=(--worker --case "$case_name" --npu "$npu" --port "$port"
    --timing-port "$timing_port" --output-root "$output_root")
  if (( ${#tokens[@]} == 1 )); then worker_args+=(--tokens "${tokens[0]}"); fi
  nohup bash "$script_path" "${worker_args[@]}" >>"$output_root/run.log" 2>&1 </dev/null &
  bg_pid=$!
  printf '%s\n' "$bg_pid" >"$output_root/background.pid"
  printf 'PID=%s\nCASES=%s; tokens=%s\nRESULTS=%s\n' "$bg_pid" "$case_name" "${tokens[*]}" "$output_root"
  printf 'Live output: tail -f %q\n' "$output_root/run.log"
  printf 'Return this file: %s/summary.txt\n' "$output_root"
  trap - EXIT  # The background worker owns the lock until its EXIT handler.
  exit 0
fi

[[ -d $output_root ]] || { echo "Worker output directory is missing" >&2; exit 2; }
[[ -d "$output_root/.suite.lock" ]] || { echo "Worker requires the launcher's output lock" >&2; exit 2; }
finish() {
  code=$?
  trap - EXIT
  batch_code=$code
  if [[ -f "$output_root/status.txt" ]] && grep -Eq 'exit=[1-9][0-9]*' "$output_root/status.txt"; then
    batch_code=1
  fi
  if ! python -u -B "$repo/script/summarize_native_moe_results.py" "$output_root"; then
    printf 'SUMMARY_FAILED; saved artifacts and run.log retained\n'
    code=1
    batch_code=1
  fi
  printf 'BATCH_EXIT=%s; LAST_JOB_EXIT=%s; SUMMARY=%s/summary.txt\n' "$batch_code" "$code" "$output_root"
  rmdir "$output_root/.suite.lock"
  exit "$code"
}
trap finish EXIT
cd "$repo"
printf 'COMMIT=%s; case=%s; physical_npu=%s\n' "$(git rev-parse HEAD)" "$case_name" "$npu"
python -c 'import sys; print("PYTHON=" + sys.executable)'
overall=0
for token_count in "${tokens[@]}"; do
  printf '\nSTART max_tokens=%s; case=%s\n' "$token_count" "$case_name"
  if python -u -B "$runner" --case "$case_name" --npu "$npu" \
    --port "$port" --timing-port "$timing_port" --max-tokens "$token_count" \
    --warmup-requests 5 --requests 20 --startup-timeout 1800 \
    --request-timeout 600 --profile-timeout 600 --min-profile-free-gib 5 \
    --bubble-stream-name auto --output-dir "$output_root/token${token_count}"; then
    code=0
  else
    code=$?
    overall=1
  fi
  printf 'max_tokens=%s; exit=%s; case=%s; physical_npu=%s\n' \
    "$token_count" "$code" "$case_name" "$npu" | tee -a "$output_root/status.txt"
done
exit "$overall"
