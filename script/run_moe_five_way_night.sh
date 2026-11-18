#!/usr/bin/env bash
# Run inside the configured container. No installation or source modifications.
set -euo pipefail

work_root=/data/w00919836/moeoffload
npu=1
port=8156
timing_port=18156
output_root=
worker=0
while (($#)); do
  case "$1" in
    --work-root) work_root=$2; shift 2 ;;
    --npu) npu=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --timing-port) timing_port=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --worker) worker=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
for value in "$npu" "$port" "$timing_port"; do
  [[ $value =~ ^[0-9]+$ ]] || { echo "NPU and ports must be integers" >&2; exit 2; }
done
[[ $port != "$timing_port" ]] || { echo "Service and timing ports must differ" >&2; exit 2; }
work_root=$(cd "$work_root" && pwd -P)
repo="$work_root/vllm-ascend"
[[ -f "$work_root/env.sh" && -f "$repo/script/run_moe_callback_worker_compare.py" ]] || {
  echo "Missing env.sh or project in $work_root" >&2; exit 2;
}
script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")

if (( ! worker )); then
  output_root=${output_root:-"$work_root/bench_results/night_five_way_$(date +%Y%m%d_%H%M%S)_$$"}
  # Refuse reuse: never overwrite an earlier experiment.
  mkdir -p "$(dirname "$output_root")"
  mkdir "$output_root"
  output_root=$(cd "$output_root" && pwd -P)
  nohup bash "$script_path" --worker --work-root "$work_root" --npu "$npu" \
    --port "$port" --timing-port "$timing_port" --output-root "$output_root" \
    >"$output_root/run.log" 2>&1 </dev/null &
  bg_pid=$!
  printf '%s\n' "$bg_pid" >"$output_root/background.pid"
  printf 'PID=%s\nRESULTS=%s\n' "$bg_pid" "$output_root"
  printf 'Live output: tail -f %q\n' "$output_root/run.log"
  printf 'Combined results: %s/summary.txt\n' "$output_root"
  exit 0
fi

[[ -n $output_root && -d $output_root ]] || { echo "Missing output directory" >&2; exit 2; }
finish() {
  exit_code=$?
  trap - EXIT
  {
    printf 'BATCH_EXIT=%s; physical_npu=%s; port=%s; timing_port=%s\n' "$exit_code" "$npu" "$port" "$timing_port"
    [[ ! -f "$output_root/status.txt" ]] || cat "$output_root/status.txt"
    for tokens in 5 128; do
      printf '\n===== max_tokens=%s =====\n' "$tokens"
      if [[ -f "$output_root/token${tokens}/summary.txt" ]]; then
        cat "$output_root/token${tokens}/summary.txt"
      else
        printf 'No summary: check run.log; this run did not produce complete results.\n'
      fi
    done
  } >"$output_root/summary.txt"
  printf 'BATCH_EXIT=%s; SUMMARY=%s/summary.txt\n' "$exit_code" "$output_root"
  exit "$exit_code"
}
trap finish EXIT
cd "$work_root"
# Vendor environment scripts may read optional unset variables.
set +u
source "$work_root/env.sh"
set -u
cd "$repo"
printf 'COMMIT=%s\n' "$(git rev-parse HEAD)"
printf 'PHYSICAL_NPU=%s; work_root=%s\n' "$npu" "$work_root"
python -c 'import sys; print("PYTHON=" + sys.executable)'
git diff --stat
overall=0
for tokens in 5 128; do
  extra=()
  if (( tokens == 128 )); then extra+=(--ignore-eos); fi
  printf '\nSTART max_tokens=%s; five independent services\n' "$tokens"
  if python -u -B script/run_moe_callback_worker_compare.py \
    --include-legacy-callback --include-python-callback --measure-tpot \
    --npu "$npu" --port "$port" --timing-port "$timing_port" \
    --max-tokens "$tokens" "${extra[@]}" \
    --warmup-requests 5 --timing-requests 20 --requests 1 \
    --startup-timeout 1800 --request-timeout 600 --profile-timeout 600 \
    --bubble-stream-name 'Stream 56' --output-dir "$output_root/token${tokens}"; then
    code=0
  else
    code=$?
    overall=1
  fi
  printf 'max_tokens=%s; exit=%s\n' "$tokens" "$code" | tee -a "$output_root/status.txt"
done
exit "$overall"
