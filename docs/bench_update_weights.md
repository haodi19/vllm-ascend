# Measuring update_weights host overhead

Run from the repository in the green container. Output goes to the console;
the script writes no result files. Use `python -B` to suppress bytecode files.

For opt-in native submission and a breakdown of `load_submit`, see the
[phase-3 H2D experiment](lrc_cpp_h2d_experiment.md). These options are confined
to the standalone benchmark; the default remains TorchCopy.

For the current **literal Python-to-C++ migration**, use
[`--load-backend python|cpp`](load_submit_cpp_migration.md) with TorchCopy.
This is separate from the older ACL batching experiment.

```bash
python -B bench_update_weights.py
ASCEND_RT_VISIBLE_DEVICES=3 python -B bench_update_weights.py --mode npu
```

The first command measures CPU planning only: stream switching, transfer task
construction, submission and synchronization are replaced by no-ops. It is a
partial measurement, not the complete host overhead of a real transfer.

The second command uses real TorchCopy H2D with synthetic byte buffers, the
original manager's transfer-task construction and stream synchronization.
Synchronization is retained for correctness but its elapsed time is subtracted.
The device index defaults to zero after visibility filtering (`--device 0`).
No model, vLLM service, graph capture or host callback is launched.

For every call the script computes:

```text
outer = total - observe - choose_victims - synchronize
```

Percentiles are computed after subtraction, not subtracted from one another.
`load_submit` is part of `outer`, not an additional cost. Timers and wrappers add
some overhead, which remains in `outer`. This is elapsed host time, not CPU-only
execution time: submission can also contain runtime blocking.

## Inputs and scope

- Defaults align with `bench_lrc_policy.py`: `script/test_graph2.sh`, one token,
  seed 20261009, 1000 warmup calls and 10000 measured calls. Warmup is extended
  when necessary to fill the LRC history window.
- `--model-config /path/config.json` reads expert count and top-k only. Without
  it, fallback dimensions are 256 experts and top-k 6. CLI overrides include
  `--experts`, `--topk`, `--cache-size`, `--tokens`, `--recent-window` and
  `--[no-]router-scores`. Tokens is routed rows per call, not an MTP setting.
- `--scenario hit`, `mixed` (default uniform routes), or `miss` controls routing
  relative to a full synthetic cache. Residency is reset outside every timed
  call; LRC history evolves. Repeated calls therefore do not silently become
  all cache hits. A token's top-k IDs are distinct; tokens can share experts.
- `--prefetch` measures the prefetch branch. It skips observation and uses the
  script's `expert_prefetch_num` to limit loads.
- Debug logging is OFF by default even if the launch script enables it.
  `--debug` includes real console logging and will produce many lines. Decode
  statistics, expert substitution and pruning are disabled in this harness.
- NPU mode uses two 1 MiB buffers per expert by default. `--weight-mib` changes
  each buffer size. These are synthetic unquantized bytes, not real model
  shapes, quantization scales, NZ tensors or MemFabric transfers. A model
  config does not change buffer sizes. Memory estimates are printed before
  allocation. Run on an available device with sufficient memory.
- The script compiles selected methods directly from the repository's manager
  source and loads its LRC policy and TorchCopy transport without importing
  the vLLM plugin. Source hashes are printed. This avoids model initialization
  and does not modify production code.

To test four routed token rows with the same other settings:

```bash
ASCEND_RT_VISIBLE_DEVICES=3 python -B bench_update_weights.py --mode npu --tokens 4
```

Start with a short NPU smoke test if desired:

```bash
ASCEND_RT_VISIBLE_DEVICES=3 python -B bench_update_weights.py --mode npu --warmup 20 --iterations 100
```

Copy the configuration header, timing table and final validation line back
after checking for sensitive information. Do not send files. Results estimate
the specified synthetic workload; they do not measure graph replay scheduling,
callback entry waits or explain an entire gap between hardware memcpy events.

## Blue checks

```bash
python -B tests/ut/expert_offload/test_bench_update_weights.py
```

Tests cover per-call subtraction, absent observation, input validation, cache
state restoration across scenarios and the original transfer helpers on CPU.
Real NPU transport still requires validation in the green environment.
