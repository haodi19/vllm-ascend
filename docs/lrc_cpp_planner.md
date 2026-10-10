# Fused single-card C++ cache planner (phase 2)

Three explicit backends are available:

| Backend | LRC | Hit/miss selection and slot planning |
| --- | --- | --- |
| `python` (default) | Original Python | Original Python |
| `cpp` | Phase-1 C++ | Original Python |
| `cpp_fused` | C++ | Same C++ call as LRC |

The original launch script is unchanged. Transfers, synchronization, callback
registration, debug logging and decode statistics remain in their original
paths. Multi-card configuration rejects `cpp_fused`; use `python` or `cpp` there.

## Build and test in the green container

The optional CPU extension must be rebuilt: this phase adds API 2 and package
version 0.2.0. No rebuild of the main vLLM Ascend NPU operators is needed.

```bash
cd /home/l00992607/vllm-ascend
git pull --ff-only origin dev/moeoffload
python -m pip install -v ./csrc/lrc_policy --no-build-isolation --no-deps
python -B tests/ut/expert_offload/test_lrc_cpp.py --require-cpp
python -B tests/ut/expert_offload/test_lrc_cpp_plan.py --require-cpp
```

Existing phase-1 build dependencies (pybind11, wheel, setuptools, Python headers
and a C++17 compiler) are sufficient. Both commands must report OK without
skipping native tests. The new tests exercise the actual manager method with
only transfers/stream operations replaced by CPU test doubles.

## Compare latency

```bash
python -B bench_update_weights.py --backend cpp
python -B bench_update_weights.py --backend cpp_fused
```

These default to CPU planning only. For synthetic NPU transfers, use the same
available device and the same arguments for both runs:

```bash
ASCEND_RT_VISIBLE_DEVICES=3 python -B bench_update_weights.py --mode npu --backend cpp
ASCEND_RT_VISIBLE_DEVICES=3 python -B bench_update_weights.py --mode npu --backend cpp_fused
```

Add `--tokens 4` to both commands for four routed rows. Add `--prefetch` to both
to measure prefetch planning. Model dimensions, debug configuration, warmup and
iteration counts must match. The standalone NPU mode still uses synthetic
TorchCopy buffers rather than the full model's quantization layout.

The main comparison is **host_without_sync = total - synchronize**. This keeps
the same whole-call boundary across backends. `outer` still subtracts LRC, but
fused LRC is timed inside C++, whereas the old paths measure its Python-facing
calls including argument conversion. Consequently the two `outer` boundaries
are not identical. Do not attribute the whole difference to cache planning.

`fused_plan` includes LRC, buffer access, hit/miss selection, victim selection,
mapping changes and returned metadata. It is already inside `total`; never add
it to `outer` or `total`. The benchmark enables internal observation/selection
timers; production calls leave these timers disabled.

## Run the model and switch back

```bash
python -B script/run_lrc_graph.py --backend cpp_fused
```

The launcher runs the existing `test_graph2.sh` settings, including device
selection, system tuning and profiling. Stop that service before running the
same port with either baseline:

```bash
python -B script/run_lrc_graph.py --backend cpp
python -B script/run_lrc_graph.py --backend python
```

Run only one of these service commands at a time. No source edit or uninstall
is needed to switch back. The direct config key is
`expert_offload_config.cache_policy_backend = "cpp_fused"`.

## Implementation and parity limits

`plan_update` obtains NumPy views of already-staged CPU tensors and enters the
native planner once. There is no extra device-to-host transfer or `.tolist()`.
The native planner reads int32/int64 routing, optional float32/float64 scores,
and the writable int32/int64 mapping. It returns the loads and pre-eviction
sets needed by existing Python diagnostics. The original mapping is updated
in place, so its later device writeback is preserved.

For exact assignment parity with the Python reference, a few small sets use
the CPython set API inside the native call. This preserves hash-collision and
iteration behavior; the implementation does not claim to eliminate every
Python object. Arithmetic, candidate ranking and per-expert planning loops
run in C++. The GIL remains held, as in phase 1.

Differential tests check exact Python-reference load order, slots, mapping,
LRC history, counters, shortfalls and prefetch statistics. Phase-1 `cpp` converts
an unordered C++ set into Python, so its physical assignment order can differ;
tests compare its resident/loaded expert sets and statistics rather than require
identical slot numbering. Both old backends retain their implementation paths.

The fused interface validates buffers before changing state. It rejects
read-only/non-contiguous mappings, duplicate physical slots, incompatible
dtypes/shapes and invalid prefetch IDs. Reactive invalid IDs are filtered as
in the reference. Routing and score views may be strided. Empty/unfilled caches
retain the reference's shortfall behavior rather than inventing new free slots.

Return only reviewed console configuration headers, test summaries and timing
tables. These tests/benchmarks create no result files. Full service profiling
continues to create the normal profile artifacts configured by the launch script.
