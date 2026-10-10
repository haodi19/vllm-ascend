# Optional C++ LRC backend (phase 1)

The Python backend remains the default and its implementation is unchanged.
This phase moves LRC observation, scoring, eviction and state into a CPU C++
extension. It does not change transfer submission, stream synchronization,
host callback registration or the outer cache-planning loop.

## Build and validate in the green container

The extension is independent of the main vLLM Ascend build. It needs Python
development headers, a C++17 compiler, setuptools, wheel and pybind11, but does
not link Torch or CANN. Installing it does not select it automatically.

```bash
cd /home/l00992607/vllm-ascend
git pull --ff-only origin dev/moeoffload
python -m pip install 'pybind11>=2.11' wheel
python -m pip install -v ./csrc/lrc_policy --no-build-isolation --no-deps
python -B tests/ut/expert_offload/test_lrc_cpp.py --require-cpp
```

`--require-cpp` fails if the extension is absent. Without it, native tests are
explicitly skipped, allowing default-Python checks on machines without a
compiler. Source changes to C++ require rebuilding the extension.

## Compare latency

```bash
python -B bench_lrc_policy.py --backend python
python -B bench_lrc_policy.py --backend cpp
python -B bench_update_weights.py --backend python
python -B bench_update_weights.py --backend cpp
```

Use the same `--tokens`, model dimensions, cache settings, warmup and iterations
for both backends. `--model-config /path/config.json` and `--tokens 4` remain
available. Whole-call comparison includes Python/C++ argument conversion.
The default outer benchmark is CPU planning only. To include synthetic NPU
TorchCopy submission, add `--mode npu` to both runs on the same available card.

For this phase expect potential improvements in `observe`, `choose_victims`
and `total`. The `outer` metric subtracts both LRC functions, so it need not
improve when only LRC is accelerated. No speedup is promised before measurement.

## Enable in a service and switch back

The configuration key is:

```json
{"expert_offload_config": {"cache_policy_backend": "cpp"}}
```

Merge that key into the existing additional configuration, keeping all other
options. Values are `python` (default) and `cpp`. Missing native binaries produce
an actionable error when `cpp` is explicitly selected; no silent fallback.

To run the existing `script/test_graph2.sh` configuration without editing it:

```bash
python -B script/run_lrc_graph.py --backend cpp
```

Stop that service before starting the comparison on the same port:

```bash
python -B script/run_lrc_graph.py --backend python
```

The launcher only changes the LRC backend in memory and pipes the shell script
to Bash. It executes the existing script's system tuning, model, device and
profiling settings. It creates no temporary file. Running the original
`bash script/test_graph2.sh` also retains its original default-Python behavior.
The profiler configured by that service still creates its normal profile files;
the standalone benchmarks and parity tests only print to the console.

## Compatibility and limits

- The Python class is retained unchanged; selection uses `create_lrc_policy`.
  Existing direct imports of `LRCExpertCachePolicy` still select Python.
- State is owned by each native policy instance. The binding keeps the GIL;
  this phase does not attempt concurrent mutation or native callback dispatch.
- The native adapter accepts the same ordinary numeric list/array inputs used
  by the manager. Python/STL conversion is included in benchmark times.
- Recent history advances per token row. Duplicate routes increment frequency
  multiple times but contribute only one EMA hit per row. Invalid IDs are
  filtered without shifting their router-score positions.
- Eviction uses `(hotness, last_used, expert_id)` ordering and protects required
  experts. Hot preload seeding retains Python's rounding behavior.
- `layer_states` on the C++ adapter is a diagnostic snapshot. Reading fields
  works for existing logging; editing a snapshot does not modify native state.
  Update native state through policy methods. The Python backend's live state
  behavior is unchanged. Policy hyperparameters are fixed at construction for
  the native backend; rebuild the instance to change them.
- EMA is float32. Scalar hotness accounts for NumPy 1/2 promotion differences;
  compiler floating-point contraction is disabled. Run differential tests in
  the actual green NumPy environment before trusting eviction equivalence.
- Tests include global counts, but multi-card service execution, graph replay
  and full-model correctness/performance still require green validation.

Return only reviewed console summaries: parity test results, configuration
headers and latency tables. No files need to be sent back.
