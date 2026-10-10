# Literal Python to C++ load_submit migration

The goal is implementation migration and measurement, **not optimization**.
This supersedes the proposed prepared-address experiment, which was withdrawn
before commit. The older `--h2d-backend acl` experiment remains available only
for historical comparison; it is not a literal translation and is not enabled
by the migration option.

## Current scope

`--load-backend cpp` selects four C++ translations in the standalone benchmark:

| Python reference | C++ method in `csrc/lrc_policy/load_submit.h` |
| --- | --- |
| `_load_expert_weights_into_slots` | `load` |
| `_build_expert_h2d_tasks` | `build_tasks` |
| `_refresh_expert_fp32_scale` | `refresh_scale` |
| `TorchCopyH2DTransport.copy_batch` | `copy_batch` |

The translations retain list materialization, expert iteration order, storage
slices, H2DCopyTask construction and names, quant-task extension, one `copy_`
per task with its original non_blocking flag, and the subsequent scale refresh
order. Empty batches and exceptions retain their reference behavior. GIL is
held. There is no descriptor caching, batching API change, skipped validation,
extra synchronization or asynchronous execution redesign.

This is C++ control flow operating on Python objects through pybind11. It still
calls the same Python Tensor/Storage methods; it is not a switch to the ATen or
ACL C++ copy APIs. This boundary deliberately preserves the current behavior,
and may carry binding overhead. It does not imply a speedup.

`_expert_weight`, `_expert_src_storage`, `_build_quant_attr_h2d_tasks`, transport
selection and synchronization remain the original Python dependencies. The
rest of `_update_weights` is unchanged. The standalone benchmark still uses
synthetic unquantized buffers; CPU parity tests additionally exercise packed
attribute resolution, quant tasks and scale conversion.

Production factories and `test_graph2.sh` are unchanged. `--load-backend cpp`
requires `--mode npu --h2d-backend torch`. Both sides must use the same LRC
backend. Use `--backend cpp` below to avoid including the earlier fused planner
in this comparison. The implementation has no dependence on that policy choice.

## Green execution

Rebuild the small CPU extension (API 3); no main Ascend plugin rebuild is needed:

```bash
cd /home/l00992607/vllm-ascend
git pull --ff-only origin dev/moeoffload
python -m pip install -v ./csrc/lrc_policy --no-build-isolation --no-deps
python -B tests/ut/expert_offload/test_load_submit_cpp.py
```

First check copied contents each call. Stop on any failure:

```bash
for implementation in python cpp; do
  ASCEND_RT_VISIBLE_DEVICES=4 python -B bench_update_weights.py --mode npu --backend cpp --h2d-backend torch --load-backend "$implementation" --warmup 40 --iterations 100 --check-every 1 || break
done
```

After both pass, compare performance with identical inputs and alternate order:

```bash
for implementation in python cpp cpp python; do
  ASCEND_RT_VISIBLE_DEVICES=4 python -B bench_update_weights.py --mode npu --backend cpp --h2d-backend torch --load-backend "$implementation" || break
done
```

Finally, collect the same detail timers on each side:

```bash
for implementation in python cpp; do
  ASCEND_RT_VISIBLE_DEVICES=4 python -B bench_update_weights.py --mode npu --backend cpp --h2d-backend torch --load-backend "$implementation" --detail-load || break
done
```

Compare `load_submit`, `host_without_sync`, `outer`, and `total` in runs without
detail timers. The detailed run explains `load_prepare`, `copy_submit` and
`scale_refresh`; do not compare its absolute time directly with a run lacking
the extra instrumentation. LRC behavior and synthetic load counts should match.
Report a slowdown as a slowdown; performance tuning is outside the current scope.

`python` always retains the original functions. Source edits or uninstalling the
extension are not needed to switch back. Console output only; return reviewed
headers, timing tables and validation/test summaries, not result files.

## Interpretation of the preceding ACL experiment

The two ordinary runs per transport (10000 samples each) gave these ranges of
mean times in microseconds:

| Metric | Torch | Experimental ACL |
| --- | ---: | ---: |
| total | 286.074–301.589 | 350.491–359.154 |
| host_without_sync | 72.315–79.046 | 153.606–165.059 |
| load_submit | 52.131–56.256 | 129.797–140.496 |

All provided byte checks passed, but this alternative was slower. The detailed
ACL run spent 53.005 us in descriptor validation/construction and 38.282 us in
the native call, versus 30.921 us for Torch copy submission in its detailed run.
This supports rejecting that implementation as a demonstrated speedup. It does
not measure the cost of a literal translation and does not establish which
compile-time ACL batch API was active. No model-service speedup was established.
