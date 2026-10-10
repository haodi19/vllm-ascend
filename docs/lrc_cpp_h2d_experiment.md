# Phase 3: measure and compare native H2D submission

This is a standalone **experiment**, not a new model-service backend. The
Python, cpp and cpp_fused policies and the original TorchCopy transport remain
available. No changes to `test_graph2.sh`, service configuration, model loading,
quantization or graph callbacks are made in this phase.

## Why this experiment

The supplied phase-2 run used identical synthetic dimensions, seed and average
load count (5.47 experts/call). Mean times in microseconds were:

| Metric | cpp | cpp_fused |
| --- | ---: | ---: |
| Planning host_without_sync | 13.184 | 7.416 |
| NPU host_without_sync | 85.535 | 81.584 |
| NPU load_submit | 57.899 | 60.140 |
| NPU synchronize | 213.982 | 212.210 |
| NPU total | 299.517 | 293.794 |

Planning improved by 43.75%, but NPU host time only by 4.62%. Submission plus
task preparation consumes about 74% of the fused host time. This makes it the
next useful target. The fused NPU p99 was worse in this run, so neither a tail
latency improvement nor a model-serving speedup is established.

## What changes

`bench_update_weights.py --h2d-backend acl` replaces the Python per-task
`destination.copy_(source)` loop with the repository's existing native operator:

```text
Python manager builds H2DCopyTask objects (unchanged)
    -> ExperimentalAclStorageTransport validates tasks/builds three CPU arrays
    -> torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 0)
    -> C++ in csrc/torch_binding.cpp
    -> aclrtMemcpyBatchAsync OR a C++ loop of aclrtMemcpyAsync
    -> original load-stream synchronization
```

This reuses existing C++; it does not introduce a new native kernel or move task
construction into C++. The installed binary's compile-time setting determines
which ACL API runs; choosing `acl` does not prove the batch API is enabled.
Validation and descriptor allocation are deliberately included in timings.
They can offset the benefit of fewer Python/native calls: faster performance
is a hypothesis to test, not an assumed result.

The experiment accepts only pinned CPU to local NPU **UntypedStorage** copies
on one bound stream, with non-overlapping destinations and valid byte ranges.
It rejects tensors/quantization attributes/remote sources, retains owners and
descriptors until synchronization, and does not silently retry partially
submitted batches. It is not registered in the production transport factory.

Native submission can include host runtime queue draining, so `copy_submit`
is elapsed host time, not pure Python execution. For example, the upstream
[NPUStream implementation](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUStream.cpp)
can drain queued host tasks when obtaining a raw stream. The container's
installed version is authoritative. Compare `total` and `synchronize` as well
as `host_without_sync`; time moving between submission and waiting is not
necessarily an overall gain.

## Green-container commands

Update the existing checkout; no LRC extension rebuild is needed for this change:

```bash
cd /home/l00992607/vllm-ascend
git pull --ff-only origin dev/moeoffload
```

First verify that the installed main Ascend extension exposes the operator:

```bash
python -B -c 'import torch, torch_npu; from vllm_ascend import vllm_ascend_C; print(torch.ops._C_ascend.swap_blocks_batch)'
```

Only if that extension/operator is missing, rebuild the **main** plugin using
the container's existing CANN/compiler environment (not `csrc/lrc_policy`):

```bash
COMPILE_CUSTOM_KERNELS=1 python -m pip install -v -e . --no-build-isolation --no-deps
```

Run correctness smoke checks before timings. The following uses physical NPU 4
and logical device 0, matching the supplied output. Each invocation must end
with `Validation passed`; stop on an error and return its console text.

```bash
for transport in torch acl; do
  ASCEND_RT_VISIBLE_DEVICES=4 python -B bench_update_weights.py --mode npu --backend cpp_fused --h2d-backend "$transport" --warmup 40 --iterations 100 --check-every 1 || break
done
```

`--check-every 1` checks both weight buffers after every call, outside the timed
interval. Different w13/w2 byte patterns detect accidentally swapped buffers.
Frequent checks perturb the device/CPU state; do not use this run for performance
claims. Default runs check the last call's copied bytes as before.

After correctness passes, compare without detailed instrumentation. Use the
same idle card, one process at a time, and alternate order to expose drift:

```bash
for transport in torch acl acl torch; do
  ASCEND_RT_VISIBLE_DEVICES=4 python -B bench_update_weights.py --mode npu --backend cpp_fused --h2d-backend "$transport" || break
done
```

Then obtain a more detailed breakdown, with extra timer overhead:

```bash
for transport in torch acl; do
  ASCEND_RT_VISIBLE_DEVICES=4 python -B bench_update_weights.py --mode npu --backend cpp_fused --h2d-backend "$transport" --detail-load || break
done
```

All earlier dimensions/config overrides still work. For extra coverage, repeat
the short correctness run with `--tokens 4`, `--prefetch`, or `--scenario hit`
separately. Do not change these between comparison backends.

## Reading the added metrics

| Metric | What is timed |
| --- | --- |
| `load_prepare` | Sum of `_build_expert_h2d_tasks` calls: source/destination storage slices and task construction |
| `copy_submit` | Transport `copy_batch`: Python copies OR validation, descriptors and native submission |
| `scale_refresh` | Sum of `_refresh_expert_fp32_scale` calls; synthetic unquantized mode has no actual scale copy |
| `load_other` | Remaining `load_submit` work, including list handling and timer-wrapper overhead |
| `acl_descriptors` | Validation and CPU descriptor construction inside ACL `copy_submit` |
| `acl_native` | Native operator invocation inside ACL `copy_submit`; includes dispatch/runtime overhead |

Per invocation:

```text
load_submit = load_prepare + copy_submit + scale_refresh + load_other
outer       = total - observe - choose_victims - synchronize
```

`load_submit` is already part of `outer`. The two `acl_*` metrics are already
part of `copy_submit`. Do not add either twice. Percentiles of separate metrics
do not obey the above sum; the equations apply per call and to means.
Do not compare detailed runs directly with the previous less-instrumented run.

Return only reviewed console headers, validation results and timing tables.
The benchmark writes no result files. Real NPU transfers, installed native
binary compatibility, and model/graph accuracy cannot be validated on blue.

## Blue checks

```bash
python -B tests/ut/expert_offload/test_bench_update_weights.py
python -B tests/ut/expert_offload/test_experimental_acl_h2d.py
```

The second test injects a CPU copy function and fake NPU storage. It covers
descriptor byte correctness, stream checks, validation, lifetime and partial
submission errors; it does not execute CANN or demonstrate NPU speedups.
