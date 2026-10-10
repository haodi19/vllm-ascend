"""Opt-in storage-only H2D experiment for bench_update_weights.py.

Not registered as a model-service transport: tensor/quantized attribute copies
and graph host callbacks need separate validation. The existing native operator
uses aclrtMemcpyBatchAsync when built with support, otherwise a C++ loop of
aclrtMemcpyAsync. This module does not imply the installed binary uses batching.
"""

import torch

DIRECTION_H2D = 0
INT64_MAX = (1 << 63) - 1


class ExperimentalAclStorageTransport:
    """Submit byte-preserving storage copies on one caller-selected stream.

    Calls must be serialized, inside the bound stream's context. Keep owners
    and descriptors alive until successful synchronization, including after a
    partial submission failure. No silent fallback or resubmission is allowed.
    """

    def __init__(self, stream, device_id, *, copy_op=None, current_stream=None):
        if copy_op is None:
            # Lazy loading leaves the default benchmark independent of this
            # optional compiled module and avoids loading general plugins.
            try:
                from vllm_ascend import vllm_ascend_C  # noqa: F401

                copy_op = torch.ops._C_ascend.swap_blocks_batch
            except (ImportError, AttributeError, OSError) as exc:
                raise RuntimeError(
                    "ACL experiment requires vllm_ascend.vllm_ascend_C with "
                    "swap_blocks_batch. Rebuild vllm-ascend in the NPU container; "
                    "the standalone LRC extension does not provide this operator."
                ) from exc
        if current_stream is None:
            import torch_npu

            current_stream = lambda: torch_npu.npu.current_stream(device_id)
        self.stream = stream
        self.device_id = device_id
        self._copy_op = copy_op
        self._current_stream = current_stream
        self._inflight = []
        self._closed = False
        self._failed = False

    def _build_descriptors(self, tasks):
        sources, destinations, sizes, ranges = [], [], [], []
        for task in tasks:
            src, dst = task.source, task.destination
            if not isinstance(src, torch.UntypedStorage) or not isinstance(dst, torch.UntypedStorage):
                raise TypeError("ACL experiment accepts only UntypedStorage tasks, not tensors or remote pointers")
            if not task.non_blocking:
                raise ValueError("ACL experiment requires non_blocking=True")
            # Storage.is_pinned defaults to CUDA even in PyTorch 2.10; select
            # NPU explicitly. These validation costs remain in the measurement.
            if src.device.type != "cpu" or not src.is_pinned(device="npu"):
                raise ValueError("ACL source must be pinned CPU storage")
            if dst.device.type != "npu" or dst.device.index != self.device_id:
                raise ValueError("ACL destination must be on the configured NPU")
            size = task.nbytes
            if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= min(src.nbytes(), dst.nbytes()):
                raise ValueError("ACL byte count must fit both storages")
            source, destination = src.data_ptr(), dst.data_ptr()
            if not 0 < source <= INT64_MAX - size or not 0 < destination <= INT64_MAX - size:
                raise ValueError("ACL pointer range cannot be represented in int64")
            sources.append(source)
            destinations.append(destination)
            sizes.append(size)
            ranges.append((destination, destination + size))
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("ACL batch destinations must not overlap")
        # Three CPU arrays, never device tensors. Native code requires flat,
        # contiguous int64 arrays; constructing them here enforces that contract.
        return tuple(torch.tensor(values, dtype=torch.int64, device="cpu") for values in (sources, destinations, sizes))

    def _submit(self, descriptors):
        self._copy_op(*descriptors, DIRECTION_H2D)

    def copy_batch(self, tasks):
        if self._closed or self._failed:
            raise RuntimeError("ACL transport is closed or needs synchronization after a failed submission")
        if not tasks:
            return
        if self._current_stream().npu_stream != self.stream.npu_stream:
            raise RuntimeError("ACL copy must run inside the bound load stream context")
        tasks = tuple(tasks)
        descriptors = self._build_descriptors(tasks)
        # Retain before submission: an error may occur after some copies launch.
        self._inflight.append((tasks, descriptors))
        try:
            self._submit(descriptors)
        except BaseException:
            self._failed = True
            raise

    def synchronize(self, stream):
        if stream.npu_stream != self.stream.npu_stream:
            raise RuntimeError("ACL copies must be synchronized on their bound stream")
        stream.synchronize()
        self._inflight.clear()
        self._failed = False

    def close(self):
        if not self._closed:
            self.synchronize(self.stream)
            self._closed = True
