# SPDX-License-Identifier: Apache-2.0
"""Small NPU test: byte copies of prepacked scales vs Tensor.copy_; no model."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import _vllm_ascend_lrc as native
import numpy as np
import torch
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_update", ROOT / "vllm_ascend/expert_offload/native_update.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def check_shared_mxfp_scale(n, k):
    """One loaded CPU buffer: original reference, eager, prefill, native replay.

    Include realistic per-expert byte sizes and nonzero destination offsets.
    The CPU loader tests separately exercise both w1/w3 assembly orders.
    """
    raw = (torch.arange(n * k).reshape(n, k) % 251).to(torch.uint8)
    logical = raw.reshape(n, k // 2, 2).transpose(0, 1)
    shape, stride = tuple(logical.shape), tuple(logical.stride())
    target = torch.empty_strided((2,) + shape, (n * k,) + stride, dtype=torch.uint8, device="npu")
    pool = torch.empty_strided((3,) + shape, (n * k,) + stride, dtype=torch.uint8, device="npu")
    original = torch.empty(shape, dtype=torch.uint8, device="npu")
    original.copy_(logical.contiguous())
    allocate = lambda shape, dtype: torch.empty(shape, dtype=dtype, pin_memory=True)
    source = module.allocate_scale_like(allocate, logical)
    source.copy_(logical)
    pointer = source.data_ptr()
    with patch.object(module, "pack_scale", side_effect=AssertionError("unexpected second scale buffer")):
        reused, ok = module.prepare_scale(torch, source, target[1], require_reuse=True)
    if not ok or reused is not source:
        raise AssertionError("native runtime did not reuse the loaded scale")
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        # Exercise the actual Storage copy path used by eager/hot preload and prefill.
        for tensor, slot in ((target, 1), (pool, 2)):
            tensor.zero_()
            src, dst = module.scale_copy_views(source, tensor[slot])
            if isinstance(src, torch.Tensor) or isinstance(dst, torch.Tensor):
                raise AssertionError("matching final layout fell back to Tensor.copy_")
            if src.data_ptr() != pointer or dst.data_ptr() != tensor[slot].data_ptr():
                raise AssertionError("scale storage offset was lost")
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
    for tensor, slot in ((target, 1), (pool, 2)):
        torch.testing.assert_close(tensor[slot].cpu(), original.cpu(), rtol=0, atol=0)
        if tensor[0].cpu().any():
            raise AssertionError("scale copy overwrote another expert slot")
    # C++ decode addresses must read those exact same CPU bytes.
    table = native.NativeTransfers(
        [[pointer], [pointer]], [[target[0].data_ptr()]], [n * k], int(stream.npu_stream), "", False
    )
    policy = native.Policy(1, 2, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
    task = native.native_update_task(
        policy,
        table,
        0,
        np.array([[1]], dtype=np.int64),
        None,
        np.array([0, -1], dtype=np.int32),
        False,
        1,
        (source, target),
    )
    native.run_native_update_for_test(task)
    torch.testing.assert_close(target[0].cpu(), original.cpu(), rtol=0, atol=0)
    # native_full prefill: one C++ call, all experts, same original pinned bytes.
    full = native.NativeTransfers(
        [[pointer]] * 3,
        [[pool[eid].data_ptr()] for eid in range(3)],
        [n * k],
        int(stream.npu_stream),
        "",
        False,
    )
    full.enqueue_all()
    stream.synchronize()  # The manager retains this exact completion gate.
    for eid in range(3):
        torch.testing.assert_close(pool[eid].cpu(), original.cpu(), rtol=0, atol=0)
    # A second eager batch reads updated contents, not a stale packed replica.
    source.add_(1)
    full.enqueue([(2, 1)])
    full.enqueue([])
    stream.synchronize()
    torch.testing.assert_close(pool[2].cpu(), source, rtol=0, atol=0)
    torch.testing.assert_close(pool[0].cpu(), original.cpu(), rtol=0, atol=0)
    with torch.npu.stream(stream):
        pool.zero_()
        stream.synchronize()
    try:
        full.enqueue([(0, 0), (3, 0)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid full batch accepted")
    stream.synchronize()
    if pool.cpu().any():
        raise AssertionError("invalid full batch partially submitted")
    print(f"PASS native_full eager/prefill batch: bytes={n * k}; duplicate_bytes=0", flush=True)
    print(f"PASS shared scale loader/eager/prefill/native: bytes={n * k}; duplicate_bytes=0", flush=True)


def main():
    torch.npu.set_device(0)
    for dtype in (torch.uint8, torch.float32):
        for shape, stride in (((7, 8), (8, 1)), ((7, 8), (1, 7)), ((16, 64, 2), (2, 32, 1))):
            stream = torch.npu.Stream()
            target = torch.empty_strided(shape, stride, dtype=dtype, device="npu")
            reference = torch.empty_strided(shape, stride, dtype=dtype, device="npu")
            if tuple(target.stride()) != stride or tuple(reference.stride()) != stride:
                raise RuntimeError("NPU did not preserve requested scale test strides")
            source = torch.arange(target.numel()).reshape(shape).to(dtype)
            packed = module.pack_scale(torch, source, target)
            reused, did_reuse = module.prepare_scale(torch, packed, target)
            if not did_reuse or reused is not packed:
                raise RuntimeError("matching pinned scale layout was not reused")
            reference.copy_(source)
            torch.npu.synchronize()
            # Use a second expert/slot to exercise a miss without an empty slot.
            table = native.NativeTransfers(
                [[packed.data_ptr()], [packed.data_ptr()]],
                [[target.data_ptr()]],
                [packed.numel() * packed.element_size()],
                int(stream.npu_stream),
                "",
                False,
            )
            policy = native.Policy(1, 2, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
            ids = np.array([[1]], dtype=np.int64)
            mapping = np.array([0, -1], dtype=np.int32)
            task = native.native_update_task(policy, table, 0, ids, None, mapping, False, 1, (target, packed))
            native.run_native_update_for_test(task)
            torch.testing.assert_close(target.cpu(), reference.cpu(), rtol=0, atol=0)
            print(f"PASS native scale copy dtype={dtype} shape={shape} stride={stride}", flush=True)
    for n, k in ((16, 8), (2048, 128), (4096, 128)):
        check_shared_mxfp_scale(n, k)


if __name__ == "__main__":
    main()
