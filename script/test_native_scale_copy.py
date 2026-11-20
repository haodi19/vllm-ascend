# SPDX-License-Identifier: Apache-2.0
"""Small NPU test: byte copies of prepacked scales vs Tensor.copy_; no model."""

import importlib.util
from pathlib import Path

import _vllm_ascend_lrc as native
import numpy as np
import torch
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_update", ROOT / "vllm_ascend/expert_offload/native_update.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    torch.npu.set_device(0)
    for dtype in (torch.uint8, torch.float32):
        for shape, stride in (((7, 8), (8, 1)), ((7, 8), (1, 7))):
            stream = torch.npu.Stream()
            target = torch.empty_strided(shape, stride, dtype=dtype, device="npu")
            reference = torch.empty_strided(shape, stride, dtype=dtype, device="npu")
            if tuple(target.stride()) != stride or tuple(reference.stride()) != stride:
                raise RuntimeError("NPU did not preserve requested scale test strides")
            source = torch.arange(56).reshape(shape).to(dtype)
            packed = module.pack_scale(torch, source, target)
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


if __name__ == "__main__":
    main()
