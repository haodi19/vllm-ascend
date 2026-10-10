"""CPU contract tests: python -B tests/ut/expert_offload/test_experimental_acl_h2d.py.

Fake NPU storage is backed by CPU memory; these are NOT CANN/NPU validation.
"""

import ctypes
import gc
import runpy
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import torch

REPO = Path(__file__).resolve().parents[3]


class FakeStorage:
    def __init__(self, data, device="cpu", pinned=True):
        self.buffer = ctypes.create_string_buffer(data, len(data))
        self.device = types.SimpleNamespace(type=device, index=0 if device == "npu" else None)
        self.pinned = pinned

    def data_ptr(self):
        return ctypes.addressof(self.buffer)

    def nbytes(self):
        return len(self.buffer)

    def is_pinned(self, device):
        if device != "npu":
            raise AssertionError("Pinned storage must be checked for NPU")
        return self.pinned


class TestAclStorageTransport(unittest.TestCase):
    def setUp(self):
        cls = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/experimental_acl_h2d.py"))[
            "ExperimentalAclStorageTransport"
        ]
        self.stream = types.SimpleNamespace(npu_stream=12, synchronize=Mock())
        self.current = Mock(return_value=self.stream)
        self.native = Mock(side_effect=self.copy_bytes)
        self.transport = cls(self.stream, 0, copy_op=self.native, current_stream=self.current)
        storage_patch = patch.object(torch, "UntypedStorage", FakeStorage)
        storage_patch.start()
        self.addCleanup(storage_patch.stop)

    def copy_bytes(self, src, dst, sizes, direction):
        self.assertEqual(direction, 0)
        for array in (src, dst, sizes):
            self.assertEqual(array.dtype, torch.int64)
            self.assertEqual(array.device.type, "cpu")
            self.assertEqual(array.ndim, 1)
            self.assertTrue(array.is_contiguous())
        for source, destination, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(destination, source, size)

    def task(self, data=b"abcd"):
        return types.SimpleNamespace(
            source=FakeStorage(data),
            destination=FakeStorage(b"?" * len(data), "npu"),
            nbytes=len(data),
            non_blocking=True,
        )

    def test_batch_byte_content_and_ownership_until_sync(self):
        tasks = [self.task(b"weight13"), self.task(b"WEIGHT2")]
        owner = weakref.ref(tasks[0].source)
        self.transport.copy_batch(tasks)
        self.native.assert_called_once()
        self.assertEqual(tasks[0].destination.buffer.raw, b"weight13")
        self.assertEqual(tasks[1].destination.buffer.raw, b"WEIGHT2")
        del tasks
        gc.collect()
        self.assertIsNotNone(owner())
        self.transport.synchronize(self.stream)
        gc.collect()
        self.assertIsNone(owner())

    def test_empty_and_close(self):
        self.transport.copy_batch([])
        self.native.assert_not_called()
        self.transport.close()
        self.transport.close()
        self.stream.synchronize.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.transport.copy_batch([self.task()])

    def test_validate_whole_batch_before_native_submission(self):
        for invalid in ("unpinned", "size", "dtype", "blocking", "device", "pointer", "overlap"):
            with self.subTest(invalid=invalid):
                first, second = self.task(), self.task()
                if invalid == "unpinned":
                    second.source.pinned = False
                elif invalid == "size":
                    second.nbytes += 1
                elif invalid == "dtype":
                    second.source = torch.zeros(4)
                elif invalid == "blocking":
                    second.non_blocking = False
                elif invalid == "device":
                    second.destination.device.index = 1
                elif invalid == "pointer":
                    second.destination.data_ptr = lambda: 0
                elif invalid == "overlap":
                    second.destination = first.destination
                with self.assertRaises((TypeError, ValueError)):
                    self.transport.copy_batch([first, second])
                self.native.assert_not_called()
                self.assertFalse(self.transport._inflight)

    def test_wrong_stream_rejected(self):
        other = types.SimpleNamespace(npu_stream=99, synchronize=Mock())
        self.current.return_value = other
        with self.assertRaisesRegex(RuntimeError, "stream context"):
            self.transport.copy_batch([self.task()])
        with self.assertRaisesRegex(RuntimeError, "bound stream"):
            self.transport.synchronize(other)
        self.native.assert_not_called()

    def test_partial_failure_holds_buffers_and_blocks_retry_until_sync(self):
        tasks = [self.task()]
        owner = weakref.ref(tasks[0].source)
        self.native.side_effect = RuntimeError("partial native submission")
        with self.assertRaisesRegex(RuntimeError, "partial native"):
            self.transport.copy_batch(tasks)
        del tasks
        gc.collect()
        self.assertIsNotNone(owner())
        with self.assertRaisesRegex(RuntimeError, "needs synchronization"):
            self.transport.copy_batch([self.task()])
        self.stream.synchronize.side_effect = RuntimeError("sync failure")
        with self.assertRaisesRegex(RuntimeError, "sync failure"):
            self.transport.synchronize(self.stream)
        self.assertTrue(self.transport._inflight)
        self.stream.synchronize.side_effect = None
        self.transport.synchronize(self.stream)
        self.assertFalse(self.transport._inflight)
        self.native.side_effect = self.copy_bytes
        self.transport.copy_batch([self.task()])
        self.transport.close()


if __name__ == "__main__":
    unittest.main()
