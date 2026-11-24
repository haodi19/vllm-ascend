# SPDX-License-Identifier: Apache-2.0
"""Memory policy tests never clear caches or allocate model buffers."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
memory = importlib.import_module("moe_host_memory")


def snapshot(free=200, available=300, node0=180, node1=180, allowed=(0, 1)):
    return dict(
        host=dict(MemFree=free, MemAvailable=available),
        nodes=dict(node0=dict(MemFree=node0), node1=dict(MemFree=node1)),
        allowed_nodes=list(allowed),
    )


class HostMemoryTests(unittest.TestCase):
    def test_reads_host_and_node_fields_with_allowed_node_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "self").mkdir()
            (root / "self/status").write_text("Mems_allowed_list:\t1-2,4\n")
            (root / "meminfo").write_text("MemFree: 1048576 kB\nMemAvailable: 2097152 kB\nHugePages_Free: 0\n")
            node = root / "node1"
            node.mkdir()
            (node / "meminfo").write_text("Node 1 MemFree: 1048576 kB\nNode 1 FilePages: 3145728 kB\n")
            result = memory.memory_snapshot(root, root)
        self.assertEqual(result["host"], dict(MemFree=1, MemAvailable=2))
        self.assertEqual(result["nodes"]["node1"], dict(MemFree=1, FilePages=3))
        self.assertEqual(result["allowed_nodes"], [1, 2, 4])

    def test_before_service_only_records_memory_without_thresholds(self):
        for reading in (snapshot(), snapshot(free=10, available=20, node0=1)):
            with (
                patch.object(memory, "log_memory", return_value=reading),
                patch.object(memory, "drop_page_cache") as drop,
            ):
                memory.check_host_memory("before", lambda _: None)
            drop.assert_not_called()

    def test_python_always_drops_exactly_once_regardless_of_free_memory(self):
        for reading in (snapshot(), snapshot(free=10, available=20, node0=1)):
            lines = []
            with (
                patch.object(memory, "wait_group_exit"),
                patch.object(memory, "log_memory", return_value=reading) as read,
                patch.object(memory, "drop_page_cache") as drop,
            ):
                memory.after_service_memory("after python", lines.append, 123, drop_cache=True)
            drop.assert_called_once()
            self.assertEqual(read.call_count, 2)
            self.assertTrue(any("HOST_CACHE_RECLAIM_PASS" in line for line in lines))
            self.assertTrue(any("HOST_CACHE_DELTA" in line for line in lines))

    def test_native_does_not_drop_cache_even_with_low_memory(self):
        with (
            patch.object(memory, "wait_group_exit"),
            patch.object(memory, "log_memory", return_value=snapshot(free=10, available=20)),
            patch.object(memory, "drop_page_cache") as drop,
        ):
            memory.after_service_memory("after native", lambda _: None, 123)
        drop.assert_not_called()

    def test_unavailable_memory_reading_does_not_skip_fixed_drop(self):
        with (
            patch.object(memory, "wait_group_exit"),
            patch.object(memory, "log_memory", side_effect=FileNotFoundError("meminfo")),
            patch.object(memory, "drop_page_cache") as drop,
        ):
            memory.after_service_memory("after python", lambda _: None, 123, drop_cache=True)
        drop.assert_called_once()

    def test_permission_failure_is_not_silently_ignored(self):
        lines = []
        with (
            patch.object(memory, "wait_group_exit"),
            patch.object(memory, "log_memory", return_value=snapshot(free=10)),
            patch.object(memory, "drop_page_cache", side_effect=PermissionError("read-only proc")),
            self.assertRaisesRegex(RuntimeError, "next service not started"),
        ):
            memory.after_service_memory("after python", lines.append, 123, drop_cache=True)
        self.assertTrue(any("read-only proc" in line for line in lines))

    def test_live_group_blocks_cache_reclaim(self):
        with (
            patch.object(memory, "group_members", return_value=[123]),
            patch.object(memory.time, "monotonic", side_effect=[0, 16]),
            patch.object(memory, "log_memory") as read,
            patch.object(memory, "drop_page_cache") as drop,
            self.assertRaisesRegex(RuntimeError, "still has live processes"),
        ):
            memory.after_service_memory("after python", lambda _: None, 123, drop_cache=True)
        read.assert_not_called()
        drop.assert_not_called()

    def test_successful_cleanup_checks_group_before_snapshot_and_reclaim(self):
        order = []
        with (
            patch.object(memory, "wait_group_exit", side_effect=lambda *a: order.append("group exited")),
            patch.object(memory, "log_memory", side_effect=lambda *a: order.append("snapshot") or snapshot(free=10)),
            patch.object(memory, "drop_page_cache", side_effect=lambda: order.append("drop")),
        ):
            memory.after_service_memory("after python", lambda _: None, 123, drop_cache=True)
        self.assertEqual(order, ["group exited", "snapshot", "drop", "snapshot"])

    def test_group_parser_excludes_zombies_and_other_services(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, state, group in ((100, "R", 100), (101, "Z", 100), (102, "S", 500)):
                path = root / str(pid)
                path.mkdir()
                (path / "stat").write_text(f"{pid} (name with ) inside) {state} 1 {group} 100 0")
            self.assertEqual(memory.group_members(100, root), [100])

    def test_original_failure_keeps_memory_evidence_without_reclaim(self):
        with (
            patch.object(memory, "check_host_memory") as check,
            patch.object(memory, "drop_page_cache") as drop,
            self.assertRaisesRegex(ValueError, "original model failure"),
        ):
            try:
                raise ValueError("original model failure")
            finally:
                memory.after_service_memory("after failed", lambda _: None, 123, drop_cache=True)
        drop.assert_not_called()
        self.assertEqual(check.call_args.kwargs["service_pgid"], 123)

    def test_failed_diagnostic_does_not_mask_original_exception(self):
        with (
            patch.object(memory, "check_host_memory", side_effect=RuntimeError("diagnostic failed")),
            self.assertRaisesRegex(ValueError, "original"),
        ):
            try:
                raise ValueError("original")
            finally:
                memory.after_service_memory("after failed", lambda _: None, 123)

    def test_drop_command_is_exact_requested_echo(self):
        with patch.object(memory.subprocess, "run") as run:
            memory.drop_page_cache()
        run.assert_called_once_with(
            ["sh", "-c", "echo 1 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
            text=True,
            timeout=memory.RECLAIM_TIMEOUT,
        )

    def test_missing_numactl_still_records_memory(self):
        lines = []
        with (
            patch.object(memory.subprocess, "run", side_effect=FileNotFoundError("numactl")),
            patch.object(memory, "memory_snapshot", return_value=snapshot()),
            patch.object(memory, "cgroup_snapshot", return_value={}),
        ):
            memory.log_memory("before", lines.append)
        self.assertTrue(any(line.startswith("NUMACTL_UNAVAILABLE") for line in lines))
        self.assertTrue(any(line.startswith("HOST_MEMORY ") for line in lines))


if __name__ == "__main__":
    unittest.main()
