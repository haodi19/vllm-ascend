# SPDX-License-Identifier: Apache-2.0
"""Memory policy tests never clear caches or allocate model buffers."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

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

    def test_low_node_triggers_even_when_global_available_is_high(self):
        reasons = memory.reclaim_reasons(snapshot(node0=15))
        self.assertEqual(reasons, ["node0 MemFree<160GiB"])
        self.assertEqual(memory.reclaim_reasons(snapshot(node0=15, allowed=(1,))), [])

    def test_healthy_memory_and_failure_diagnostics_never_reclaim(self):
        for reclaim, reading in ((True, snapshot()), (False, snapshot(free=10, node0=1))):
            with (
                patch.object(memory, "log_memory", return_value=reading),
                patch.object(memory, "drop_page_cache") as drop,
            ):
                memory.check_host_memory("before", lambda _: None, before_start=True, reclaim=reclaim)
            drop.assert_not_called()

    def test_one_reclaim_with_before_after_evidence_and_no_repeated_attempt(self):
        lines = []
        with (
            patch.object(memory, "log_memory", side_effect=[snapshot(free=100, node0=10), snapshot(node0=140)]) as read,
            patch.object(memory, "drop_page_cache") as drop,
        ):
            memory.check_host_memory("after python", lines.append)
        drop.assert_called_once()
        self.assertEqual(read.call_count, 2)
        self.assertTrue(any("HOST_CACHE_DELTA" in line for line in lines))
        self.assertTrue(any("remaining_low_free=['node0" in line for line in lines))

    def test_available_still_low_stops_next_model(self):
        with (
            patch.object(memory, "log_memory", return_value=snapshot(free=10, available=20)),
            patch.object(memory, "drop_page_cache") as drop,
            self.assertRaisesRegex(RuntimeError, "next service not started"),
        ):
            memory.check_host_memory("before native", lambda _: None, before_start=True)
        drop.assert_called_once()

    def test_permission_failure_is_not_silently_ignored(self):
        lines = []
        with (
            patch.object(memory, "log_memory", return_value=snapshot(free=10)),
            patch.object(memory, "drop_page_cache", side_effect=PermissionError("read-only proc")),
            self.assertRaisesRegex(RuntimeError, "Page-cache reclaim failed"),
        ):
            memory.check_host_memory("before", lines.append)
        self.assertTrue(any("read-only proc" in line for line in lines))

    def test_live_group_blocks_cache_reclaim(self):
        with (
            patch.object(memory, "group_members", return_value=[123]),
            patch.object(memory.time, "monotonic", side_effect=[0, 16]),
            patch.object(memory, "log_memory") as read,
            patch.object(memory, "drop_page_cache") as drop,
            self.assertRaisesRegex(RuntimeError, "still has live processes"),
        ):
            memory.check_host_memory("after", lambda _: None, service_pgid=123)
        read.assert_not_called()
        drop.assert_not_called()

    def test_successful_cleanup_checks_group_before_snapshot_and_reclaim(self):
        order = []
        with (
            patch.object(memory, "wait_group_exit", side_effect=lambda *a: order.append("group exited")),
            patch.object(memory, "log_memory", side_effect=lambda *a: order.append("snapshot") or snapshot(free=10)),
            patch.object(memory, "drop_page_cache", side_effect=lambda: order.append("drop")),
        ):
            memory.after_service_memory("after python", lambda _: None, 123)
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
            self.assertRaisesRegex(ValueError, "original model failure"),
        ):
            try:
                raise ValueError("original model failure")
            finally:
                memory.after_service_memory("after failed", lambda _: None, 123)
        self.assertFalse(check.call_args.kwargs["reclaim"])
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

    def test_drop_command_syncs_before_writing_only_one(self):
        with patch.object(memory.subprocess, "run") as run:
            memory.drop_page_cache()
        self.assertEqual(run.call_args.kwargs["timeout"], memory.RECLAIM_TIMEOUT)
        code = run.call_args.args[0][-1]
        fake_os, fake_pathlib = ModuleType("os"), ModuleType("pathlib")
        order = []
        fake_os.sync = lambda: order.append("sync")
        target = Mock()
        target.write_text.side_effect = lambda text: order.append(text)
        fake_pathlib.Path = Mock(return_value=target)
        with patch.dict(sys.modules, os=fake_os, pathlib=fake_pathlib):
            exec(compile(code, "drop-cache-check", "exec"), {})
        fake_pathlib.Path.assert_called_once_with("/proc/sys/vm/drop_caches")
        self.assertEqual(order, ["sync", "1\n"])

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
