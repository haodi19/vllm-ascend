# SPDX-License-Identifier: Apache-2.0
"""CPU tests for separating observed child exits, startup phases, and cleanup."""

import importlib.util
import json
import unittest
from pathlib import Path


class StartupSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[3] / "script/summarize_mte_startup.py"
        spec = importlib.util.spec_from_file_location("mte_startup_summary_test", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @staticmethod
    def trace(kind, **record):
        return f"[MOE-MTE {kind}] " + json.dumps(record)

    def phase(self, phase, event, pid=101, tid=1, when=10, **fields):
        return self.trace(
            "STARTUP", pid=pid, tid=tid, phase=phase, event=event, monotonic_ns=when, wall_time_ns=when + 1000, **fields
        )

    def process(self, event, when=10, **fields):
        return self.trace(
            "PROCESS", observer_pid=50, event=event, monotonic_ns=when, wall_time_ns=when + 1000, **fields
        )

    def armed(self):
        return [
            self.process("armed"),
            self.process("start", child_pid=101, name="EngineCore_DP0"),
            self.phase("runner.faulthandler", "begin"),
            self.phase("runner.faulthandler", "end"),
        ]

    def summary(self, lines, wrapper_lines=()):
        return self.module.startup_evidence("\n".join(lines), "\n".join(wrapper_lines))

    @staticmethod
    def exit_records(lines):
        return [
            json.loads(line.split("STARTUP_ENGINE_EXIT ", 1)[1])
            for line in lines
            if line.startswith("STARTUP_ENGINE_EXIT {")
        ]

    def test_missing_trace_and_api_exit_never_supply_an_engine_exit(self):
        result = "\n".join(
            self.summary(
                [
                    "(EngineCore pid=101) Loading weights...",
                    "(APIServer pid=50) RuntimeError: Engine initialization failed",
                    "Service exited during startup: 1",
                ]
            )
        )
        self.assertIn("STARTUP_TRACE_STATUS=UNVERIFIED", result)
        self.assertIn("STARTUP_ENGINE_EXIT pid=101: NOT_OBSERVED", result)
        self.assertNotIn('"returncode": 1', result)

    def test_only_matching_engine_child_status_is_reported(self):
        lines = self.armed() + [
            self.process("exit", child_pid=777, name="resource_tracker", returncode=1),
            self.process("exit", child_pid=101, returncode=-11, signal="SIGSEGV"),
        ]
        summary = self.summary(lines)
        exits = self.exit_records(summary)
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["pid"], 101)
        self.assertEqual(exits[0]["returncode"], -11)
        self.assertIn("STARTUP_TRACE_STATUS=ARMED_EVIDENCE_PRESENT", summary[0])
        self.assertIn("实际死亡时刻", "\n".join(summary))

    def test_cleanup_comparison_uses_matching_api_and_child(self):
        lines = self.armed() + [
            self.process("signal_request", when=20, child_pid=101, signal="SIGTERM"),
            self.trace("RUNNER", event="cleanup_begin", pid=10, api_pid=50, monotonic_ns=25),
            self.process("exit", when=30, child_pid=101, returncode=-15, signal="SIGTERM"),
        ]
        wrapper = [self.trace("RUNNER", event="wrapper_cleanup_begin", wrapper_pid=99, monotonic_ns=27)]
        exits = self.exit_records(self.summary(lines, wrapper))
        self.assertTrue(exits[0]["observed_after_parent_signal_request"])
        self.assertTrue(exits[0]["observed_after_runner_cleanup"])
        self.assertTrue(exits[0]["observed_after_run_wrapper_cleanup"])

        unrelated = self.armed() + [
            self.process("signal_request", when=20, child_pid=777, signal="SIGTERM"),
            self.trace("RUNNER", event="cleanup_begin", api_pid=999, monotonic_ns=20),
            self.process("exit", when=30, child_pid=101, returncode=-15, signal="SIGTERM"),
        ]
        exits = self.exit_records(self.summary(unrelated))
        self.assertFalse(exits[0]["observed_after_parent_signal_request"])
        self.assertFalse(exits[0]["observed_after_runner_cleanup"])

    def test_later_cleanup_and_missing_clocks_do_not_claim_preceding_cleanup(self):
        lines = self.armed() + [
            self.process("exit", when=20, child_pid=101, returncode=1),
            self.process("signal_request", when=30, child_pid=101, signal="SIGTERM"),
            self.trace("RUNNER", event="cleanup_begin", api_pid=50),
        ]
        record = self.exit_records(self.summary(lines))[0]
        self.assertFalse(record["observed_after_parent_signal_request"])
        self.assertIsNone(record["observed_after_runner_cleanup"])

    def test_nested_phases_track_unclosed_begin_and_exception_separately(self):
        lines = self.armed() + [
            self.phase("runner.model_load_profile", "begin"),
            self.phase("draft.share_lm_head", "begin"),
            self.phase("draft.lm_head_equal", "begin"),
            self.phase("draft.lm_head_equal", "exception", exception_type="RuntimeError"),
            self.phase("draft.share_lm_head", "exception", exception_type="RuntimeError"),
        ]
        result = "\n".join(self.summary(lines))
        self.assertIn("STARTUP_OPEN_PHASES=1; PHASE_EXCEPTIONS=2", result)
        self.assertIn("phase=runner.model_load_profile unmatched_begin=1", result)
        self.assertNotIn("phase=draft.lm_head_equal unmatched_begin=", result)

    def test_phase_pairing_is_thread_specific_and_repeated_markers_are_deduplicated(self):
        begin = self.phase("draft.lm_head_equal", "begin", tid=1)
        lines = self.armed() + [begin, self.phase("draft.lm_head_equal", "end", tid=2)]
        result = "\n".join(self.summary(lines, [begin]))
        self.assertIn("duplicates=1", result)
        self.assertIn("unmatched_end=1", result)
        self.assertIn("STARTUP_TRACE_STATUS=UNVERIFIED", result)
        self.assertIn("STARTUP_OPEN pid=101 tid=1", result)

    def test_malformed_markers_and_custom_observer_gap_are_visible(self):
        lines = self.armed() + [
            "[MOE-MTE STARTUP] {broken",
            "[MOE-MTE PROCESS] []",
            self.process("observer_gap", child_pid=101),
        ]
        result = "\n".join(self.summary(lines))
        self.assertIn("malformed=2", result)
        self.assertIn("OBSERVER_GAPS=1", result)
        self.assertIn("STARTUP_TRACE_STATUS=UNVERIFIED", result)

    def test_process_exit_is_retained_when_phase_records_exceed_display_limit(self):
        lines = self.armed()
        for index in range(self.module.MAX_TRACE_RECORDS):
            lines.extend((self.phase(f"draft.step_{index}", "begin"), self.phase(f"draft.step_{index}", "end")))
        lines.append(self.process("exit", child_pid=101, returncode=7))
        result = "\n".join(self.summary(lines))
        self.assertIn('"returncode": 7', result)
        self.assertIn('TRACE source=service [MOE-MTE PROCESS] {"child_pid": 101, "event": "exit"', result)
        self.assertNotIn("OMITTED=0", result)

    def test_current_thread_stack_survives_many_other_thread_blocks(self):
        lines = ["Fatal Python error: Segmentation fault", ""]
        for index in range(10):
            lines.append(f"Thread 0x{index + 1:08x} (most recent call first):")
            lines.extend(f'  File "other.py", line {frame}, in idle' for frame in range(15))
            lines.append("")
        lines.extend(
            [
                "Current thread 0x0000abcd (most recent call first):",
                '  File "llm_base_proposer.py", line 708, in _maybe_share_lm_head',
                "",
            ]
        )
        result = "\n".join(self.summary(lines))
        self.assertIn("Current thread 0x0000abcd", result)
        self.assertIn("_maybe_share_lm_head", result)
        self.assertIn("STARTUP_ENGINE_EXIT=NOT_OBSERVED", result)
        self.assertIn("未据此自动认定属于EngineCore", result)


if __name__ == "__main__":
    unittest.main()
