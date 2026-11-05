# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for callback-batch gaps, without Torch imports."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class OffloadBubbleStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "offload_bubble_stats_test", root / "script/offload_bubble_stats.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    @staticmethod
    def callback(call, kind, layer, start, end):
        return dict(call=call, kind=kind, layer=layer, start=start, end=end, event_index=call, pid="host", tid="worker")

    @staticmethod
    def task(index, kind, start, end):
        return dict(event_index=index, kind=kind, start=start, end=end, duration=end - start)

    def analyze(self, tasks, callbacks):
        return self.module.analyze_offload_batches(tasks, callbacks, lower_us=100, long_us=5000)

    def windows(self):
        return [self.callback(1, "prefetch", 1, 10, 40), self.callback(2, "reactive", 1, 60, 100)]

    def test_explicit_bounds_exclude_nanosecond_seams_and_five_ms(self):
        result = self.module.bounded_gap_stats([0, 0.02, 9.99, 10, 4999.9, 5000, 10000], 10, 5000)
        self.assertEqual(result["candidates"], 7)
        self.assertEqual(result["excluded_short"], 3)
        self.assertEqual(result["excluded_long"], 2)
        self.assertEqual(result["kept"]["n"], 2)
        self.assertAlmostEqual(result["kept"]["mean"], 2504.95)

    def test_bounds_do_not_apply_additional_iqr_pruning(self):
        result = self.module.bounded_gap_stats([10, 11, 12, 13, 14, 15, 16, 17, 4000], 10, 5000)
        self.assertEqual(result["kept"]["n"], 9)
        self.assertEqual(result["kept"]["max"], 4000)

    def test_activity_union_handles_overlap_and_short_copy_transpose_seams(self):
        tasks = [
            self.task(1, "copy", 0, 10),
            self.task(2, "transpose", 10.02, 50),
            self.task(3, "copy", 20, 30),
            self.task(4, "copy", 70, 80),
        ]
        result = self.module.activity_window_gaps(tasks, lower_us=10, long_us=5000)
        self.assertEqual(result["overlaps_merged"], 1)
        self.assertEqual(result["excluded_short"], 1)
        self.assertEqual(result["kept"]["n"], 1)
        self.assertEqual(result["kept"]["mean"], 20)

    def test_callback_envelope_keeps_between_callback_gap_without_clipping_edges(self):
        tasks = [self.task(i, "copy", start, start + 5) for i, start in enumerate((0, 20, 50, 80, 120))]
        full = self.module.activity_window_gaps(tasks, lower_us=10, long_us=5000)
        scoped = self.module.activity_window_gaps(tasks, lower_us=10, long_us=5000, window=(40, 100))
        self.assertEqual(full["kept"]["n"], 4)
        self.assertEqual(scoped["kept"]["n"], 1)
        self.assertEqual(scoped["kept"]["mean"], 25)
        self.assertEqual(scoped["excluded_outside_window"], 1)
        self.assertEqual(scoped["excluded_crossing_window"], 2)

    def test_no_activity_does_not_invent_a_window_sized_gap(self):
        result = self.module.activity_window_gaps([], lower_us=10, long_us=5000, window=(0, 100))
        self.assertIsNone(result["kept"])
        self.assertEqual(result["candidates"], 0)

    def test_prefill_and_post_callback_copies_do_not_enter_decode_gaps(self):
        tasks = [
            self.task(1, "copy", 0, 5),
            self.task(2, "copy", 12, 20),
            self.task(3, "copy", 65, 75),
            self.task(4, "copy", 110, 120),
        ]
        result = self.analyze(tasks, self.windows())
        self.assertEqual(result["copy_phases"], {"before_callbacks": 1, "inside_unique": 2, "after_callbacks": 1})
        self.assertEqual([pair["gap_us"] for pair in result["pairs"]], [45])
        self.assertFalse(result["partial"])

    def test_transpose_is_busy_time_and_changes_last_batch_boundary(self):
        tasks = [self.task(1, "copy", 12, 17), self.task(2, "transpose", 17, 35), self.task(3, "copy", 65, 75)]
        result = self.analyze(tasks, self.windows())
        pair = result["pairs"][0]
        self.assertEqual(pair["gap_us"], 30)
        self.assertEqual(pair["copy_only_gap_us"], 48)
        self.assertEqual(pair["copy_only_excess_us"], 18)
        self.assertEqual(pair["left_task"]["kind"], "transpose")
        self.assertEqual(pair["left_batch"]["transpose_tasks"], 1)
        self.assertEqual(result["activity_phases"], {"inside_unique": {"copy": 2, "transpose": 1}})

    def test_all_device_tasks_within_batch_extend_its_boundary(self):
        tasks = [self.task(1, "copy", 12, 17), self.task(2, "compute", 25, 39), self.task(3, "copy", 65, 75)]
        result = self.analyze(tasks, self.windows())
        self.assertEqual(result["pairs"][0]["gap_us"], 26)
        self.assertEqual(result["pairs"][0]["left_task"]["kind"], "compute")

    def test_unmatched_device_work_between_callbacks_is_not_idle(self):
        tasks = [self.task(1, "copy", 12, 20), self.task(2, "compute", 45, 55), self.task(3, "copy", 65, 75)]
        result = self.analyze(tasks, self.windows())
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["excluded_pairs"]["occupied_by_other_device_task"], 1)
        self.assertTrue(result["partial"])

    def test_empty_callback_is_not_skipped_to_construct_a_pair(self):
        callbacks = [
            self.callback(1, "prefetch", 1, 0, 10),
            self.callback(2, "reactive", 1, 20, 30),
            self.callback(3, "prefetch", 2, 40, 50),
        ]
        tasks = [self.task(1, "copy", 1, 2), self.task(2, "copy", 41, 42)]
        result = self.analyze(tasks, callbacks)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["excluded_pairs"]["empty_or_unmatched_callback"], 2)

    def test_directions_use_layers_and_round_reset_is_separate(self):
        callbacks = [
            self.callback(1, "reactive", 0, 0, 10),
            self.callback(2, "prefetch", 1, 20, 30),
            self.callback(3, "reactive", 1, 40, 50),
            self.callback(4, "reactive", 0, 10000, 10010),
        ]
        tasks = [self.task(i, "copy", start + 1, start + 2) for i, start in enumerate((0, 20, 40, 10000), 1)]
        result = self.analyze(tasks, callbacks)
        self.assertEqual([pair["direction"] for pair in result["pairs"]], ["reactive->prefetch", "prefetch->reactive"])
        self.assertEqual([pair["gap_us"] for pair in result["pairs"]], [19, 19])
        self.assertEqual(len(result["round_pairs"]), 1)
        self.assertEqual(result["round_pairs"][0]["gap_us"], 9959)
        self.assertEqual(result["round_gaps"]["all"]["raw"]["n"], 1)

    def test_long_gap_with_normal_layers_is_not_reclassified_as_round(self):
        callbacks = [self.callback(1, "prefetch", 1, 0, 10), self.callback(2, "reactive", 1, 10020, 10040)]
        tasks = [self.task(1, "copy", 2, 6), self.task(2, "copy", 10025, 10030)]
        result = self.analyze(tasks, callbacks)
        self.assertEqual(result["round_pairs"], [])
        stats = result["directions"]["prefetch->reactive"]
        self.assertEqual(stats["all"]["raw"]["mean"], 10019)
        self.assertEqual(stats["long_band"]["raw"]["n"], 1)
        self.assertIsNone(stats["regular_band"]["raw"])

    def test_crossing_device_task_contaminates_window_instead_of_being_clipped(self):
        tasks = [self.task(1, "copy", 5, 15), self.task(2, "copy", 20, 25), self.task(3, "copy", 65, 75)]
        result = self.analyze(tasks, self.windows())
        self.assertEqual(result["task_matches"]["cross_boundary"], 1)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["excluded_pairs"]["ambiguous_or_crossing_device_task"], 1)
        self.assertTrue(result["partial"])

    def test_overlapping_callback_windows_are_ambiguous(self):
        callbacks = [self.callback(1, "prefetch", 1, 0, 30), self.callback(2, "reactive", 1, 20, 50)]
        tasks = [self.task(1, "copy", 1, 2), self.task(2, "copy", 25, 26), self.task(3, "copy", 40, 42)]
        result = self.analyze(tasks, callbacks)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["task_matches"]["ambiguous"], 1)
        self.assertEqual(result["partial_reasons"]["overlapping_callbacks"], 2)
        self.assertTrue(result["partial"])

    def test_bands_keep_real_short_gaps_and_report_long_tail_without_dropping_it(self):
        result = self.module.distributions([0, 50, 99, 100, 4999, 5000, 10000], 100, 5000)
        self.assertEqual(result["all"]["raw"]["n"], 7)
        self.assertEqual(result["regular_band"]["raw"]["n"], 5)
        self.assertEqual(result["large_band"]["raw"]["n"], 2)
        self.assertEqual(result["long_band"]["raw"]["n"], 2)
        self.assertEqual(result["zero_gaps"], 1)
        self.assertEqual(result["below_large_threshold"], 3)
        self.assertAlmostEqual(
            result["regular_band"]["raw"]["total"] + result["long_band"]["raw"]["total"],
            result["all"]["raw"]["total"],
        )

    def test_back_to_back_batches_keep_zero_gap(self):
        callbacks = [self.callback(1, "prefetch", 1, 0, 10), self.callback(2, "reactive", 1, 10, 20)]
        tasks = [self.task(1, "copy", 2, 10), self.task(2, "copy", 10, 15)]
        result = self.analyze(tasks, callbacks)
        self.assertFalse(result["partial"])
        self.assertEqual(result["pairs"][0]["gap_us"], 0)
        self.assertEqual(result["directions"]["prefetch->reactive"]["regular_band"]["raw"]["n"], 1)
        self.assertEqual(result["directions"]["prefetch->reactive"]["zero_gaps"], 1)

    def test_missing_callback_windows_fail_instead_of_using_whole_trace(self):
        with self.assertRaisesRegex(ValueError, "host_callback"):
            self.analyze([self.task(1, "copy", 0, 5)], [])

    def test_duplicate_call_identity_is_not_trusted(self):
        callbacks = self.windows()
        callbacks[1]["call"] = callbacks[0]["call"]
        result = self.analyze([self.task(1, "copy", 12, 20), self.task(2, "copy", 65, 75)], callbacks)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["partial_reasons"]["invalid_callback_identity"], 2)

    def test_sorting_does_not_mutate_original_task_or_callback_records(self):
        tasks = [self.task(2, "copy", 65, 75), self.task(1, "copy", 12, 20)]
        callbacks = list(reversed(self.windows()))
        original = copy.deepcopy((tasks, callbacks))
        result = self.analyze(tasks, callbacks)
        self.assertEqual(result["pairs"][0]["gap_us"], 45)
        self.assertEqual((tasks, callbacks), original)


if __name__ == "__main__":
    unittest.main()
