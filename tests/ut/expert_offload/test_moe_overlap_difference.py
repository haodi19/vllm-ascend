# SPDX-License-Identifier: Apache-2.0
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))

from analyze_moe_overlap_difference import coordinates, decompose, matched_rows, transitions


def sample(duration, delta, layer=1):
    return dict(
        layer=layer,
        prefetch_duration_us=duration,
        readiness_minus_prefetch_end_bounds_us=[delta, delta],
        prefetch_loads=1,
        reactive_loads=2,
        last_compute_name="RoutingKernel",
        **{"class": "early" if delta < 0 else "late"},
    )


class OverlapDifferenceTests(unittest.TestCase):
    def test_shorter_host_can_flip_class_without_later_compute(self):
        pairs, rejected, _ = matched_rows({"rows": [sample(600, -100)]}, {"rows": [sample(400, 100)]})
        self.assertFalse(rejected)
        values = decompose(pairs)
        self.assertEqual(values["changes"], dict(host=-200, compute=0, delta=200))
        self.assertEqual(transitions(pairs, 40), {"early->late": 1})

    def test_later_relative_compute_without_host_change(self):
        pairs, _, _ = matched_rows({"rows": [sample(400, -100)]}, {"rows": [sample(400, 100)]})
        self.assertEqual(decompose(pairs)["changes"], dict(host=0, compute=200, delta=200))

    def test_unmapped_is_not_zero_and_unequal_loads_or_kernels_are_excluded(self):
        left = {"rows": [sample(400, -10, i) for i in range(4)]}
        right = copy.deepcopy(left)
        right["rows"][0]["class"] = "unmapped"
        right["rows"][1]["prefetch_loads"] = None
        right["rows"][2]["reactive_loads"] = 3
        right["rows"][3]["last_compute_name"] = "DifferentKernel"
        pairs, rejected, _ = matched_rows(left, right)
        self.assertFalse(pairs)
        self.assertEqual(sum(rejected.values()), 4)
        self.assertEqual(rejected["boundary_unavailable"], 1)
        self.assertIsNone(decompose(pairs))

    def test_layer_sequence_must_match(self):
        with self.assertRaisesRegex(ValueError, "layer sequence"):
            matched_rows({"rows": [sample(10, 1, 1)]}, {"rows": [sample(10, 1, 2)]})

    def test_invalid_bounds_rejected_and_uncertain_can_be_compared(self):
        row = sample(100, 20)
        row["class"] = "uncertain"
        self.assertEqual(coordinates(row)["delta"], 20)
        row["readiness_minus_prefetch_end_bounds_us"] = [30, 20]
        with self.assertRaisesRegex(ValueError, "invalid saved"):
            coordinates(row)

    def test_common_margin_and_zero_margin_are_separate(self):
        pairs, _, _ = matched_rows({"rows": [sample(100, -10)]}, {"rows": [sample(100, 10)]})
        self.assertEqual(transitions(pairs, 40), {"uncertain->uncertain": 1})
        self.assertEqual(transitions(pairs, 0), {"early->late": 1})


if __name__ == "__main__":
    unittest.main()
