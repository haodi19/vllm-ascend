"""Zero-load diagnoses must use IDs, not duration or visual overlap."""

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
REPORT = importlib.import_module("moe_empty_expert_report")
sys.path.pop(0)


def record(visit=1, layer=0):
    return dict(
        event_id=visit,
        kind="reactive",
        layer=layer,
        layer_visit=visit,
        status="completed",
        h2d_wait_returned=True,
        needed_ids=[3],
        hit_ids=[3],
        missing_ids=[],
        resident_before={"3": 0},
        loads=[],
        copies=[],
    )


class EmptyExpertTests(unittest.TestCase):
    def test_resident_set_is_recomputed(self):
        r = record()
        self.assertEqual(REPORT.classify(r), "all_needed_resident")
        r["resident_before"] = {"4": 0}
        self.assertEqual(REPORT.classify(r), "unknown_inconsistent_sets")

    def test_empty_candidates_are_not_cache_hits(self):
        r = record()
        r.update(needed_ids=[], hit_ids=[])
        self.assertEqual(REPORT.classify(r), "empty_candidates")

    def test_missing_without_load_is_not_cache_hit(self):
        r = record()
        r.update(hit_ids=[], missing_ids=[3], resident_before={"4": 0})
        self.assertEqual(REPORT.classify(r), "missing_but_not_loaded")

    def test_invalid_capture_does_not_claim_success(self):
        text = list(REPORT.report(dict(complete=False, errors=0, dropped=1)))
        self.assertIn("INCOMPLETE", text[-1])

    def test_same_layers_across_visits_and_different_ids(self):
        rows = [record(1), record(2)]
        rows[1].update(needed_ids=[4], hit_ids=[4], resident_before={"4": 0})
        trace = dict(complete=True, errors=0, dropped=0, metadata={"moe_layers": 1}, records=rows)
        text = "\n".join(REPORT.report(trace))
        self.assertIn("[False]", text)
        self.assertIn("EMPTY_CHECK=PASS", text)
        self.assertNotIn("EMPTY_CHAIN", text)  # never bridge two sweeps

    def test_copies_without_planned_loads_are_unknown(self):
        r = record()
        r["copies"] = [{"bytes": 128}]
        self.assertEqual(REPORT.classify(r), "unknown_copies_without_loads")


if __name__ == "__main__":
    unittest.main()
