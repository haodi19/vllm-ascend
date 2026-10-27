"""CPU-only polling policy, timing and lifecycle regression."""

import sys
import unittest
from pathlib import Path

import vllm_ascend_host_sync as hs

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
from host_sync_benchmark import summarize_worker
from run_worker_poll_tests import cases


class WorkerPollingTests(unittest.TestCase):
    def test_modes_and_timing_preserve_generations(self):
        for mode in ("yield", "spin"):
            for timing in (False, True):
                with self.subTest(mode=mode, timing=timing), hs.Session(slots=4, backend="cpu") as session:
                    calls = []
                    session.configure_polling(mode, timing)
                    session.register_task(3, lambda calls=calls: calls.append(1))
                    session.start()
                    for generation in range(1, 7):
                        session.submit_for_test(3, generation)
                        session.wait(3, generation)
                    session.stop()
                    stats = session.polling_stats()
                    self.assertEqual(len(calls), 6)
                    self.assertEqual(stats["idle_mode"], mode)
                    if timing:
                        self.assertEqual([row[:2] for row in stats["task_samples"]], [[3, n] for n in range(1, 7)])
                        self.assertTrue(all(v >= 0 for row in stats["task_samples"] for v in row))
                        self.assertEqual(summarize_worker(stats, 2, 4)["task"]["gil_wait_us"]["count"], 4)
                    else:
                        self.assertEqual(stats["task_samples"], [])

    def test_configuration_and_snapshot_lifecycle(self):
        with hs.Session(backend="cpu") as session:
            with self.assertRaises(ValueError):
                session.configure_polling("invalid")
            session.register_task(0, lambda: None)
            session.start()
            with self.assertRaises(RuntimeError):
                session.configure_polling("spin")
            with self.assertRaises(RuntimeError):
                session.polling_stats()
            session.stop()
            self.assertEqual(session.polling_stats()["idle_mode"], "yield")

    def test_spin_callback_error_is_not_success(self):
        def fail():
            raise ValueError("expected failure")

        with hs.Session(backend="cpu") as session:
            session.configure_polling("spin", True)
            session.register_task(0, fail)
            session.start()
            session.submit_for_test(0, 1)
            with self.assertRaises(RuntimeError):
                session.wait(0, 1)
            session.stop()
            self.assertEqual(session.snapshot(0)["done_generation"], 0)

    def test_comparison_order(self):
        self.assertEqual(
            list(cases(2, [1]))[:4],
            [(1, False, 0, "yield"), (1, False, 0, "spin"), (1, False, 1, "spin"), (1, False, 1, "yield")],
        )


if __name__ == "__main__":
    unittest.main()
