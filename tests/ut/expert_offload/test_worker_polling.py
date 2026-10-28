"""CPU-only polling policy, timing and lifecycle regression."""

import os
import sys
import threading
import unittest
from pathlib import Path

import test_host_sync_session as lifecycle_tests
import vllm_ascend_host_sync as hs

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
from host_sync_benchmark import summarize_worker
from run_worker_poll_tests import cases, scan_cases


class WorkerPollingTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux worker affinity")
    def test_affinity_changes_only_worker(self):
        allowed = os.sched_getaffinity(0)
        cpu = max(allowed)
        main_thread = threading.get_ident()
        seen = []
        with hs.Session(backend="cpu") as session:
            session.configure_worker_cpu(cpu)
            session.register_task(0, lambda: seen.append((threading.get_ident(), os.sched_getaffinity(0))))
            session.start()
            with self.assertRaises(RuntimeError):
                session.configure_worker_cpu(-1)
            session.submit_for_test(0, 1)
            session.wait(0, 1)
            session.stop()
            self.assertEqual(session.polling_stats()["worker_affinity"], [cpu])
            self.assertEqual(seen[0][1], {cpu})
            self.assertNotEqual(seen[0][0], main_thread)
            self.assertEqual(os.sched_getaffinity(0), allowed)

    def test_affinity_configuration_errors(self):
        with hs.Session(backend="cpu") as session:
            for cpu in (-2, True, "0"):
                with self.assertRaises(ValueError):
                    session.configure_worker_cpu(cpu)

    @unittest.skipUnless(sys.platform == "linux", "Linux worker affinity")
    def test_disallowed_worker_cpu_fails_start_without_silent_fallback(self):
        excluded = next((cpu for cpu in range(1024) if cpu not in os.sched_getaffinity(0)), None)
        if excluded is None:
            self.skipTest("no excluded CPU in supported range")
        with hs.Session(backend="cpu") as session:
            session.configure_worker_cpu(excluded)
            session.register_task(0, lambda: None)
            with self.assertRaisesRegex(RuntimeError, "outside inherited allowed set"):
                session.start()

    def test_idle_probe_guards_and_error_reuse(self):
        calls = []

        def callback():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("first request failed")

        with hs.Session(slots=2048, backend="cpu") as session:
            with self.assertRaises(ValueError):
                session.configure_idle_probe("bad")
            session.configure_idle_probe("atomic")
            session.register_task(2047, callback)
            session.start()
            with self.assertRaises(RuntimeError):
                session.configure_idle_probe("locked")
            session.submit_for_test(2047, 1)
            with self.assertRaisesRegex(RuntimeError, "first request failed"):
                session.wait(2047, 1)
            for generation in range(2, 102):
                session.submit_for_test(2047, generation)
                session.wait(2047, generation)
            session.stop()
            self.assertEqual(len(calls), 101)

    def test_selected_slots_alternate_without_loss_or_duplicates(self):
        for selected in (False, True):
            with self.subTest(selected=selected), hs.Session(slots=2048, backend="cpu") as session:
                calls = []
                for slot in (0, 17, 2047):
                    session.register_task(slot, lambda slot=slot, calls=calls: calls.append(slot))
                if selected:
                    session.configure_scan_slots([2047, 0, 17])
                session.start()
                expected = []
                for generation in range(1, 21):
                    for slot in (17, 2047, 0):
                        session.submit_for_test(slot, generation)
                        session.wait(slot, generation)
                        expected.append(slot)
                session.stop()
                self.assertEqual(calls, expected)

    def test_scan_configuration_guards_and_callback_failure(self):
        def fail():
            raise ValueError("selected callback failure")

        with hs.Session(slots=4, backend="cpu") as session:
            for invalid in ([], [1, 1], [4], [-1], [True]):
                with self.assertRaises((ValueError, IndexError)):
                    session.configure_scan_slots(invalid)
            session.configure_scan_slots([3])
            with self.assertRaises(RuntimeError):
                session.start()
            session.register_task(3, fail)
            session.register_task(0, lambda: None)
            session.start()
            with self.assertRaises(RuntimeError):
                session.configure_scan_slots([0])
            with self.assertRaisesRegex(RuntimeError, "excluded"):
                session.submit_for_test(0, 1)
            session.submit_for_test(3, 1)
            with self.assertRaisesRegex(RuntimeError, "selected callback failure"):
                session.wait(3, 1)
            session.stop()

    def test_scan_comparison_keeps_other_options_fixed(self):
        self.assertEqual(
            list(scan_cases(2, [2048])),
            [
                (2048, False, 0, "yield", "all"),
                (2048, False, 0, "yield", "active"),
                (2048, False, 1, "yield", "active"),
                (2048, False, 1, "yield", "all"),
            ],
        )

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


class AtomicIdleLifecycleTests(lifecycle_tests.HostSyncSessionTests):
    """Reuse lifecycle race/timeout tests with atomic probing on factory sessions."""

    def make_session(self, **kwargs):
        session = super().make_session(**kwargs)
        session.configure_idle_probe("atomic")
        return session


if __name__ == "__main__":
    unittest.main()
