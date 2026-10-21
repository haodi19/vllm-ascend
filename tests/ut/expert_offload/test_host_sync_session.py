# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly with unittest; no vLLM, PyTorch or NPU is needed."""

import os
import subprocess
import sys
import threading
import unittest

import vllm_ascend_host_sync as hs


class HostSyncSessionTests(unittest.TestCase):
    def make_session(self, **kwargs):
        session = hs.Session(backend="cpu", **kwargs)
        self.addCleanup(session.close)
        return session

    def test_allocation_and_initial_state(self):
        session = self.make_session(slots=3)
        info = session.info()
        self.assertEqual(info["backend"], "cpu")
        self.assertEqual(info["slots"], 3)
        self.assertEqual(info["state"], "created")
        self.assertGreater(info["host_address"], 0)
        self.assertGreater(info["slot_stride"], 0)
        self.assertGreaterEqual(info["allocation_bytes"], 3 * info["slot_stride"])
        for slot in range(3):
            snapshot = session.snapshot(slot)
            self.assertEqual(snapshot["request_generation"], 0)
            self.assertEqual(snapshot["done_generation"], 0)
            self.assertEqual(snapshot["error_generation"], 0)
            self.assertEqual(snapshot["status"], 0)

    def test_many_rounds_and_slots_use_one_worker(self):
        session = self.make_session(slots=3)
        calls = [0, 0, 0]
        worker_ids = set()

        def task(slot):
            def callback():
                calls[slot] += 1
                worker_ids.add(threading.get_ident())

            return callback

        for slot in range(3):
            session.register_task(slot, task(slot))
        session.start()
        for generation in range(1, 51):
            for slot in range(3):
                session.submit_for_test(slot, generation)
            for slot in range(3):
                result = session.wait(slot, generation)
                self.assertEqual(result["request_generation"], generation)
                self.assertEqual(result["done_generation"], generation)
                self.assertEqual(result["status"], 2)
                self.assertEqual(result["error_generation"], 0)
        self.assertEqual(calls, [50, 50, 50])
        self.assertEqual(len(worker_ids), 1)
        self.assertNotIn(threading.get_ident(), worker_ids)
        self.assertTrue(session.info()["worker_thread_id"])

    def test_generation_must_increase_by_one(self):
        session = self.make_session()
        session.register_task(0, lambda: None)
        session.start()
        for generation in (0, 2):
            with self.assertRaises((ValueError, RuntimeError)):
                session.submit_for_test(0, generation)
        session.submit_for_test(0, 1)
        session.wait(0, 1)
        for generation in (1, 3):
            with self.assertRaises((ValueError, RuntimeError)):
                session.submit_for_test(0, generation)
        session.submit_for_test(0, 2)
        self.assertEqual(session.wait(0, 2)["done_generation"], 2)

    def test_task_cannot_be_changed_after_start(self):
        session = self.make_session()
        session.register_task(0, lambda: None)
        session.start()
        with self.assertRaises(RuntimeError):
            session.register_task(0, lambda: None)
        with self.assertRaises(RuntimeError):
            session.start()

    def test_submit_requires_running_registered_slot(self):
        session = self.make_session(slots=2)
        with self.assertRaises(RuntimeError):
            session.start()
        session.register_task(0, lambda: None)
        with self.assertRaises(RuntimeError):
            session.submit_for_test(0, 1)
        session.start()
        with self.assertRaises(RuntimeError):
            session.submit_for_test(1, 1)
        with self.assertRaises(ValueError):
            session.wait(0, 1)

    def test_outstanding_request_cannot_be_overwritten(self):
        session = self.make_session()
        entered = threading.Event()
        release = threading.Event()

        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release callback")

        session.register_task(0, callback)
        session.start()
        session.submit_for_test(0, 1)
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(RuntimeError):
                session.submit_for_test(0, 2)
            self.assertEqual(session.snapshot(0)["request_generation"], 1)
        finally:
            release.set()
        session.wait(0, 1)

    def test_callback_exception_is_not_success(self):
        session = self.make_session()

        def callback():
            raise ValueError("stage1 deliberate callback failure")

        session.register_task(0, callback)
        session.start()
        session.submit_for_test(0, 1)
        with self.assertRaisesRegex(RuntimeError, "stage1 deliberate callback failure"):
            session.wait(0, 1)
        snapshot = session.snapshot(0)
        self.assertEqual(snapshot["error_generation"], 1)
        self.assertEqual(snapshot["status"], 3)
        self.assertIn("stage1 deliberate callback failure", snapshot["error"])
        self.assertNotEqual(snapshot["done_generation"], 1)

    def test_wait_timeout_does_not_destroy_session(self):
        session = self.make_session()
        entered = threading.Event()
        release = threading.Event()

        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release callback")

        session.register_task(0, callback)
        session.start()
        session.submit_for_test(0, 1)
        try:
            self.assertTrue(entered.wait(2), "worker could not acquire the GIL")
            with self.assertRaises(TimeoutError):
                session.wait(0, 1, timeout_ms=20)
            self.assertEqual(session.info()["state"], "running")
        finally:
            release.set()
        self.assertEqual(session.wait(0, 1)["status"], 2)

    def test_pending_task_is_cancelled_and_stop_can_be_retried(self):
        session = self.make_session(slots=2)
        entered = threading.Event()
        release = threading.Event()
        pending_called = threading.Event()

        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release callback")

        session.register_task(0, callback)
        session.register_task(1, pending_called.set)
        session.start()
        session.submit_for_test(0, 1)
        try:
            self.assertTrue(entered.wait(2))
            session.submit_for_test(1, 1)
            with self.assertRaises(TimeoutError):
                session.stop(timeout_ms=20)
            self.assertEqual(session.info()["state"], "stopping")
            self.assertGreater(session.info()["host_address"], 0)
            with self.assertRaises(RuntimeError):
                session.submit_for_test(0, 2)
        finally:
            release.set()
        session.stop()
        self.assertEqual(session.info()["state"], "stopped")
        self.assertFalse(pending_called.is_set())
        self.assertEqual(session.snapshot(1)["status"], 4)
        with self.assertRaises(RuntimeError):
            session.wait(1, 1)
        with self.assertRaises(RuntimeError):
            session.start()

    def test_close_timeout_keeps_resources_until_retry(self):
        session = self.make_session()
        entered = threading.Event()
        release = threading.Event()

        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release callback")

        session.register_task(0, callback)
        session.start()
        session.submit_for_test(0, 1)
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(TimeoutError):
                session.close(timeout_ms=20)
            self.assertNotEqual(session.info()["state"], "closed")
            self.assertGreater(session.info()["host_address"], 0)
        finally:
            release.set()
        session.close()
        self.assertEqual(session.info()["state"], "closed")
        session.close()

    def test_callback_cannot_wait_stop_or_close_its_own_worker(self):
        session = self.make_session()
        rejected = []

        def callback():
            for operation in (
                lambda: session.wait(0, 1, timeout_ms=20),
                lambda: session.stop(timeout_ms=20),
                lambda: session.close(timeout_ms=20),
            ):
                try:
                    operation()
                except RuntimeError:
                    rejected.append(True)

        session.register_task(0, callback)
        session.start()
        session.submit_for_test(0, 1)
        session.wait(0, 1)
        self.assertEqual(rejected, [True, True, True])
        self.assertEqual(session.info()["state"], "running")

    def test_idle_worker_does_not_prevent_python_thread_progress(self):
        session = self.make_session()
        session.register_task(0, lambda: None)
        session.start()
        progressed = threading.Event()
        other = threading.Thread(target=progressed.set, daemon=True)
        other.start()
        self.assertTrue(progressed.wait(2))
        other.join(timeout=2)
        self.assertFalse(other.is_alive())
        session.submit_for_test(0, 1)
        session.wait(0, 1)

    def test_invalid_slot_and_slot_count(self):
        with self.assertRaises((ValueError, RuntimeError)):
            hs.Session(slots=0, backend="cpu")
        session = self.make_session()
        for slot in (-1, 1):
            with self.assertRaises((ValueError, IndexError)):
                session.snapshot(slot)
            with self.assertRaises((ValueError, IndexError)):
                session.register_task(slot, lambda: None)

    def test_bad_acl_library_path_fails_without_fallback(self):
        nonexistent = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "__host_sync_missing_acl_runtime_library__",
        )
        with self.assertRaises(RuntimeError):
            hs.Session(backend="acl", acl_library=nonexistent)

    def test_context_manager_and_closed_lifecycle(self):
        with hs.Session(backend="cpu") as session:
            self.assertEqual(session.info()["state"], "created")
            session.register_task(0, lambda: None)
            session.start()
            session.submit_for_test(0, 1)
            session.wait(0, 1)
        self.assertEqual(session.info()["state"], "closed")
        session.close()
        session.stop()
        for operation in (
            session.start,
            lambda: session.snapshot(0),
            lambda: session.register_task(0, lambda: None),
            lambda: session.submit_for_test(0, 2),
            lambda: session.wait(0, 1),
        ):
            with self.assertRaises(RuntimeError):
                operation()

    def test_process_exit_stops_idle_worker(self):
        source = (
            "import vllm_ascend_host_sync as hs\n"
            "session = hs.Session(backend='cpu')\n"
            "session.register_task(0, lambda: None)\n"
            "session.start()\n"
            "session.submit_for_test(0, 1)\n"
            "session.wait(0, 1)\n"
            "print('IDLE_EXIT_OK', flush=True)\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", source],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("IDLE_EXIT_OK", result.stdout)

    def test_wait_and_close_from_different_threads(self):
        # close() may free signals before a notified waiter reacquires the
        # mutex. The waiter must check lifecycle before reading that memory.
        for _ in range(10):
            session = self.make_session()
            entered = threading.Event()
            release = threading.Event()
            failures = []

            def callback(entered=entered, release=release):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release callback")

            def wait_task(session=session, failures=failures):
                try:
                    session.wait(0, 1)
                except RuntimeError as exc:
                    if "closed" not in str(exc):
                        failures.append(str(exc))
                except Exception as exc:
                    failures.append(str(exc))

            def close_session(session=session, failures=failures):
                try:
                    session.close()
                except Exception as exc:
                    failures.append(str(exc))

            session.register_task(0, callback)
            session.start()
            session.submit_for_test(0, 1)
            self.assertTrue(entered.wait(2))
            waiter = threading.Thread(target=wait_task, daemon=True)
            closer = threading.Thread(target=close_session, daemon=True)
            waiter.start()
            closer.start()
            release.set()
            waiter.join(timeout=5)
            closer.join(timeout=5)
            self.assertFalse(waiter.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(session.info()["state"], "closed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
