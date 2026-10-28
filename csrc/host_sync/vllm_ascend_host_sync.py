"""Optional Host synchronization foundation and experimental device access lease."""

import atexit

from _vllm_ascend_host_sync import API_VERSION, _Session


class Session:
    """Own mapped control memory and one native worker; explicitly close before NPU teardown.

    ``cpu`` is a simulation backend. ``acl`` validates allocation/registration
    only in stage 1. Stage 2 device tests must acquire a device access lease
    and synchronize the device before releasing it. Callbacks take no arguments.
    """

    def __init__(self, slots=1, backend="acl", device_id=0, acl_library=""):
        if not isinstance(slots, int) or isinstance(slots, bool) or not 1 <= slots <= 4096:
            raise ValueError("slots must be an integer in [1, 4096]")
        self._native = _Session(slots, backend, device_id, acl_library)
        # Retains ownership even if the caller drops its reference while the
        # native thread is running. No module-global Session registry is needed.
        atexit.register(self._atexit_close)

    def register_task(self, slot, callback):
        self._check_slot(slot)
        self._native.register_task(slot, callback)

    def start(self, timeout_ms=5000):
        self._native.start(timeout_ms)

    def configure_polling(self, idle_mode="yield", timing=False):
        """Opt-in experiment; call before start. Defaults preserve the old worker."""
        self._native.configure_polling(idle_mode, timing)

    def polling_stats(self):
        """Read bounded diagnostic samples after stop, before close."""
        return self._native.polling_stats()

    def configure_idle_probe(self, mode="locked"):
        """Before start: optionally skip the mutex for atomic idle checks."""
        self._native.configure_idle_probe(mode)

    def configure_worker_cpu(self, cpu=-1):
        """Before start: bind only the worker; -1 preserves inherited affinity."""
        if not isinstance(cpu, int) or isinstance(cpu, bool):
            raise ValueError("worker CPU must be an integer")
        self._native.configure_worker_cpu(cpu)

    def configure_scan_slots(self, slots):
        """Before start: producer promises to publish only to these fixed slots."""
        slots = list(slots)
        for slot in slots:
            self._check_slot(slot)
        self._native.configure_scan_slots(slots)

    def configure_scan_prefix(self, count):
        """Before start: scan [0, count); producer must expand before new publications."""
        self._check_slot(count)
        self._native.configure_scan_prefix(count)

    def expand_scan_prefix(self, count):
        """While running: monotonically publish a larger initialized prefix."""
        self._check_slot(count)
        self._native.expand_scan_prefix(count)

    def submit_for_test(self, slot, generation):
        """Simulate the device producer on Host; never mix with device producers."""
        self._check_slot(slot)
        self._check_generation(generation)
        self._native.submit_for_test(slot, generation)

    def wait(self, slot, generation, timeout_ms=5000):
        self._check_slot(slot)
        self._check_generation(generation)
        return self._native.wait(slot, generation, timeout_ms)

    def snapshot(self, slot):
        self._check_slot(slot)
        return self._native.snapshot(slot)

    @staticmethod
    def _check_slot(slot):
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            raise ValueError("slot must be a non-negative integer")

    @staticmethod
    def _check_generation(generation):
        if not isinstance(generation, int) or isinstance(generation, bool) or not 0 < generation < 2**64:
            raise ValueError("generation must be an integer in [1, 2**64 - 1]")

    def info(self):
        return self._native.info()

    def begin_device_access(self):
        """Reserve registered memory for the isolated stage 2 device producer."""
        return self._native.begin_device_access()

    def end_device_access(self, *, stream_synchronized):
        """Caller must have completed device synchronization; a timeout is not enough."""
        self._native.end_device_access(stream_synchronized)

    def stop(self, timeout_ms=5000):
        """Stop accepting requests; finish active work and cancel unstarted work."""
        self._native.stop(timeout_ms)

    def close(self, timeout_ms=5000):
        """Release only after the worker exits; a timeout retains all resources."""
        self._native.close(timeout_ms)
        atexit.unregister(self._atexit_close)

    def _atexit_close(self):
        # An arbitrary Python callback cannot be forcibly interrupted safely.
        # Final cleanup waits for it; explicit close has a bounded default.
        self.close(timeout_ms=-1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


__all__ = ["API_VERSION", "Session"]
