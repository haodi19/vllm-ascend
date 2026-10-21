"""Optional Host synchronization foundation; device polling is not yet implemented."""

import atexit

from _vllm_ascend_host_sync import API_VERSION, _Session


class Session:
    """Own mapped control memory and one native worker; explicitly close before NPU teardown.

    ``cpu`` is a simulation backend. ``acl`` validates allocation/registration
    only in stage 1. Neither backend authorizes a device kernel to use the
    experimental addresses yet. Callbacks are zero-argument functions.
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
