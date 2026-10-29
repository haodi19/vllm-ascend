# SPDX-License-Identifier: Apache-2.0
"""CPU-only startup markers for the opt-in MTE wait-placement diagnostic."""

import faulthandler
import json
import os
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext


class StartupDiagnostics:
    def __init__(self, enabled=False):
        self.enabled = enabled

    @classmethod
    def from_config(cls, vllm_config):
        additional = getattr(vllm_config, "additional_config", None)
        offload = additional.get("expert_offload_config", {}) if isinstance(additional, Mapping) else {}
        enabled = isinstance(offload, Mapping) and offload.get("host_sync_diagnose_prefetch_wait_at_join") is True
        return cls(enabled)

    def mark(self, phase, event, exception_type=None):
        if not self.enabled:
            return
        try:
            record = {
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "phase": phase,
                "event": event,
            }
            if exception_type is not None:
                record["exception_type"] = exception_type
            print("[MOE-MTE STARTUP] " + json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
        except Exception:
            # A closed/broken diagnostic stream must not fail a successful load
            # or replace the model exception being reported by _stage.
            pass

    def arm(self, phase="faulthandler"):
        if not self.enabled:
            return
        with self.stage(phase):
            # FD 2 remains the service log even when Python stderr is a proxy.
            # SIGKILL is not catchable; the API parent records the actual exit.
            faulthandler.enable(file=2, all_threads=True)

    def stage(self, phase):
        if not self.enabled:
            return nullcontext()
        return self._stage(phase)

    @contextmanager
    def _stage(self, phase):
        self.mark(phase, "begin")
        try:
            yield
        except BaseException as error:
            # Do not stringify an exception: it could include tensors/weights.
            self.mark(phase, "exception", type(error).__name__)
            raise
        else:
            self.mark(phase, "end")
