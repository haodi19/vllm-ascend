# SPDX-License-Identifier: Apache-2.0
"""Control native MSTX measurement without wrapping native tasks in Python."""

from vllm_ascend.expert_offload.callback_timing import CallbackTimings


class NativeTimings(CallbackTimings):
    def __init__(self, profiler):
        # mstx is only an enabled-state indicator for the existing control API.
        super().__init__("cpp", mstx=True)
        self.profiler = profiler
        self.transport = "acl"
        self.load_backend = "native"

    def start(self):
        self.profiler.start()
        self.marker_errors = 0
        self.active = True
        self.mark_loading_stream()

    def stop(self):
        result = self.profiler.stop()
        self.active = False
        return dict(
            backend="cpp",
            count=result["count"],
            errors=0,
            dropped=0,
            groups=[],
            units="us",
            mstx_enabled=True,
            mstx_errors=result["marker_errors"] + self.marker_errors,
            loading_stream=self.loading_stream,
            native=True,
            scope="native task entry through mapping commit; no Python/GIL in task",
        )


def install_native_profile(manager, port):
    from torch_npu.npu import current_device

    from vllm_ascend.expert_offload.profile_loading_stream import loading_stream_identity

    profiler = manager._native_update.profiler
    if profiler is None:
        raise ValueError("native MSTX profiler was not prepared")
    collector = NativeTimings(profiler)
    device = current_device()
    collector.loading_stream = loading_stream_identity(manager.load_stream, device)
    collector.serve(port, clock_device=device)
    manager._callback_timing = collector
