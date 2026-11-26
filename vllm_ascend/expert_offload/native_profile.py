# SPDX-License-Identifier: Apache-2.0
"""Control native MSTX measurement without wrapping native tasks in Python."""

from vllm_ascend.expert_offload.callback_timing import CallbackTimings


class NativeTimings(CallbackTimings):
    def __init__(self, profiler, pipeline=None):
        # mstx is only an enabled-state indicator for the existing control API.
        super().__init__("cpp", mstx=True)
        self.profiler = profiler
        self.pipeline = pipeline
        self.pipeline_start = 0
        self.transport = "acl"
        self.load_backend = "native"

    def status(self):
        result = super().status()
        native = self.profiler.status()
        result.update(
            active=self.active or native["active"],
            mstx_active=self.active or native["active"],
            inflight=native["inflight"],
            mstx_marker_attempts=self.marker_attempts + native["marker_attempts"],
            native=True,
            native_count=native["count"],
            native_marker_errors=native["marker_errors"],
        )
        if self.pipeline is not None:
            result.update(pipeline=True, pipeline_submitted=self.pipeline.submitted_count())
        return result

    def start(self):
        if self.pipeline is not None:
            self.pipeline_start = self.pipeline.submitted_count()
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
            pipeline=self.pipeline is not None,
            pipeline_jobs=(self.pipeline.submitted_count() - self.pipeline_start) if self.pipeline else None,
            scope=(
                "Host task=plan+enqueue; DMA submit on separate C++ thread; device completion after weights+mapping"
                if self.pipeline is not None
                else "native task entry through mapping commit; no Python/GIL in task"
            ),
        )


def install_native_profile(manager, port):
    from torch_npu.npu import current_device

    from vllm_ascend.expert_offload.profile_loading_stream import loading_stream_identity

    profiler = manager._native_update.profiler
    if profiler is None:
        raise ValueError("native MSTX profiler was not prepared")
    collector = NativeTimings(profiler, manager._native_update.pipeline)
    device = current_device()
    collector.loading_stream = loading_stream_identity(manager.load_stream, device)
    collector.serve(port, clock_device=device)
    manager._callback_timing = collector
