# SPDX-License-Identifier: Apache-2.0
"""Record the actual weight-loading stream outside measured model requests."""

import ctypes


def loading_stream_identity(stream, device):
    """Query an existing ACL stream; never create a stream or initialize ACL.

    CANN 9.1: aclError aclrtStreamGetId(aclrtStream, int32_t *).
    A torch stream handle is a pointer, not the profiler's numeric stream ID.
    """
    try:
        handle = int(stream.npu_stream)
        if handle <= 0:
            raise ValueError("expected an explicit loading stream, not the default null handle")
        library = ctypes.CDLL("libascendcl.so")
        query = library.aclrtStreamGetId
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        query.restype = ctypes.c_int32
        stream_id = ctypes.c_int32(-1)
        code = query(ctypes.c_void_p(handle), ctypes.byref(stream_id))
        if code or stream_id.value < 0:
            raise RuntimeError(f"aclrtStreamGetId failed: code={code}, stream_id={stream_id.value}")
        return dict(status="ok", schema=1, stream_id=stream_id.value, device=int(device), source="aclrtStreamGetId")
    except Exception as exc:
        # Optional diagnostics must not break the original inference path.
        # Automatic offline selection will refuse to guess if no marker exists.
        return dict(status="unavailable", source="aclrtStreamGetId", error=f"{type(exc).__name__}: {exc}")


def mark_loading_stream(identity):
    """Embed identity in this capture on the control thread, before its request."""
    from torch_npu.npu import mstx

    name = (
        f"moe_offload/loading_stream|schema={identity['schema']}|"
        f"stream_id={identity['stream_id']}|device={identity['device']}"
    )
    marker = mstx.range_start(name)
    if not isinstance(marker, int) or marker <= 0:
        raise ValueError("loading-stream MSTX marker was not created")
    mstx.range_end(marker)
