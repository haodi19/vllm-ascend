# SPDX-License-Identifier: Apache-2.0
"""Isolated real-copy correctness probe; synthetic weights, no MoE model."""

import argparse
import ctypes
import json
import os
import traceback
from pathlib import Path

from host_sync_device_probe import check_result

EXPERTS = 4
WEIGHT_BASE = 1000
ACL_HOST_TO_DEVICE = 1
ACL_DEVICE_TO_HOST = 2


def expected_output(round_id, elements):
    first = (round_id % EXPERTS + 1) * WEIGHT_BASE
    return [first + 2 * round_id, first + elements - 1, round_id, round_id]


def check_output(actual, round_id, elements):
    if actual != expected_output(round_id, elements):
        raise RuntimeError(f"Stale/incorrect consumer output at round {round_id}: {actual}")


def run(args):
    import torch
    import torch_npu
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    print(
        f"backend={args.backend}; device={torch.npu.get_device_name(0)}; weight_bytes={args.elements * 8}", flush=True
    )
    acl = ctypes.CDLL("libascendcl.so")
    copy = acl.aclrtMemcpyAsync
    copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    copy.restype = ctypes.c_int
    synchronize = acl.aclrtSynchronizeStream
    synchronize.argtypes = [ctypes.c_void_p]
    synchronize.restype = ctypes.c_int
    kernels = ctypes.CDLL(str(args.library.resolve()))
    publish, wait = kernels.host_sync_publish_launch, kernels.host_sync_wait_launch
    consume = kernels.host_sync_consume_launch
    publish.argtypes = [ctypes.c_void_p] * 3
    wait.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64]
    consume.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint64] * 2
    publish.restype = wait.restype = consume.restype = None
    compute, transfer = torch.npu.Stream(), torch.npu.Stream()
    compute_ptr, transfer_ptr = int(compute.npu_stream), int(transfer.npu_stream)
    route_h = torch.zeros(8, dtype=torch.int64, pin_memory=True)
    mapping_h = torch.zeros(8, dtype=torch.int64, pin_memory=True)
    bank = [torch.empty(args.elements, dtype=torch.int64, pin_memory=True) for _ in range(EXPERTS)]
    for expert, weights in enumerate(bank):
        weights.copy_(torch.arange(args.elements, dtype=torch.int64) + (expert + 1) * WEIGHT_BASE)
    route_d = torch.zeros(8, dtype=torch.int64, device="npu:0")
    mapping_d = torch.zeros_like(route_d)
    weights_d = torch.zeros(args.elements, dtype=torch.int64, device="npu:0")
    state = torch.zeros(4, dtype=torch.int64, device="npu:0")
    output = torch.zeros_like(state)
    torch.npu.synchronize()
    if any(t.data_ptr() % 64 for t in [route_h, mapping_h, *bank, route_d, mapping_d, weights_d]):
        raise RuntimeError("ACL copy requires aligned buffers")

    def copy_tensor(dst, src, direction, stream):
        size = src.numel() * src.element_size()
        result = copy(dst.data_ptr(), dst.numel() * dst.element_size(), src.data_ptr(), size, direction, stream)
        if result:
            raise RuntimeError(f"aclrtMemcpyAsync failed: {result}")

    calls, errors = [], []

    def callback(_unused=None):
        try:
            round_id = int(route_h[0])  # CPU pinned memory; D2H precedes publication.
            if round_id != len(calls) + 1:
                raise RuntimeError(f"Worker saw stale route: {round_id}, calls={calls}")
            copy_tensor(weights_d, bank[round_id % EXPERTS], ACL_HOST_TO_DEVICE, transfer_ptr)
            result = synchronize(transfer_ptr)
            if result:
                raise RuntimeError(f"Transfer stream synchronization failed: {result}")
            mapping_h.fill_(round_id)
            calls.append(round_id)
        except Exception as exc:
            errors.append(str(exc))
            raise

    session = None
    alias = None
    if args.backend == "mte":
        session = hs.Session(slots=1, backend="acl", device_id=0)
        info = session.info()
        session.register_task(0, callback)
        session.start()
        alias = session.begin_device_access()
    else:
        torch_npu.npu._subscribe_report(compute)

    def submit():
        copy_tensor(route_h, route_d, ACL_DEVICE_TO_HOST, compute_ptr)
        if session is not None:
            publish(compute_ptr, alias + info["request_offset"], state.data_ptr())
            wait(compute_ptr, alias + info["done_offset"], state.data_ptr(), args.max_polls)
        else:
            torch_npu.npu._launch_host_func(compute, callback, None)
        copy_tensor(mapping_d, mapping_h, ACL_HOST_TO_DEVICE, compute_ptr)
        consume(
            compute_ptr,
            weights_d.data_ptr(),
            mapping_d.data_ptr(),
            route_d.data_ptr(),
            state.data_ptr(),
            output.data_ptr(),
            args.elements,
            int(session is not None),
        )

    def validate(round_id):
        if errors or calls != list(range(1, round_id + 1)):
            raise RuntimeError(f"Callback validation failed: calls={calls}; errors={errors}")
        if session is not None:
            check_result(state.cpu().tolist(), round_id, 1)
        check_output(output.cpu().tolist(), round_id, args.elements)
        if not torch.equal(weights_d.cpu(), bank[round_id % EXPERTS]):
            raise RuntimeError(f"Full weight content mismatch at round {round_id}")

    route_d.fill_(1)
    torch.npu.synchronize()
    submit()
    compute.synchronize()
    validate(1)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=compute):
        submit()
    compute.synchronize()
    validate(1)
    for round_id in range(2, args.rounds + 2):
        # Supply new input outside capture. No Torch operators are mixed with
        # native launch/copy calls inside the captured graph in this stage.
        route_d.fill_(round_id)
        torch.npu.synchronize()
        with torch.npu.stream(compute):
            graph.replay()
        compute.synchronize()
        validate(round_id)
    graph.reset()
    if session is not None:
        session.end_device_access(stream_synchronized=True)
        session.stop()
        session.close()
    else:
        torch_npu.npu._unsubscribe_report(compute)
    print(
        "DATAFLOW_RESULT "
        + json.dumps(
            {
                "backend": args.backend,
                "result": "PASS",
                "graph_replays": args.rounds,
                "host_callbacks": len(calls),
                "weight_bytes_per_round": args.elements * 8,
                "route_d2h_bytes": 64,
                "mapping_h2d_bytes": 64,
                "consumer_output": output.cpu().tolist(),
                "full_weights_checked_each_round": True,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("callback", "mte"), required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--max-polls", type=int, default=1000000)
    args = parser.parse_args()
    if not (
        1 <= args.rounds <= 10000
        and 8 <= args.elements <= 1048576
        and args.elements % 8 == 0
        and 1 <= args.max_polls <= 10000000
    ):
        parser.error("rounds: 1..10000; elements: multiple of 8 in 8..1048576; max-polls: 1..10000000")
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        print("FAIL: isolated dataflow; no MoE integration or performance claim.", flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
