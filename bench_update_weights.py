#!/usr/bin/env python3
"""Standalone single-card _update_weights benchmark; console output only.

Planning mode skips H2D; NPU mode uses synthetic unquantized byte buffers and
the real TorchCopy transport. Neither mode runs a model or a graph callback.
"""

import argparse
import ast
import contextlib
import hashlib
import logging
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from bench_lrc_policy import load_policy, model_dimensions, project_settings, report

INPUT_CASES = 256
MIB = 1024 * 1024
METHODS = (
    "_update_weights",
    "_record_cache_stats",
    "_synchronize_h2d",
    "_load_expert_weights_into_slots",
    "_build_expert_h2d_tasks",
    "_refresh_expert_fp32_scale",
)


def load_transport(path):
    module = types.ModuleType("_update_bench_transport")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def load_manager(path, npu_module, transport, logger):
    """Compile selected original methods, without importing the vLLM plugin."""
    source = path.read_bytes()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExpertOffloadManager")
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    if {n.name for n in selected} != set(METHODS):
        raise ValueError("Manager source changed: required benchmark methods are missing")
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_expert_weight")
    cls.body = selected
    namespace = {
        "torch": torch,
        "torch_npu": npu_module,
        "logger": logger,
        "H2DCopyTask": transport.H2DCopyTask,
    }
    # Pruning/substitution/decode-statistics branches are deliberately disabled.
    exec(compile(ast.Module(body=[helper, cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ExpertOffloadManager"], hashlib.sha256(source).hexdigest()


class Timings:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.values = {}

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            start = self.clock()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = self.clock() - start
                self.values[name] = self.values.get(name, 0) + elapsed

        return timed

    def run(self, fn, args):
        self.values.clear()
        start = self.clock()
        fn(args)
        total = self.clock() - start
        result = {
            name: self.values.get(name, 0) for name in ("observe", "choose_victims", "synchronize", "load_submit")
        }
        result["total"] = total
        result["outer"] = total - sum(result[name] for name in ("observe", "choose_victims", "synchronize"))
        return result


def make_cases(experts, capacity, topk, tokens, scenario, seed):
    rng = random.Random(seed)
    population = list(range(capacity)) if scenario == "hit" else list(range(experts))
    if scenario == "miss":
        population = list(range(capacity, experts))
    if len(population) < topk:
        raise ValueError(f"scenario={scenario} needs at least topk eligible experts")
    result = []
    for _ in range(INPUT_CASES):
        ids = [rng.sample(population, topk) for _ in range(tokens)]
        scores = []
        for _ in ids:
            row = [rng.random() for _ in range(topk)]
            total = sum(row)
            scores.append([x / total for x in row])
        result.append((torch.tensor(ids, dtype=torch.int32), torch.tensor(scores, dtype=torch.float32)))
    return result


def build_harness(args, settings, repo):
    # Runtime import keeps CPU planning usable without torch_npu installed.
    if args.mode == "npu":
        import torch_npu

        torch_npu.npu.set_device(args.device)
        stream = torch_npu.npu.Stream()
        device = f"npu:{args.device}"
    else:
        torch_npu = types.SimpleNamespace(npu=types.SimpleNamespace(stream=lambda _: contextlib.nullcontext()))
        stream = None
        device = "cpu"

    logger = logging.getLogger("update_weights_bench")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO if args.debug else logging.WARNING)
    base = repo / "vllm_ascend/expert_offload"
    transport_module = load_transport(base / "h2d_transfer.py")
    manager_class, digest = load_manager(base / "expert_offload_manager.py", torch_npu, transport_module, logger)
    manager = manager_class()
    policy_class, policy_digest = load_policy(base / "lrc_policy.py")
    manager.cache_policy = policy_class(
        num_layers=1,
        num_experts=args.experts,
        cache_size=args.cache_size,
        topk=args.topk,
        recent_window=args.recent_window,
        **{
            key.removeprefix("cache_"): settings[key]
            for key in (
                "cache_ema_beta",
                "cache_recent_weight",
                "cache_ema_weight",
                "cache_router_weight",
                "cache_age_weight",
            )
        },
    )
    manager.offload_config = types.SimpleNamespace(
        experts_pruning_enabled=False, cache_stats_log_interval=settings["cache_stats_log_interval"]
    )
    manager._stats = None
    manager._debug = args.debug
    manager.num_total_experts = args.experts
    manager.prefetch_topk = max(1, min(args.topk, settings["expert_prefetch_num"]))
    manager.load_stream = stream
    for name in ("cache_calls", "cache_requests", "cache_hits", "cache_misses"):
        setattr(manager, name, [0])
    manager.last_hit_experts = [[]]
    manager.last_miss_experts = [[]]
    layer = types.SimpleNamespace()
    loaded = []
    if args.mode == "npu":
        nbytes = int(args.weight_mib * MIB)
        manager.w13_expert_size_bytes = nbytes
        manager.w2_expert_size_bytes = nbytes
        # Byte buffers test transport/host overhead, not model numerical accuracy.
        sources = {}
        for name in ("w13", "w2"):
            host = torch.empty((args.experts, nbytes), dtype=torch.uint8, pin_memory=True)
            for eid in range(args.experts):
                host[eid].fill_(eid % 251)
            sources[name] = host
            setattr(layer, name + "_weight", torch.zeros((args.cache_size, nbytes), dtype=torch.uint8, device=device))
        manager._expert_src_storage = lambda layer_idx, eid, name: (
            sources[name].untyped_storage()[eid * nbytes : (eid + 1) * nbytes]
        )
        manager._build_quant_attr_h2d_tasks = lambda *unused: []
        transport = transport_module.TorchCopyH2DTransport()
        manager._get_h2d_transport = lambda: transport
        torch_npu.npu.synchronize()
        original_load = manager._load_expert_weights_into_slots
    else:
        original_load = lambda *unused: None
        manager._synchronize_h2d = lambda: None

    def load(layer, layer_idx, plans):
        loaded.extend(plans)
        return original_load(layer, layer_idx, plans)

    manager._load_expert_weights_into_slots = load
    timings = Timings()
    for name in ("observe", "choose_victims"):
        setattr(manager.cache_policy, name, timings.wrap(name, getattr(manager.cache_policy, name)))
    manager._synchronize_h2d = timings.wrap("synchronize", manager._synchronize_h2d)
    manager._load_expert_weights_into_slots = timings.wrap("load_submit", manager._load_expert_weights_into_slots)
    return manager, layer, timings, loaded, digest, policy_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent
    parser.add_argument("--mode", choices=("planning", "npu"), default="planning")
    parser.add_argument("--test-script", type=Path, default=repo / "script/test_graph2.sh")
    parser.add_argument("--config-source", type=Path, default=repo / "vllm_ascend/ascend_config.py")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--cache-size", type=int)
    parser.add_argument("--recent-window", type=int)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--router-scores", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scenario", choices=("hit", "mixed", "miss"), default="mixed")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Include real console debug logging (verbose)")
    parser.add_argument("--device", type=int, default=0, help="NPU index after visibility filtering")
    parser.add_argument(
        "--weight-mib", type=float, default=1, help="NPU mode: MiB per expert per weight buffer (two buffers)"
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    try:
        settings = project_settings(args.test_script, args.config_source)
        if settings["enable_multi_card"] or not settings["cache_policy_enabled"]:
            raise ValueError("This benchmark requires single-card LRC enabled")
        experts, topk = model_dimensions(args.model_config) if args.model_config else (256, 6)
        args.experts = experts if args.experts is None else args.experts
        args.topk = topk if args.topk is None else args.topk
        args.cache_size = settings["num_device_experts"] if args.cache_size is None else args.cache_size
        args.recent_window = settings["cache_recent_window"] if args.recent_window is None else args.recent_window
        args.router_scores = settings["cache_router_weight"] != 0 if args.router_scores is None else args.router_scores
        for name in ("experts", "topk", "cache_size", "recent_window", "tokens", "warmup", "iterations"):
            if not isinstance(getattr(args, name), int) or getattr(args, name) < 1:
                raise ValueError(f"{name} must be a positive integer; per-layer capacity needs --cache-size")
        if not args.topk <= args.cache_size <= args.experts:
            raise ValueError("Require topk <= cache-size <= experts")
        if args.tokens * args.topk > args.cache_size:
            raise ValueError("tokens exceeds the single-card paging threshold")
        if not math.isfinite(args.weight_mib) or args.weight_mib * MIB < 1 or args.device < 0:
            raise ValueError("Invalid weight-mib or device")
        cases = make_cases(args.experts, args.cache_size, args.topk, args.tokens, args.scenario, args.seed)
    except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
        parser.error(str(exc))

    print(f"mode={args.mode} scenario={args.scenario} prefetch={args.prefetch}", flush=True)
    print(f"experts={args.experts} cache_size={args.cache_size} topk={args.topk} tokens={args.tokens}")
    print(f"recent_window={args.recent_window} router_scores={args.router_scores} seed={args.seed}")
    print(
        f"debug={args.debug} (script debug={settings.get('moe_offload_debug')}); decode stats/pruning/substitution OFF"
    )
    print("Model dimensions: " + ("config + CLI overrides" if args.model_config else "fallback 256/6 + CLI overrides"))
    print(f"torch={torch.__version__} numpy={np.__version__}")
    if args.mode == "npu":
        print(f"TorchCopy only; synthetic bytes, no quantization; weight_mib={args.weight_mib} per buffer")
        print(
            f"Approx buffers: host={2 * args.experts * args.weight_mib:.1f} MiB; "
            f"device={2 * args.cache_size * args.weight_mib:.1f} MiB",
            flush=True,
        )
    else:
        print("PLANNING ONLY: load_submit/synchronize are no-ops; excludes transfer preparation/submission.")
    manager, layer, timings, loaded, digest, policy_digest = build_harness(args, settings, repo)
    print(f"manager_sha256={digest}\npolicy_sha256={policy_digest}")
    mapping = np.full(args.experts, -1, dtype=np.int64)
    initial = mapping.copy()
    initial[: args.cache_size] = np.arange(args.cache_size)
    samples = {name: [] for name in ("total", "observe", "choose_victims", "synchronize", "outer", "load_submit")}
    counts = []
    warmup = max(args.warmup, math.ceil(args.recent_window / args.tokens) + 1)
    for index in range(warmup + args.iterations):
        ids, scores = cases[index % len(cases)]
        mapping[:] = initial  # Restore residency outside timing; avoids becoming all-hit.
        loaded.clear()
        call_args = (ids, mapping, layer, 0, scores if args.router_scores else None, args.prefetch)
        result = timings.run(manager._update_weights, call_args)
        # Validate state and actual transfer destinations outside timing.
        expected = set(ids.reshape(-1).tolist()) - set(range(args.cache_size))
        expected_count = min(len(expected), manager.prefetch_topk) if args.prefetch else len(expected)
        if len(loaded) != expected_count or any(mapping[eid] != slot for slot, eid in loaded):
            raise RuntimeError("Invalid load count or updated mapping")
        resident = mapping[mapping >= 0]
        if len(resident) != args.cache_size or len(set(resident.tolist())) != args.cache_size:
            raise RuntimeError("Invalid cache slot ownership")
        if index >= warmup:
            for name, value in result.items():
                samples[name].append(value)
            counts.append(len(loaded))
    if args.mode == "npu":
        # Last iteration's writes must be complete and contain the intended expert bytes.
        for slot, eid in loaded:
            for name in ("w13_weight", "w2_weight"):
                if not bool((getattr(layer, name)[slot].cpu() == eid % 251).all()):
                    raise RuntimeError("Synthetic H2D content check failed")
    print(
        f"warmup={warmup} iterations={args.iterations}; loads min/mean/max="
        f"{min(counts)}/{sum(counts) / len(counts):.2f}/{max(counts)}"
    )
    print("outer = total - observe - choose_victims - synchronize (computed per call)")
    print(
        f"{'metric':26s} {'mean_us':>10s} {'p50_us':>10s} {'p95_us':>10s} "
        f"{'p99_us':>10s} {'min_us':>10s} {'max_us':>10s}"
    )
    for name, values in samples.items():
        report(name, values)
    print("load_submit is INCLUDED in outer; do not add it again.")
    print("NPU mode retains synchronization for safety, subtracting its measured duration.")
    print("Outer includes instrumentation overhead; no graph callback/GIL scheduling measurement.")
    print("Synthetic routes; fixed residency reset, evolving LRC history; no model weights loaded.")
    print("Validation passed. Console output only; no result files written.")


if __name__ == "__main__":
    main()
