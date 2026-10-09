#!/usr/bin/env python3
"""CPU-only LRC latency benchmark. Prints to stdout; creates no output files.

Run from the repository:
    python -B bench_lrc_policy.py
    python -B bench_lrc_policy.py --model-config /models/model/config.json
    python -B bench_lrc_policy.py --tokens 3
"""
import argparse
import ast
import hashlib
import json
import math
import platform
import random
import shlex
import statistics
import sys
import time
import types
from pathlib import Path

INPUT_CASES = 256
MODULE_NAME = "_standalone_lrc_benchmark"


def project_settings(script_path, config_path):
    """Read literal configuration only; never execute a shell script or import vLLM."""
    settings = {}
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    config_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "ExpertOffloadConfig")
    for node in config_class.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_defaults"
            for target in node.targets
        ):
            settings = ast.literal_eval(node.value)
            break
    if not settings:
        raise ValueError("Cannot read ExpertOffloadConfig._defaults")
    source = script_path.read_text(encoding="utf-8")
    # Preserve comment newlines: joining a commented trailing backslash can
    # accidentally comment out the following active command.
    words = [word for word in shlex.split(source, comments=True) if word.strip()]
    configs = []
    for index, word in enumerate(words):
        if word == "--additional-config":
            configs.append(json.loads(words[index + 1]))
        elif word.startswith("--additional-config="):
            configs.append(json.loads(word.split("=", 1)[1]))
    if len(configs) != 1:
        raise ValueError("Expected exactly one active literal --additional-config")
    settings.update(configs[0].get("expert_offload_config", {}))
    return settings


def model_dimensions(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    configs = [config]
    if isinstance(config.get("text_config"), dict):
        configs.append(config["text_config"])

    def lookup(keys):
        for item in configs:
            for key in keys:
                value = item.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        raise ValueError(f"Model config has no supported field among {keys}")

    return (lookup(("n_routed_experts", "num_experts", "num_local_experts")),
            lookup(("num_experts_per_tok", "num_experts_per_token")))


def load_policy(path):
    # Avoid importing vllm_ascend/__init__.py or writing __pycache__.
    source = path.read_bytes()
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = str(path)
    sys.modules[MODULE_NAME] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.LRCExpertCachePolicy, hashlib.sha256(source).hexdigest()


def percentile(values, q):
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def report(name, samples):
    values = sorted(value / 1000 for value in samples)
    print(
        f"{name:26s} {statistics.mean(values):10.3f} "
        f"{percentile(values, .50):10.3f} {percentile(values, .95):10.3f} "
        f"{percentile(values, .99):10.3f} {values[0]:10.3f} {values[-1]:10.3f}",
        flush=True,
    )


def measure(fn, cases, warmup, iterations):
    # Input selection, result validation and output are outside timed intervals.
    for index in range(warmup):
        fn(*cases[index % len(cases)])
    samples = []
    result = None
    for index in range(iterations):
        args = cases[index % len(cases)]
        start = time.perf_counter_ns()
        result = fn(*args)
        stop = time.perf_counter_ns()
        samples.append(stop - start)
    return samples, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent
    parser.add_argument(
        "--policy-file", type=Path,
        default=Path(__file__).resolve().parent
        / "vllm_ascend/expert_offload/lrc_policy.py",
    )
    parser.add_argument("--test-script", type=Path, default=repo / "script/test_graph2.sh")
    parser.add_argument("--config-source", type=Path, default=repo / "vllm_ascend/ascend_config.py")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--experts", type=int, help="Override model expert count")
    parser.add_argument("--cache-size", type=int, help="Override launch-script slot count")
    parser.add_argument("--topk", type=int, help="Override model top-k")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--recent-window", type=int)
    parser.add_argument("--router-scores", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--batch-victims", type=int, default=None,
                        help="Override count; default is the actual synthetic cache-miss count")
    args = parser.parse_args()
    try:
        settings = project_settings(args.test_script, args.config_source)
        if settings.get("enable_multi_card"):
            parser.error("This observe benchmark models the single-card path, not observe_global_counts")
        if args.model_config:
            model_experts, model_topk = model_dimensions(args.model_config)
        else:
            model_experts, model_topk = 256, 6
        args.experts = model_experts if args.experts is None else args.experts
        args.topk = model_topk if args.topk is None else args.topk
        if args.cache_size is None:
            capacity = settings["num_device_experts"]
            if not isinstance(capacity, int):
                parser.error("Per-layer capacities require an explicit --cache-size for the tested layer")
            args.cache_size = capacity
        if args.recent_window is None:
            args.recent_window = settings["cache_recent_window"]
        if args.router_scores is None:
            args.router_scores = settings["cache_router_weight"] != 0
    except (OSError, ValueError, KeyError, IndexError, StopIteration) as exc:
        parser.error(str(exc))
    for key in ("experts", "cache_size", "topk", "tokens",
                "recent_window", "warmup", "iterations"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.cache_size > args.experts or args.topk > args.experts:
        parser.error("cache-size and topk must not exceed experts")
    if args.batch_victims is not None and args.batch_victims < 0:
        parser.error("batch-victims must be nonnegative")
    if args.tokens > args.cache_size // args.topk:
        parser.error("tokens exceeds the single-card paging threshold; use a smaller value")
    if not args.policy_file.is_file():
        parser.error(f"Policy file not found: {args.policy_file}")
    try:
        policy_class, digest = load_policy(args.policy_file)
    except ModuleNotFoundError as exc:
        parser.exit(1, f"Missing dependency: {exc.name}. Use an environment with NumPy.\n")

    import numpy as np

    rng = random.Random(args.seed)
    observe_cases = []
    for _ in range(INPUT_CASES):
        ids = [rng.sample(range(args.experts), args.topk)
               for _ in range(args.tokens)]
        scores = None
        if args.router_scores:
            scores = []
            for _ in ids:
                row = [rng.random() for _ in range(args.topk)]
                total = sum(row)
                scores.append([value / total for value in row])
        observe_cases.append((0, ids, scores))

    def make_policy():
        return policy_class(
            num_layers=1, num_experts=args.experts,
            cache_size=args.cache_size, topk=args.topk,
            recent_window=args.recent_window,
            ema_beta=settings["cache_ema_beta"],
            recent_weight=settings["cache_recent_weight"],
            ema_weight=settings["cache_ema_weight"],
            router_weight=settings["cache_router_weight"],
            age_weight=settings["cache_age_weight"],
        )

    # One evolving state for observe; a separate warmed state for eviction.
    observer = make_policy()
    chooser = make_policy()
    warmup = max(args.warmup, math.ceil(args.recent_window / args.tokens) + 1)
    for index in range(warmup):
        chooser.observe(*observe_cases[index % len(observe_cases)])
    resident = rng.sample(range(args.experts), args.cache_size)
    slot_owner = dict(enumerate(resident))
    resident_set = set(resident)
    victim_cases = []
    for _, ids, _ in observe_cases:
        needed = {eid for row in ids for eid in row}
        misses = len(needed - resident_set)
        count = misses if args.batch_victims is None else args.batch_victims
        if count > len(resident_set - needed):
            parser.error("batch-victims exceeds eligible residents for a generated case")
        victim_cases.append((0, slot_owner, needed, count))

    # Correctness checks are outside timing.
    checker = make_policy()
    route_case = observe_cases[0]
    expected = {expert for row in route_case[1] for expert in row}
    if checker.observe(*route_case) != expected:
        raise RuntimeError("observe returned unexpected expert IDs")
    if not hasattr(chooser, "choose_victims"):
        parser.error("This policy version has no choose_victims method")
    for case in victim_cases:
        victims = chooser.choose_victims(*case)
        if (len(victims) != case[3] or len(set(victims)) != len(victims)
                or not set(victims) <= resident_set - case[2]):
            raise RuntimeError("choose_victims returned invalid experts")
    if chooser.choose_victims(0, slot_owner, resident_set, 1) != []:
        raise RuntimeError("All-protected case must return an empty list")
    if chooser.choose_victims(0, slot_owner, set(), 0) != []:
        raise RuntimeError("Zero-count case must return an empty list")

    print("LRC CPU microbenchmark (all latency columns are microseconds)")
    print(f"Policy: {args.policy_file.resolve()}")
    print(f"SHA256: {digest}")
    print(f"Launch script (read only): {args.test_script.resolve()}")
    print(f"Model config: {args.model_config or 'not provided; fallback experts=256, topk=6 unless overridden'}")
    print(f"Python: {platform.python_version()}  NumPy: {np.__version__}")
    print(f"Platform: {platform.system()} / {platform.machine()}")
    print(f"experts={args.experts} cache_size={args.cache_size} topk={args.topk} "
          f"tokens/call={args.tokens} recent_window={args.recent_window}")
    print(f"router_scores={args.router_scores} (synthetic normalized scores)")
    print("LRC weights:", {key: settings[key] for key in (
        "cache_ema_beta", "cache_recent_weight", "cache_ema_weight",
        "cache_router_weight", "cache_age_weight")})
    counts = [case[3] for case in victim_cases]
    protected_counts = [len(case[2] & resident_set) for case in victim_cases]
    print(f"count mode={'actual synthetic misses' if args.batch_victims is None else 'fixed override'}; "
          f"count min/mean/max={min(counts)}/{statistics.mean(counts):.2f}/{max(counts)}")
    print(f"protected resident min/max={min(protected_counts)}/{max(protected_counts)}")
    print(f"warmup={warmup} measured_calls={args.iterations} seed={args.seed}")
    print("Eviction uses fixed resident slots and warmed statistics; no H2D or real eviction.")
    print("Observe updates its state each call; choose_victims selects a batch per call.")
    print("Uniform random routes are synthetic, not a replay of model routing/cache history.")
    print(f"{'function':26s} {'mean_us':>10s} {'p50_us':>10s} "
          f"{'p95_us':>10s} {'p99_us':>10s} {'min_us':>10s} {'max_us':>10s}")

    def noop():
        return None

    baseline, _ = measure(noop, [()], warmup, args.iterations)
    report("empty_call_baseline", baseline)
    samples, _ = measure(observer.observe, observe_cases, warmup, args.iterations)
    report("observe", samples)
    samples, _ = measure(chooser.choose_victims, victim_cases, warmup, args.iterations)
    report("choose_victims", samples)
    print("Baseline is diagnostic only; it is not subtracted.")
    print("Results exclude input generation, .tolist(), callbacks, copies and synchronization.")
    print("No profiler, CPU pinning or GC setting changes are applied.")


if __name__ == "__main__":
    main()
