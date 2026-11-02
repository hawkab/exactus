"""Build reproducible arithmetic schedules and source provenance."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import hashlib
import json
import platform
import random
from pathlib import Path

from arithmetic_cli import arguments
from bench_live_support import SEED, tasks
from bench_manifest import load_runtime, source_provenance
from bench_report import replay

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer


def prepare_run(harness):
    """Build the benchmark schedule, source manifest, and replaceable arm functions."""
    ap, args = arguments()
    if args.replay:
        print(json.dumps(replay(args.replay), ensure_ascii=False, indent=2))
        return None
    if min(args.runs, args.n, args.max_tokens) < 1 or args.warmup < 0:
        ap.error("runs/n/max-tokens must be positive and warmup nonnegative")
    external_computer = InProcessExternalComputer(HashlibHasher())
    functions = {
        "alone": lambda t: harness.run_alone(args.port, args.model, t, args.max_tokens),
        "marker": lambda t: harness.run_marker(
            args.port, args.model, t, external_computer, args.max_tokens
        ),
        "marker-loop": lambda t: harness.run_marker_loop(
            args.port, args.model, t, external_computer, args.max_tokens
        ),
        "toolcall": lambda t: harness.run_toolcall(
            args.port, args.model, t, args.max_tokens
        ),
    }
    names = args.arms.split(",")
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in functions for n in names)
    ):
        ap.error("unknown or duplicate arm")
    rng, schedule = random.Random(SEED), []
    for task in tasks(args.runs * args.n):
        order = names[:]
        rng.shuffle(order)
        schedule.extend((name, task) for name in order)
    root = Path(__file__).resolve().parents[1]
    provenance = source_provenance(root)
    source_hash = hashlib.sha256()
    for folder in (root / "src", root / "benchmarks"):
        for path in sorted(folder.rglob("*.py")):
            source_hash.update(
                str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
            )
    manifest = {
        **provenance,
        "model": args.model,
        "seed": SEED,
        "source_tree_sha256": source_hash.hexdigest(),
        "python": platform.python_version(),
        "max_tokens_per_task": args.max_tokens,
        "unique_tasks": args.n * args.runs,
        "arm_task_observations": len(schedule),
        "http_calls": "counted from raw segments, not arm-task observations",
        "concurrency": 1,
        "warmup_calls_per_arm": args.warmup,
        "scope": "HTTP protocol microbenchmark; NOT native/MTP or agent graph",
        "ttft_scope": "first raw model content, not non-stream proxy visibility",
        "runtime": load_runtime(args.runtime_manifest, names),
        "runtime_provenance": "user_supplied_not_attested",
    }
    if args.dry_run:
        print(json.dumps(manifest))
        return None
    if args.runtime_manifest is None:
        ap.error(
            "live evidence requires --runtime-manifest; use --dry-run for offline smoke"
        )
    out = Path(args.out)
    if out.suffix != ".json":
        ap.error("--out must end in .json; raw evidence is a separate .jsonl")
    if out.exists() or out.with_suffix(".jsonl").exists():
        ap.error("fresh output paths required; existing evidence is never overwritten")
    if not args.capacity_url or not args.engine_metrics_url:
        ap.error("live benchmark requires --capacity-url and --engine-metrics-url")
    return args, manifest, schedule, functions, out, names
