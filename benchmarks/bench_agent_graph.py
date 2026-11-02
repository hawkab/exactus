"""Real 1+2 DAG through deployed surfaces. Dry-run never calls a model."""

__author__ = "Olshansky Grigory"

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_cases import case
from agent_surface import Surface
from agent_workflow import run_graph as run_graph
from bench_manifest import load_runtime
from bench_report import aggregate, metric_row, paired


def main():
    """Parse command-line options and run this module workload."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name=raw|proxy|native@http://host:port",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument(
        "--warmup", type=int, default=1, help="excluded full graphs per arm; 0 disables"
    )
    ap.add_argument("--runtime-manifest")
    ap.add_argument("--out", default="evidence/agent-graph.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if min(args.cases, args.max_tokens) < 1 or args.warmup < 0:
        ap.error("positive cases/max-tokens and nonnegative warmup required")
    arms = {}
    for spec in args.arm:
        name, sep, target = spec.partition("=")
        kind, at, url = target.partition("@")
        if not sep or not at or name in arms:
            ap.error("invalid or duplicate arm")
        arms[name] = Surface(url, kind, args.model, args.max_tokens)
    manifest = {
        "rubric_version": 2,
        "model": args.model,
        "seed": args.seed,
        "cases": args.cases,
        "max_tokens_per_call": args.max_tokens,
        "surfaces": args.arm,
        "concurrency": 3,
        "dag": "[lead-plan,capacity,audit] -> lead-synthesis",
        "scope": "actual HTTP surfaces; one shared target model; finite facts/safety rubric",
        "ttft_scope": "not measured; all surfaces non-streaming; graph wall is consumer makespan",
        "runtime": load_runtime(args.runtime_manifest, arms),
    }
    manifest["baseline"] = next(iter(arms))
    manifest["runtime_provenance"] = "user_supplied_not_attested"
    manifest["warmup_graphs_per_arm"] = args.warmup
    manifest["cache_policy"] = (
        "engine configuration; no cache resets; separate warmup cases"
    )
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for folder in (root / "src", root / "benchmarks"):
        for path in sorted(folder.rglob("*.py")):
            digest.update(
                str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
            )
    manifest["source_tree_sha256"] = digest.hexdigest()
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if args.runtime_manifest is None:
        ap.error("live run requires --runtime-manifest")
    out = Path(args.out)
    if out.suffix != ".json" or out.exists() or out.with_suffix(".jsonl").exists():
        ap.error("fresh .json/.jsonl paths required")
    out.parent.mkdir(parents=True, exist_ok=True)
    rng, rows = random.Random(args.seed), []
    with out.with_suffix(".jsonl").open("x") as raw:
        raw.write(json.dumps({"record_type": "manifest", "manifest": manifest}) + "\n")
        raw.flush()
        for i in range(args.warmup):
            for name in arms:
                warmup = run_graph(case(i, args.seed + 1000000), arms[name])
                raw.write(
                    json.dumps(
                        {"record_type": "warmup", "arm": name, "result": warmup},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw.flush()
        for i in range(args.cases):
            task = case(i, args.seed)
            names = list(arms)
            rng.shuffle(names)
            for name in names:
                row = run_graph(task, arms[name])
                row["arm"] = name
                rows.append(metric_row(row))
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()
    summary = {
        "manifest": manifest,
        "arms": aggregate(rows),
        "paired": paired(rows, baseline=next(iter(arms))),
    }
    with out.open("x") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
