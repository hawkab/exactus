"""Real 1+2 DAG through deployed surfaces. Dry-run never calls a model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_cases import case, judge
from agent_surface import Surface
from bench_report import aggregate, paired, metric_row
from bench_manifest import load_runtime


def run_graph(task, call):
    started = time.perf_counter()
    barrier = threading.Barrier(3)

    def node(name):
        barrier.wait(timeout=30)
        reply = call(task["prompts"][name])
        reply["judge"] = judge(
            reply["text"],
            {"parallel_audits": 2} if name == "planner" else task["expected"][name],
        )
        return reply

    stages = {}
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                name: pool.submit(node, name)
                for name in ("planner", "capacity", "audit")
            }
            for name, future in futures.items():
                try:
                    stages[name] = future.result()
                except Exception as exc:
                    stages[name] = {
                        "error": str(exc),
                        "partial": getattr(exc, "partial", None),
                        "judge": {"correct": False, "failures": ["exception"]},
                    }
        if all(s["judge"]["correct"] for s in stages.values()):
            prompt = (
                task["schema"]
                + task["brief"]
                + "\nYou are the lead. Both auditors finished:\n"
                + json.dumps({k: s["judge"]["report"] for k, s in stages.items()})
            )
            prompt += "\nSynthesize rollout/recovery actions. facts: " + ", ".join(
                task["expected"]["lead"]
            )
            reply = call(prompt)
            reply["judge"] = judge(reply["text"], task["expected"]["lead"])
            stages["lead"] = reply
        failures = [
            name + ":" + failure
            for name, s in stages.items()
            for failure in s["judge"]["failures"]
        ]
        if "lead" not in stages:
            failures.append("lead:not_run")
    except Exception as exc:
        stages["lead"] = {"error": str(exc), "partial": getattr(exc, "partial", None)}
        failures = ["lead:exception"]
    success = not failures
    usage = [s.get("usage", {}) for s in stages.values()]
    valid = all(
        type(u.get("completion_tokens")) is int and u["completion_tokens"] >= 0
        for u in usage
    )
    return {
        "correct": success,
        "failures": failures,
        "truncated": any(s.get("finish") == "length" for s in stages.values()),
        "wall": time.perf_counter() - started,
        "ttft": None,
        "usage_valid": valid,
        "ctok": sum(u["completion_tokens"] for u in usage) if valid else None,
        "task": {k: task[k] for k in ("id", "seed", "index", "expected")},
        "stages": stages,
        "concurrency": 3,
        "external_operations": sum(
            s.get("external_operations", 0) for s in stages.values()
        ),
        "engine_segments": sum(s.get("engine_segments", 0) for s in stages.values()),
        "semantic_quality": None,
        "quality_scope": "facts/format/typed action contract; unrestricted prose semantics ungraded",
    }


def main():
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
