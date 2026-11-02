"""Paired operational 1+2 graph: LLM audits versus external computer offload.

NOT a decode-kernel benchmark. The candidate deliberately replaces two
mechanical auditor generations with explicit external computer executions; both arms use
the same LLM for planning and final explanation. Golden facts come from loops
over generated evidence, not from the external computer interpreter or SQLite. Every raw
request/reply, error and measured stage is retained. No hidden repair retry.
"""

__author__ = "Olshansky Grigory"

import hashlib
import json
import platform
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

if __package__:
    from . import operational_accounting as accounting
    from . import operational_case as fixtures
    from . import operational_cli as cli
    from . import operational_idle as readiness
    from . import operational_results as results
    from . import operational_transport as transport
    from . import operational_workflow as workflow
else:
    import operational_accounting as accounting
    import operational_case as fixtures
    import operational_cli as cli
    import operational_idle as readiness
    import operational_results as results
    import operational_transport as transport
    import operational_workflow as workflow

RunAccounting = accounting.RunAccounting
case = fixtures.case
arguments = cli.arguments
idle = readiness.idle
summarize, verify_graph, replay, decide = (
    results.summarize,
    results.verify_graph,
    results.replay,
    results.decide,
)
post, chat, content, same_json = (
    transport.post,
    transport.chat,
    transport.content,
    transport.same_json,
)
run_graph = workflow.graph


def graph(url, arm, seed, model, capture=None, before_merge=None):
    """Execute a paired graph with replaceable HTTP dependencies."""
    return run_graph(url, arm, seed, model, capture, before_merge, chat=chat, post=post)


def main():
    """Parse command-line options and run this module workload."""
    parser, args = arguments(replay)
    if args.replay:
        return
    from bench_manifest import load_runtime, source_provenance

    runtime = load_runtime(args.runtime_metadata)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("raw.jsonl", "summary.json")):
        parser.error("fresh output paths required")
    rng, rows, ledger = random.Random(args.seed), [], RunAccounting()
    with (out / "raw.jsonl").open("x") as raw:
        lock = threading.Lock()

        def capture(row):
            """Persist one observation under the shared output lock."""
            with lock:
                ledger.record(row)
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()

        def gate(scoped, stage):
            """Record shared-backend readiness before starting benchmark inference."""
            while True:
                with urllib.request.urlopen(args.capacity_url, timeout=5) as response:
                    capacity = json.load(response)
                with urllib.request.urlopen(
                    args.engine_metrics_url, timeout=5
                ) as response:
                    metrics = response.read(1048577).decode("utf-8")
                clear = len(metrics.encode()) <= 1048576 and idle(capacity, metrics)
                scoped(
                    {
                        "record_type": "preflight" if clear else "wait_busy",
                        "stage": stage,
                        "at": time.time(),
                        "capacity": capacity,
                        "engine_request_gauges": [
                            s
                            for s in metrics.splitlines()
                            if s.startswith("vllm:num_requests_")
                        ],
                    }
                )
                if clear:
                    return
                if not args.wait_idle:
                    raise RuntimeError("qualification refused: backend is busy")
                time.sleep(15)

        hashes = {}
        for arm in ("before", "after"):
            root = Path(getattr(args, arm + "_source")).resolve()
            digest = hashlib.sha256()
            for folder in ("src", "benchmarks"):
                for path in sorted((root / folder).rglob("*.py")):
                    digest.update(
                        str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
                    )
            hashes[arm] = {
                "root": str(root),
                "source_tree_sha256": digest.hexdigest(),
                **source_provenance(root),
            }
        capture(
            {
                "record_type": "manifest",
                "fixture_version": 2,
                "sources": hashes,
                "seed": args.seed,
                "rounds": args.rounds,
                "model": args.model,
                "python": platform.python_version(),
                "max_tokens": "server default",
                "runtime_declared_not_attested": runtime,
                "scope": "provided-program offload workflow, not program selection or GPU C3",
                "harness_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        try:
            for index in range(args.rounds):
                order = ["before", "after"]
                rng.shuffle(order)
                for arm in order:
                    gid = uuid.uuid4().hex

                    def scoped(row):
                        """Attach graph identity and workload provenance to a captured observation."""
                        capture(
                            {
                                **row,
                                "graph_id": gid,
                                "seed": args.seed + index,
                                "arm": arm,
                            }
                        )

                    gate(scoped, "before_graph")
                    scoped({"record_type": "graph_started", "at": time.time()})
                    row = graph(
                        getattr(args, arm),
                        arm,
                        args.seed + index,
                        args.model,
                        capture=scoped,
                        before_merge=lambda: gate(scoped, "before_merge"),
                    )
                    scoped(row)
                    rows.append(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("wave", "merge", "result")
                        }
                    )
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        except BaseException as exc:
            capture({"record_type": "run_aborted", "error": type(exc).__name__})
            raise
        finally:
            result = {**summarize(rows), "accounting": ledger.summary()}
            with (out / "summary.json").open("x") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
