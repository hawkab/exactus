"""Paired C3 old-marker-vs-one-shot-plan eval. All failures stay in the denominator."""

__author__ = "Olshansky Grigory"

import argparse
import hashlib
import json
import platform
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_external_computers import idle
from plan_cases import cases
from plan_measurement import measure as measure
from plan_measurement import summary as summary


def fetch(url):
    """Read a qualification endpoint response without invoking the model."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read(1048577).decode()


def main():
    """Parse command-line options and run this module workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "before",
        "after",
        "capacity_url",
        "metrics_url",
        "before_source",
        "after_source",
        "runtime_metadata",
    ):
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=22112588)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--view", choices=("full", "schema"), default="full")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    lock = threading.Lock()
    with (args.out / "raw.jsonl").open("x") as raw:

        def capture(row):
            """Persist one observation under the shared output lock."""
            with lock:
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()

        sources = {}
        for arm in ("before", "after"):
            root = Path(getattr(args, arm + "_source")).resolve()
            h = hashlib.sha256()
            for folder in ("src", "benchmarks"):
                for path in sorted((root / folder).rglob("*.py")):
                    h.update(
                        str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
                    )
            sources[arm] = {"path": str(root), "source_tree_sha256": h.hexdigest()}
        capture(
            {
                "record_type": "manifest",
                "sources": sources,
                "python": platform.python_version(),
                "runtime_observed_not_attested": json.loads(
                    Path(args.runtime_metadata).read_text()
                ),
                "seed": args.seed,
                "concurrency": 3,
                "max_tokens": 8192,
                "thinking_override": None,
                "rounds": args.rounds,
                "warmup": args.warmup,
                "compact": args.compact,
                "input_view": args.view,
            }
        )

        def preflight():
            """Record backend activity and refuse to interfere with unrelated inference."""
            capacity, metrics = (
                json.loads(fetch(args.capacity_url)),
                fetch(args.metrics_url),
            )
            ok = idle(capacity, metrics)
            capture(
                {
                    "record_type": "preflight",
                    "idle": ok,
                    "capacity": capacity,
                    "gauges": [
                        line
                        for line in metrics.splitlines()
                        if line.startswith("vllm:num_requests_")
                    ],
                }
            )
            if not ok:
                raise RuntimeError("Spark is busy; next wave refused without inference")

        rng = random.Random(args.seed)
        try:
            if args.warmup:
                for arm in ("before", "after"):
                    preflight()
                    result = measure(
                        getattr(args, arm),
                        arm,
                        cases(99, compact=args.compact)[0],
                        capture,
                        view=args.view,
                    )
                    capture({"record_type": "warmup", "case": result})
            for index in range(args.rounds):
                items = cases(args.seed + index, compact=args.compact)
                order = ["before", "after"]
                rng.shuffle(order)
                for arm in order:
                    preflight()
                    barrier = threading.Barrier(3)

                    def run(item):
                        """Execute one synchronized workload case."""
                        barrier.wait()
                        return measure(
                            getattr(args, arm), arm, item, capture, view=args.view
                        )

                    start = time.monotonic()
                    with ThreadPoolExecutor(3) as pool:
                        for result in pool.map(run, items):
                            rows.append(result)
                            capture(result)
                            print(
                                json.dumps(
                                    {
                                        k: result[k]
                                        for k in (
                                            "arm",
                                            "case_id",
                                            "correct",
                                            "elapsed_s",
                                            "completion_tokens",
                                            "error",
                                        )
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                    capture(
                        {
                            "record_type": "wave",
                            "arm": arm,
                            "round": index,
                            "elapsed_s": time.monotonic() - start,
                        }
                    )
        finally:
            result = summary(rows)
            with (args.out / "summary.json").open("x") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
