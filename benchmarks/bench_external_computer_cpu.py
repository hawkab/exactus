"""External computer CPU probe via PYTHONPATH; no model calls."""

import argparse
import json
import statistics
import time

from exactus.adapters.worker import WorkerExternalComputer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    if args.n < 10:
        parser.error("n >= 10 required")
    worker = (
        WorkerExternalComputer()
    )  # actual recycle policy, not an infinite warm worker
    try:
        worker.evaluate("17*19")
        latencies = []
        for n in range(args.n):
            start = time.perf_counter()
            result = worker.evaluate(f"({n}+17)*19")
            latencies.append((time.perf_counter() - start) * 1000)
            assert result.integer == (n + 17) * 19
        print(
            json.dumps(
                {
                    "n": args.n,
                    "median_ms": statistics.median(latencies),
                    "mean_ms": statistics.mean(latencies),
                    "p95_ms": sorted(latencies)[int((args.n - 1) * 0.95)],
                    "max_ms": max(latencies),
                    "max_calls": 100,
                    "module": __import__("exactus").__file__,
                    "latencies_ms": latencies,
                }
            )
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
