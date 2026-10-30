"""CPU-only external computer qualification; never imports torch or calls an LLM."""

import json
import platform
import statistics
import time

import exactus
from exactus.adapters.worker import WorkerExternalComputer
from bench_external_computers import case, same_json


def main():
    worker = WorkerExternalComputer()
    times, rows = [], []
    try:
        for seed in range(7403, 7503):
            tasks, expected = case(seed)
            started = time.perf_counter()
            result = worker.run_external_computers(tasks)
            elapsed = (time.perf_counter() - started) * 1000
            times.append(elapsed)
            actual = {r["id"]: r["value"] for r in result["results"]}
            rows.append(
                {
                    "seed": seed,
                    "correct": same_json(actual, expected),
                    "elapsed_ms": elapsed,
                    "result": result,
                    "expected": expected,
                }
            )
    finally:
        worker.close()
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "version": exactus.__version__,
                "module": exactus.__file__,
                "n": len(rows),
                "correct": sum(r["correct"] for r in rows),
                "median_ms": statistics.median(times),
                "p95_ms": sorted(times)[94],
                "max_ms": max(times),
                "model_calls": 0,
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
