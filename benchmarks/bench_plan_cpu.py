"""Portable 0.4 batch + caller rendering vs 0.5 plan + contracts, no model calls."""

import argparse
import json
import platform
import statistics
import time

import exactus
from exactus.adapters.worker import WorkerExternalComputer
from plan_cases import cases, gold_plan


def bind_inputs(value, inputs):
    if type(value) is list:
        return [bind_inputs(v, inputs) for v in value]
    if type(value) is dict:
        if "$input" in value:
            return inputs[value["$input"]]
        return {k: bind_inputs(v, inputs) for k, v in value.items()}
    return value


def legacy_render(value, outputs):
    # Reference fixture templates only use dict/list result selectors. This is
    # explicit client-side glue, not an extra hidden model or legacy capability.
    if type(value) is list:
        return [legacy_render(v, outputs) for v in value]
    if type(value) is dict:
        if "$ref" in value:
            result = outputs[value["$ref"]]
            for key in value.get("path", []):
                result = result[key]
            return result
        return {k: legacy_render(v, outputs) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch", "plan"), required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    if args.n < 10:
        parser.error("n must be at least 10")
    worker = WorkerExternalComputer()
    rows = []
    try:
        for seed in range(args.n):
            for item in cases(seed):
                plan = gold_plan(item)
                started = time.perf_counter()
                if args.mode == "plan":
                    result = worker.run_plan(
                        plan, inputs=item["inputs"], checks=item["checks"]
                    )
                    answer = result["answer"]
                else:
                    tasks = bind_inputs(plan["tasks"], item["inputs"])
                    result = worker.run_external_computers(tasks)
                    answer = legacy_render(
                        plan["answer"], {r["id"]: r["value"] for r in result["results"]}
                    )
                elapsed = (time.perf_counter() - started) * 1000
                ok = json.dumps(answer, sort_keys=True) == json.dumps(
                    item["expected"], sort_keys=True
                )
                rows.append({"id": item["id"], "correct": ok, "elapsed_ms": elapsed})
    finally:
        worker.close()
    times = sorted(r["elapsed_ms"] for r in rows)
    print(
        json.dumps(
            {
                "version": exactus.__version__,
                "module": exactus.__file__,
                "machine": platform.machine(),
                "mode": args.mode,
                "n": len(rows),
                "correct": sum(r["correct"] for r in rows),
                "median_ms": statistics.median(times),
                "mean_ms": statistics.mean(times),
                "p95_ms": times[int((len(times) - 1) * 0.95)],
                "max_ms": max(times),
                "max_calls": 100,
                "model_calls": 0,
                "cases": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
