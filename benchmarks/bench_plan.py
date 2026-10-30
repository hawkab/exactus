"""Paired C3 old-marker-vs-one-shot-plan eval. All failures stay in the denominator."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import threading
import time

from bench_external_computers import idle, post, same_json
from plan_cases import cases


def fetch(url):
    import urllib.request

    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read(1048577).decode()


def measure(base, arm, item, capture, *, view="full"):
    # Input facts and task are identical; only the selected compute strategy differs.
    request = {
        "model": "qwen3.8-flash-next-nvidia",
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": item["question"]}],
    }
    if arm == "after":
        request.update(
            compute_plan=True,
            plan_inputs=item["inputs"],
            plan_checks=item["checks"],
            plan_input_view=view,
        )
    else:
        request["messages"].append(
            {
                "role": "user",
                "content": "Original request data (JSON):\n"
                + json.dumps(
                    item["inputs"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    row = post(base + "/v1/chat/completions", request, capture=capture)
    body, actual, error = row.get("response", {}), None, row.get("error")
    try:
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("incomplete generation")
        text = choice["message"]["content"].strip()
        if text.startswith("```json\n") and text.endswith("\n```"):
            text = text[8:-4]
        actual = json.loads(text)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        error = error or str(exc)
    usage = body.get("upstream_usage")
    usage_valid = type(usage) is dict and all(
        type(usage.get(k)) is int and usage[k] >= 0
        for k in ("completion_tokens", "prompt_tokens")
    )
    return {
        "record_type": "case",
        "arm": arm,
        "case_id": item["id"],
        "actual": actual,
        "expected": item["expected"],
        "correct": error is None and same_json(actual, item["expected"]),
        "error": error,
        "elapsed_s": row["elapsed_s"],
        "usage_valid": usage_valid,
        "completion_tokens": usage["completion_tokens"] if usage_valid else None,
        "prompt_tokens": usage["prompt_tokens"] if usage_valid else None,
        "response": body,
        "input": item,
    }


def summary(rows):
    arms = {}
    for arm in ("before", "after"):
        selected = [r for r in rows if r["arm"] == arm]
        arms[arm] = {
            "n": len(selected),
            "correct": sum(r["correct"] for r in selected),
            "all_median_s": statistics.median(r["elapsed_s"] for r in selected)
            if selected
            else None,
            "completion_tokens": sum(r["completion_tokens"] for r in selected)
            if selected and all(r["usage_valid"] for r in selected)
            else None,
            "usage_missing": sum(not r["usage_valid"] for r in selected),
        }
    ratios = []
    for ident in sorted({r["case_id"] for r in rows}):
        pair = {r["arm"]: r for r in rows if r["case_id"] == ident}
        if len(pair) == 2 and all(r["correct"] for r in pair.values()):
            ratios.append(pair["before"]["elapsed_s"] / pair["after"]["elapsed_s"])
    return {
        "arms": arms,
        "paired_success_n": len(ratios),
        "paired_success_latency_ratio_median": statistics.median(ratios)
        if ratios
        else None,
        "latency_caveat": "C3 HTTP end-to-end latency includes queueing; not GPU decode throughput. Ratio is conditional on paired success.",
        "quality_caveat": "Generated operational fixtures; exact facts/JSON types only, not general software reasoning quality.",
    }


def main():
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
