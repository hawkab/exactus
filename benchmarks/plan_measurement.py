"""Measure and summarize paired one-shot compute-plan requests."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import statistics

from bench_external_computers import post, same_json


def measure(base, arm, item, capture, *, view="full"):
    """Measure one compute strategy using identical task inputs and exact answer checks."""
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
    """Summarize recorded observations without treating missing usage as zero."""
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
