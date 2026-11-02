"""Summarize paired graphs and verify their retained source evidence."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import math
import statistics
from pathlib import Path

try:
    from .operational_case import case
    from .operational_transport import content, same_json
except ImportError:
    from operational_case import case
    from operational_transport import content, same_json
try:
    from .agent_quality import action_contract
except ImportError:
    from agent_quality import action_contract
try:
    from .operational_accounting import RunAccounting
except ImportError:
    from operational_accounting import RunAccounting


def decide(facts):
    """Derive go/no-go from verified auditor facts only; unproven is no-go."""
    try:
        delivery = facts["delivery"]["rows"][0]
        capacity = facts["capacity"]
        n1_ok = capacity["n1_ok"] is True
        all_delivered = delivery[0] == delivery[2]
        return "go" if n1_ok and all_delivered else "no-go"
    except (KeyError, IndexError, TypeError):
        return "no-go"


def summarize(rows):
    """Aggregate all completed graphs and paired-success latency observations."""
    summary = {}
    for arm in ("before", "after"):
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "n": len(selected),
            "facts_decision_correct": sum(
                r["facts_decision_correct"] for r in selected
            ),
            "fact_format_action_contract_pass": sum(r["correct"] for r in selected),
            "semantic_quality": None,
            "median_wall_s": statistics.median(r["elapsed_s"] for r in selected)
            if selected
            else None,
            "median_guard_wait_s": statistics.median(
                r["guard_wait_s"] for r in selected
            )
            if selected
            else None,
            "llm_calls_completed_graphs": sum(r["llm_calls"] for r in selected),
            "completion_tokens_completed_graphs": sum(
                r["completion_tokens"] for r in selected
            )
            if selected and all(r["usage_valid"] for r in selected)
            else None,
        }
    ratios, all_ratios = [], []
    for seed in sorted({r["seed"] for r in rows}):
        pair = {r["arm"]: r for r in rows if r["seed"] == seed}
        if set(pair) != {"before", "after"} or pair["after"]["elapsed_s"] <= 0:
            continue
        ratio = pair["before"]["elapsed_s"] / pair["after"]["elapsed_s"]
        all_ratios.append(ratio)
        if all(r["correct"] for r in pair.values()):
            ratios.append(ratio)
    summary.update(
        paired_fact_format_action_success_n=len(ratios),
        paired_success_latency_ratio=statistics.median(ratios) if ratios else None,
        all_case_latency_ratio_not_success_speed=statistics.median(all_ratios)
        if all_ratios
        else None,
        quality_scope="facts, format, typed action contract; semantic quality ungraded",
    )
    return summary


def verify_graph(row):
    """Rescore source responses; stored 'correct' and expected facts are not truth."""
    if row.get("arm") not in ("before", "after"):
        raise ValueError("unknown graph arm")
    for key in ("elapsed_s", "first_wave_s", "guard_wait_s"):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid graph timing")
    tasks, expected = case(row["seed"])
    if not same_json(row["expected"], expected):
        raise ValueError("graph expected facts disagree with generated evidence")
    wave = row["wave"]
    facts, checks = {}, {}
    try:
        plan = content(wave[0])
        checks["plan_shape"] = type(plan) is dict and all(
            type(plan.get(k)) is list
            and len(plan[k]) == 3
            and all(type(v) is str and len(v) > 5 for v in plan[k])
            for k in ("checks", "rollback")
        )
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        checks["plan_shape"] = False
    for index, task in enumerate(tasks, 1):
        try:
            if wave[index].get("status") != 200:
                raise ValueError("auditor HTTP request did not succeed")
            value = (
                wave[index]["response"]["results"][0]["value"]
                if row["arm"] == "after"
                else content(wave[index])
            )
            facts[task["id"]] = value
            checks[task["id"]] = same_json(value, expected[task["id"]])
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            checks[task["id"]] = False
    decision = decide(facts)
    try:
        parsed = content(row["merge"])
        reason, actions = parsed.get("reason"), parsed.get("actions")
        checks["explanation_shape"] = type(reason) is str and len(reason) >= 30
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        checks["explanation_shape"] = False
        actions = [
            {"kind": k} for k in ("retain_pending", "maintain_reserve", "hold_rollout")
        ]
    checks["action_contract"] = action_contract(
        actions,
        {
            "retain_pending",
            "maintain_reserve",
            "hold_rollout" if decision == "no-go" else "proceed_rollout",
        },
    )
    checks["final_facts"] = same_json(facts, expected)
    checks["decision"] = decision == decide(expected)
    if checks != row["checks"] or bool(row["correct"]) != all(checks.values()):
        raise ValueError("stored correctness disagrees with source response scorer")
    if (
        not same_json(row["result"]["facts"], facts)
        or row["result"]["decision"] != decision
    ):
        raise ValueError("assembled answer disagrees with source responses")
    if row["facts_decision_correct"] != (checks["final_facts"] and checks["decision"]):
        raise ValueError("stored fact verdict disagrees with source responses")
    llm = [wave[0], row["merge"]] + (wave[1:] if row["arm"] == "before" else [])
    usages = [
        r.get("response", {}).get(
            "upstream_usage", r.get("response", {}).get("usage", {})
        )
        for r in llm
    ]
    valid = all(
        type(u) is dict
        and all(
            type(u.get(k)) is int and u[k] >= 0
            for k in ("completion_tokens", "prompt_tokens")
        )
        for u in usages
    )
    for key in ("completion_tokens", "prompt_tokens"):
        expected_tokens = sum(u[key] for u in usages) if valid else None
        if row[key] != expected_tokens:
            raise ValueError("stored usage disagrees with source responses")
    if row["usage_valid"] != valid or row["llm_calls"] != len(llm):
        raise ValueError("stored model call accounting disagrees with source responses")


def replay(path):
    """Recompute a summary from retained source observations without network calls."""
    rows, ledger = [], RunAccounting()
    with Path(path).open() as source:
        header = json.loads(next(source))
        if header.get("fixture_version") != 2:
            raise ValueError(
                "unsupported fixture/rubric version; do not silently rescore historical runs"
            )
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            ledger.record(row)
            if row.get("record_type") == "graph":
                verify_graph(row)
                rows.append(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("wave", "merge", "result")
                    }
                )
    return {**summarize(rows), "accounting": ledger.summary()}
