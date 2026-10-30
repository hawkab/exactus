"""Operational decision task, not three unrelated arithmetic prompts.

Lead plans a model-router rollout; capacity and delivery auditors inspect distinct
source records, then lead synthesizes an actionable decision using their outputs.
The rubric checks facts and safety decisions, not unrestricted reasoning quality.
"""

import json
import random

try:
    from .agent_quality import ACTION_HINT, action_contract
except ImportError:
    from agent_quality import ACTION_HINT, action_contract


def case(index, seed=20261030):
    rng = random.Random(seed + index)
    window, requests, tokens = 300, rng.randint(30, 110), rng.randint(400, 1000)
    rates = [rng.randint(110, 150), rng.randint(110, 150)]
    log = [
        {"root": "r1", "delivered": True, "computed": True},
        {"root": "r2", "delivered": False, "computed": False},
        {"root": "r2", "delivered": True, "computed": True},
        {"root": "r3", "delivered": False, "computed": True},
        {"root": "r4", "delivered": True, "computed": True},
    ]
    if index % 2:
        log.append({"root": "r5", "delivered": False, "computed": True})
    sources = {
        "traffic": {
            "window_s": window,
            "requests": requests,
            "output_tokens_each": tokens,
        },
        "sparks": [
            {"name": f"spark{i + 1}", "aggregate_tps_at_c3": rate, "slots": 3}
            for i, rate in enumerate(rates)
        ],
        "delivery_log": log,
        "policy": {
            "retry_only_on_elapsed_timeout": False,
            "stored_result_replay": True,
            "root_id_idempotency": True,
        },
    }
    roots = {r["root"] for r in log}
    delivered = {r["root"] for r in log if r["delivered"]}
    completed = {r["root"] for r in log if r["computed"]}
    required = (requests * tokens + window - 1) // window
    capacity = {
        "window_tokens": requests * tokens,
        "required_tps": required,
        "healthy_tps": sum(rates),
        "worst_failover_tps": min(rates),
        "n_plus_one": min(rates) >= required,
    }
    audit = {
        "unique_roots": len(roots),
        "duplicate_attempts": len(log) - len(roots),
        "answers_delivered": len(delivered),
        "stored_answers_to_replay": len(completed - delivered),
        "rerun_on_timeout_only": False,
    }
    expected = {
        "capacity": capacity,
        "audit": audit,
        "lead": {
            "admission_limit_required": not capacity["n_plus_one"],
            "replay_stored_answers": bool(audit["stored_answers_to_replay"]),
            "deduplicate_by_root": True,
        },
    }
    brief = (
        "Prepare a safe two-Spark model-router rollout and recovery recommendation. "
        "Capacity is aggregate tokens/s already measured at three concurrent requests; do not multiply it by slots. "
        "N+1 must survive loss of either Spark. Count delivery by unique root, not attempts. "
        "A computed but undelivered response must be replayed, not recomputed. "
        "Elapsed timeout alone does not authorize duplicate execution. Sources:\n"
        + json.dumps(sources)
    )
    schema = (
        "Return only a JSON object with facts (field names requested below), analysis (non-empty explanation), "
        "and actions (typed recommendations). "
        + ACTION_HINT
        + "Include retain_pending and maintain_reserve. On lead synthesis, admission_limit_required=true requires admission_control; "
        "replay_stored_answers=true requires replay_stored_answers; deduplicate_by_root=true requires deduplicate_by_root. "
        "Numeric facts may be decimal strings; booleans must be JSON booleans. "
        "If exact-compute markers are enabled, put intermediate calc spans inside JSON strings; "
        "do not use a calc-final because the deliverable is a structured report, not one number. "
    )
    prompts = {
        "planner": schema
        + brief
        + "\nPlan the two audits and their dependencies. facts: parallel_audits (must be 2).",
        "capacity": schema
        + brief
        + "\nYou are the capacity auditor. facts: "
        + ", ".join(capacity),
        "audit": schema
        + brief
        + "\nYou are the response-delivery auditor. facts: "
        + ", ".join(audit),
    }
    return {
        "id": f"rollout-{seed}-{index}",
        "seed": seed,
        "index": index,
        "brief": brief,
        "prompts": prompts,
        "expected": expected,
        "schema": schema,
    }


def judge(text, expected):
    content = text.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    try:
        report = json.loads(content)
        facts = report["facts"]
        errors = []
        for key, value in expected.items():
            actual = facts.get(key)
            if (
                type(value) is int
                and isinstance(actual, str)
                and actual.lstrip("-").isdigit()
            ):
                actual = int(actual)
            if type(actual) is not type(value) or actual != value:
                errors.append("fact:" + key)
        facts_correct = not errors
        if (
            not isinstance(report.get("analysis"), str)
            or len(report["analysis"].strip()) < 20
        ):
            errors.append("analysis_missing")
        required = {"retain_pending", "maintain_reserve"}
        required.update(
            k
            for k in (
                "admission_control",
                "replay_stored_answers",
                "deduplicate_by_root",
            )
            if expected.get(
                k if k != "admission_control" else "admission_limit_required"
            )
            is True
        )
        actions_ok = action_contract(report.get("actions"), required)
        if not actions_ok:
            errors.append("action_contract")
        return {
            "correct": not errors,
            "facts_correct": facts_correct,
            "action_contract_pass": actions_ok,
            "semantic_quality": None,
            "failures": errors,
            "report": report,
        }
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"correct": False, "failures": ["invalid_report"], "report": None}
