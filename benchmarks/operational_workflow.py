"""Execute matched planning and parallel-auditing graphs."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
    from .operational_prompts import audit_prompt, merge_prompt, planning_prompt
    from .operational_results import decide
except ImportError:
    from operational_prompts import audit_prompt, merge_prompt, planning_prompt
    from operational_results import decide


def graph(url, arm, seed, model, capture=None, before_merge=None, *, chat, post):
    """Run the matched planner, parallel auditors, and final synthesis workflow."""
    tasks, expected = case(seed)
    barrier = threading.Barrier(3)

    def launch(index):
        """Start one synchronized planner or auditor stage."""
        barrier.wait()
        if index == 0:
            return chat(url, planning_prompt(tasks, seed), model, capture=capture)
        task = tasks[index - 1]
        if arm == "after":
            return post(
                url + "/v1/exact/external-computers", {"tasks": [task]}, capture=capture
            )
        return chat(url, audit_prompt(task), model, capture=capture)

    started = time.monotonic()
    with ThreadPoolExecutor(3) as pool:
        wave = list(pool.map(launch, range(3)))
    first_wave_s = time.monotonic() - started
    facts, failures, checks = ({}, [], {})
    try:
        plan = content(wave[0])
        if type(plan) is not dict:
            raise ValueError("plan must be an object")
        checks["plan_shape"] = all(
            (
                isinstance(plan.get(k), list)
                and len(plan[k]) == 3
                and all((isinstance(x, str) and len(x) > 5 for x in plan[k]))
                for k in ("checks", "rollback")
            )
        )
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        plan = None
        failures.append("plan: " + str(exc))
        checks["plan_shape"] = False
    for index, task in enumerate(tasks, 1):
        try:
            if wave[index].get("status") != 200:
                raise ValueError("auditor HTTP request did not succeed")
            value = (
                wave[index]["response"]["results"][0]["value"]
                if arm == "after"
                else content(wave[index])
            )
            facts[task["id"]] = value
            checks[task["id"]] = same_json(value, expected[task["id"]])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            failures.append(task["id"] + ": " + str(exc))
            checks[task["id"]] = False
    decision = decide(facts)
    gate_started = time.monotonic()
    if before_merge is not None:
        before_merge()
    guard_wait_s = time.monotonic() - gate_started
    merge = chat(url, merge_prompt(facts, plan), model, capture=capture)
    reason, actions, analysis_complete = ("", [], False)
    try:
        parsed = content(merge)
        if type(parsed) is not dict:
            raise ValueError("explanation must be an object")
        reason = parsed.get("reason")
        actions = parsed.get("actions")
        checks["explanation_shape"] = isinstance(reason, str) and len(reason) >= 30
        analysis_complete = checks["explanation_shape"]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        checks["explanation_shape"] = False
        reason = "analysis_incomplete: " + str(exc)
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
    result = {
        "facts": facts,
        "decision": decision,
        "reason": reason,
        "actions": actions,
        "analysis_complete": analysis_complete,
    }
    llm_rows = [wave[0], merge] + (wave[1:] if arm == "before" else [])
    sampled, prompt, usages_valid = (0, 0, True)
    for row in llm_rows:
        usage = row.get("response", {}).get("upstream_usage", {})
        if any(
            (
                type(usage.get(key)) is not int or usage[key] < 0
                for key in ("completion_tokens", "prompt_tokens")
            )
        ):
            usages_valid = False
        else:
            sampled += usage["completion_tokens"]
            prompt += usage["prompt_tokens"]
    return {
        "record_type": "graph",
        "seed": seed,
        "arm": arm,
        "elapsed_s": time.monotonic() - started,
        "first_wave_s": first_wave_s,
        "guard_wait_s": guard_wait_s,
        "semantic_quality": None,
        "quality_scope": "exact facts, JSON format and typed action contract; prose semantics ungraded",
        "checks": checks,
        "correct": all(checks.values()) and (not failures),
        "facts_decision_correct": checks["final_facts"] and checks["decision"],
        "analysis_complete": analysis_complete,
        "failures": failures,
        "llm_calls": len(llm_rows),
        "completion_tokens": sampled if usages_valid else None,
        "prompt_tokens": prompt if usages_valid else None,
        "usage_valid": usages_valid,
        "expected": expected,
        "result": result,
        "wave": wave,
        "merge": merge,
    }
