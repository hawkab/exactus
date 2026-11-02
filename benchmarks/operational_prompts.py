"""Build matched planning, auditing, and synthesis requests."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json

try:
    from .agent_quality import ACTION_HINT
except ImportError:
    from agent_quality import ACTION_HINT


def merge_prompt(facts, plan):
    """Build the final synthesis request from auditor facts and the planning output."""
    return (
        'You are the lead engineer. Condense the canary auditors\' reports into the reasoning and actions for a rollout decision. The verified facts and the decision are assembled by the harness. Return JSON {"reason": "<explanation>", "actions": [...]}. '
        + ACTION_HINT
        + "Include retain_pending, maintain_reserve and hold_rollout for no-go or proceed_rollout for go. Explain the decision implied by the reports: go only if n1_ok=true and every root received an answer (roots=delivered_roots), otherwise no-go. Never prescribe clearing or dropping pending work without a durable recovery/replay procedure, and never propose disabling the N+1 reserve without a verified transition plan.\nReports: "
        + json.dumps(facts, ensure_ascii=False)
        + "\nPlan: "
        + json.dumps(plan, ensure_ascii=False)
    )


def audit_prompt(task):
    """Build the unassisted auditor request from an explicit task and its data."""
    return (
        'Compute the result of the following explicitly specified program on the given data. This is a canary model-router audit; do not rewrite the program. SQL uses SQLite with checked int_sub(a,b)=a-b; python_expr uses Python integer semantics. Return only the result JSON: SQL {"columns":[...],"rows":[...]}, Python — the expression value itself.\n'
        + json.dumps(task, ensure_ascii=False)
    )


def planning_prompt(tasks, seed):
    """Build the planning request from the same evidence used by both benchmark arms."""
    return f"""You are the lead engineer. Plan a canary model-router on two Spark nodes with C3 (three concurrent LLM requests), re-delivery and N+1. Evidence batch {seed}; nodes={tasks[1]["inputs"]["nodes"]}, window=1800s, demand_tokens={tasks[1]["inputs"]["tokens"]}. Return JSON {{"checks":[...],"rollback":[...]}}. Exactly three specific short checks in each array, no invented measurements."""
