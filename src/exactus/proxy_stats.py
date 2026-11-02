"""Build aggregate computation evidence without changing upstream usage."""

from __future__ import annotations

__author__ = "Olshansky Grigory"


def response_stats(answers, mode, tool_stages, checks):
    """Summarize per-choice computation, provenance, and verification."""
    computed = [value for answer in answers for value in answer.computed]
    stats = {
        "operations": len(computed),
        "computed": computed,
        "exact_final": bool(answers) and all(a.exact_final for a in answers),
        "mode": mode,
        "protocol_complete": True,
        "generation_complete": not any(tool_stages),
        "response_complete": True,
        "usage_scope": "upstream_generation",
        "task_verified": (
            all(a.proof.verification()["task_verified"] for a in answers)
            if all(a.proof.verification()["task_verified"] is not None for a in answers)
            else None
        ),
        "verification_scope": "per_choice_audit",
        "model_final_emitted": all(a.model_final_emitted for a in answers),
    }
    if answers:
        stats["audit"] = [
            a.proof.as_dict(detail=mode == "audit")
            for a in answers
            if a.proof is not None
        ]
    if checks:
        stats["answer_checks"] = checks

        if len(checks) == 1:
            stats["answer_check"] = {k: v for k, v in checks[0].items() if k != "index"}
    return stats
