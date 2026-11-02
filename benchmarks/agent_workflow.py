"""Run dependency-ordered planning and parallel agent-audit stages."""

__author__ = "Olshansky Grigory"

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agent_cases import judge


def run_graph(task, call):
    """Execute dependency-ordered agent stages and retain each stage outcome."""
    started = time.perf_counter()
    barrier = threading.Barrier(3)

    def node(name):
        """Run one agent stage and retain its measured result."""
        barrier.wait(timeout=30)
        reply = call(task["prompts"][name])
        reply["judge"] = judge(
            reply["text"],
            {"parallel_audits": 2} if name == "planner" else task["expected"][name],
        )
        return reply

    stages = {}
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                name: pool.submit(node, name)
                for name in ("planner", "capacity", "audit")
            }
            for name, future in futures.items():
                try:
                    stages[name] = future.result()
                except Exception as exc:
                    stages[name] = {
                        "error": str(exc),
                        "partial": getattr(exc, "partial", None),
                        "judge": {"correct": False, "failures": ["exception"]},
                    }
        if all(s["judge"]["correct"] for s in stages.values()):
            prompt = (
                task["schema"]
                + task["brief"]
                + "\nYou are the lead. Both auditors finished:\n"
                + json.dumps({k: s["judge"]["report"] for k, s in stages.items()})
            )
            prompt += "\nSynthesize rollout/recovery actions. facts: " + ", ".join(
                task["expected"]["lead"]
            )
            reply = call(prompt)
            reply["judge"] = judge(reply["text"], task["expected"]["lead"])
            stages["lead"] = reply
        failures = [
            name + ":" + failure
            for name, s in stages.items()
            for failure in s["judge"]["failures"]
        ]
        if "lead" not in stages:
            failures.append("lead:not_run")
    except Exception as exc:
        stages["lead"] = {"error": str(exc), "partial": getattr(exc, "partial", None)}
        failures = ["lead:exception"]
    success = not failures
    usage = [s.get("usage", {}) for s in stages.values()]
    valid = all(
        type(u.get("completion_tokens")) is int and u["completion_tokens"] >= 0
        for u in usage
    )
    return {
        "correct": success,
        "failures": failures,
        "truncated": any(s.get("finish") == "length" for s in stages.values()),
        "wall": time.perf_counter() - started,
        "ttft": None,
        "usage_valid": valid,
        "ctok": sum(u["completion_tokens"] for u in usage) if valid else None,
        "task": {k: task[k] for k in ("id", "seed", "index", "expected")},
        "stages": stages,
        "concurrency": 3,
        "external_operations": sum(
            s.get("external_operations", 0) for s in stages.values()
        ),
        "engine_segments": sum(s.get("engine_segments", 0) for s in stages.values()),
        "semantic_quality": None,
        "quality_scope": "facts/format/typed action contract; unrestricted prose semantics ungraded",
    }
