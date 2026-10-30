"""Shared aggregation and offline replay: invalid metrics never improve a score."""

from collections import Counter, defaultdict
import json
import math
import random
import statistics


def stats(values):
    ordered = sorted(
        v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0
    )
    if not ordered:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


def aggregate(rows):
    arms = defaultdict(list)
    for row in rows:
        arms[row["arm"]].append(row)
    result = {}
    for name, rs in arms.items():
        reasons = Counter(reason for r in rs for reason in r.get("failures", []))
        result[name] = {
            "correct": sum(bool(r["correct"]) for r in rs),
            "total": len(rs),
            "errors": sum(not r["correct"] for r in rs),
            "failure_reasons": dict(reasons),
            "exceptions": sum("error" in r for r in rs),
            "truncated": sum(bool(r["truncated"]) for r in rs),
            "wall_all_s": stats([r["wall"] for r in rs]),
            "wall_success_s": stats([r["wall"] for r in rs if r["correct"]]),
            "tokens": stats([r.get("ctok") for r in rs if r.get("usage_valid", False)]),
            "token_observations": sum(r.get("usage_valid") is True for r in rs),
            "token_usage_missing": sum(r.get("usage_valid") is not True for r in rs),
            "ttft_s": stats([r.get("ttft") for r in rs]),
        }
    return result


def metric_row(row):
    """Retain small scalars for statistics; large raw traces live only in JSONL."""
    keys = (
        "arm",
        "correct",
        "failures",
        "truncated",
        "wall",
        "ttft",
        "usage_valid",
        "ctok",
        "error",
    )
    return {**{k: row[k] for k in keys if k in row}, "task": {"id": row["task"]["id"]}}


def verify_row(row):
    if (
        type(row.get("wall")) not in (int, float)
        or not math.isfinite(row["wall"])
        or row["wall"] < 0
    ):
        raise ValueError("invalid wall metric")
    if row.get("usage_valid") is True and (
        type(row.get("ctok")) is not int or row["ctok"] < 0
    ):
        raise ValueError("invalid token metric marked valid")
    if "error" in row:
        if row["correct"]:
            raise ValueError("exception row claims success")
        return
    if "stages" in row:
        from agent_cases import case, judge

        task = case(row["task"]["index"], row["task"]["seed"])
        if task["expected"] != row["task"]["expected"]:
            raise ValueError("corrupt task truth")
        ok = "lead" in row["stages"]
        for name, stage in row["stages"].items():
            expected = (
                {"parallel_audits": 2} if name == "planner" else task["expected"][name]
            )
            ok = ok and judge(stage.get("text", ""), expected)["correct"]
    else:
        from bench_live_support import independent_eval, score_alone, score_marker
        from exactus.adapters.in_process import InProcessExternalComputer
        from exactus.adapters.digest import HashlibHasher

        task = row["task"]
        if independent_eval(task["expr"]) != task["truth"]:
            raise ValueError("corrupt task truth")
        if row["arm"] in ("marker", "marker-loop"):
            emitted, value, parity = score_marker(
                row["output"], InProcessExternalComputer(HashlibHasher())
            )
        else:
            value, parity = score_alone(row["output"]), True
        ok = (
            value == task["truth"]
            and parity
            and row.get("finish") == "stop"
            and row.get("usage_valid") is True
        )
    if bool(row["correct"]) != bool(ok):
        raise ValueError("stored correctness disagrees with replay scorer")


def paired(rows, baseline="alone", seed=20261030):
    cases = defaultdict(dict)
    for r in rows:
        cases[r["task"]["id"]][r["arm"]] = r
    names = sorted({r["arm"] for r in rows} - {baseline})
    rng, result = random.Random(seed), {}
    for name in names:
        pairs = [
            (rs[baseline], rs[name])
            for rs in cases.values()
            if baseline in rs and name in rs
        ]
        ratios = [
            a["wall"] / b["wall"]
            for a, b in pairs
            if a["correct"] and b["correct"] and b["wall"] > 0
        ]
        samples = (
            [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(1000)]
            if ratios
            else []
        )
        samples.sort()
        result[name] = {
            "paired_tasks": len(pairs),
            "both_success": len(ratios),
            "quality_delta": statistics.fmean(
                int(b["correct"]) - int(a["correct"]) for a, b in pairs
            )
            if pairs
            else None,
            "successful_pair_speed_ratio": statistics.median(ratios)
            if ratios
            else None,
            "bootstrap_median_ci95": [samples[24], samples[974]] if samples else None,
            "warning": "latency ratio is conditional on both arms succeeding; quality/failures separate",
        }
    return result


def replay(path):
    with open(path) as source:
        header = json.loads(next(source))
        if header.get("record_type") != "manifest":
            raise ValueError("missing raw manifest")
        all_rows = [json.loads(line) for line in source if line.strip()]
        rows = [
            row
            for row in all_rows
            if row.get("record_type") not in ("warmup", "qualification_refused")
        ]
    for row in rows:
        if "stages" in row and header["manifest"].get("rubric_version") != 2:
            raise ValueError(
                "unsupported historical agent rubric; do not silently rescore quality"
            )
        verify_row(row)
    identities = [(row["arm"], row["task"]["id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate arm/task rows")
    return {
        "manifest": header["manifest"],
        "qualification_refusals": sum(
            row.get("record_type") == "qualification_refused" for row in all_rows
        ),
        "arms": aggregate(rows),
        "paired": paired(rows, baseline=header["manifest"].get("baseline", "alone")),
    }
