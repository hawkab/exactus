#!/usr/bin/env python3
"""Paired operational 1+2 graph: LLM audits versus external computer offload.

NOT a decode-kernel benchmark. The candidate deliberately replaces two
mechanical auditor generations with explicit external computer executions; both arms use
the same LLM for planning and final explanation. Golden facts come from loops
over generated evidence, not from the external computer interpreter or SQLite. Every raw
request/reply, error and measured stage is retained. No hidden repair retry.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import platform
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid

try:
    from .agent_quality import ACTION_HINT, action_contract
except ImportError:
    from agent_quality import ACTION_HINT, action_contract


def case(seed):
    rng = random.Random(seed)
    runs = []
    for i in range(45):
        root = f"req{i:03}"
        delivered = seed % 4 in (0, 1) or i % 7 != 0
        attempts = 2 if i % 4 == 0 else 1
        for attempt in range(1, attempts + 1):
            runs.append(
                [
                    root,
                    attempt,
                    int(delivered and attempt == attempts),
                    rng.randrange(300, 9000),
                ]
            )
    nodes = [
        ["spark1", rng.randrange(47, 64), rng.randrange(51, 68)],
        ["spark2", rng.randrange(52, 69), rng.randrange(45, 62)],
    ]
    tasks = [
        {
            "id": "delivery",
            "kind": "sql_read",
            "query": "SELECT count(DISTINCT root) AS roots, int_sub(count(*),count(DISTINCT root)) AS retries, count(DISTINCT CASE WHEN delivered=1 THEN root END) AS delivered_roots, sum(CASE WHEN delivered=1 THEN tokens ELSE 0 END) AS delivered_tokens FROM runs",
            "tables": {
                "runs": {
                    "columns": ["root", "attempt", "delivered", "tokens"],
                    "rows": runs,
                }
            },
        },
        {
            "id": "capacity",
            "kind": "python_expr",
            "expression": '{"healthy_tps": sum(n[1]+n[2] for n in nodes), "worst_n1_tps": min(n[1]+n[2] for n in nodes), "required_tps": (tokens+window-1)//window, "n1_ok": min(n[1]+n[2] for n in nodes) >= (tokens+window-1)//window}',
            "inputs": {
                "nodes": nodes,
                "tokens": (160000 if seed % 4 in (0, 2) else 300000)
                + (seed % 17) * 400,
                "window": 1800,
            },
        },
    ]
    # Independently specified golden values: no execution of the candidate spec.
    roots, delivered, total_tokens = set(), set(), 0
    for root, attempt, ok, tokens in runs:
        roots.add(root)
        if ok:
            delivered.add(root)
            total_tokens += tokens
    rates = [n[1] + n[2] for n in nodes]
    demand = (tasks[1]["inputs"]["tokens"] + 1799) // 1800
    expected = {
        "delivery": {
            "columns": ["roots", "retries", "delivered_roots", "delivered_tokens"],
            "rows": [
                [len(roots), len(runs) - len(roots), len(delivered), total_tokens]
            ],
        },
        "capacity": {
            "healthy_tps": sum(rates),
            "worst_n1_tps": min(rates),
            "required_tps": demand,
            "n1_ok": min(rates) >= demand,
        },
    }
    return tasks, expected


def post(url, payload, timeout=600, capture=None):
    started = time.monotonic()
    request_id = "exactus-external-computer-bench-" + uuid.uuid4().hex
    if capture:
        capture(
            {
                "record_type": "request_started",
                "request_id": request_id,
                "url": url,
                "payload": payload,
                "at": time.time(),
            }
        )
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json", "X-Request-ID": request_id},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read(1048577))
            status = response.status
        row = {
            "request": payload,
            "response": body,
            "elapsed_s": time.monotonic() - started,
            "status": status,
        }
    except urllib.error.HTTPError as exc:
        raw = exc.read(1048577)
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"error": raw[:1024].decode("utf-8", "replace")}
        row = {
            "request": payload,
            "response": body,
            "elapsed_s": time.monotonic() - started,
            "status": exc.code,
            "error": "HTTP " + str(exc.code),
        }
    except (OSError, ValueError) as exc:
        row = {
            "request": payload,
            "error": str(exc),
            "elapsed_s": time.monotonic() - started,
            "status": 0,
        }
    row["request_id"] = request_id
    if capture:
        capture({"record_type": "request_completed", **row})
    return row


def content(row):
    if row.get("status") != 200:
        raise ValueError("HTTP request did not succeed")
    response = row.get("response", {})
    choice = response.get("choices", [{}])[0]
    if type(choice) is not dict:
        raise ValueError("invalid model choice")
    if choice.get("finish_reason") != "stop":
        raise ValueError("incomplete model output")
    text = choice.get("message", {}).get("content", "")
    if type(text) is not str:
        raise ValueError("invalid model content")
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def same_json(a, b):
    # Python bool==int is NOT an acceptable exact match.
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def chat(url, prompt, model, capture=None):
    return post(
        url + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        capture=capture,
    )


def graph(url, arm, seed, model, capture=None, before_merge=None):
    tasks, expected = case(seed)
    barrier = threading.Barrier(3)

    def launch(index):
        barrier.wait()
        if index == 0:
            return chat(
                url,
                "You are the lead engineer. Plan a canary model-router on two Spark "
                f"nodes with C3 (three concurrent LLM requests), re-delivery and N+1. Evidence batch {seed}; "
                f"nodes={tasks[1]['inputs']['nodes']}, window=1800s, demand_tokens={tasks[1]['inputs']['tokens']}. Return JSON "
                '{"checks":[...],"rollback":[...]}. Exactly three specific short '
                "checks in each array, no invented measurements.",
                model,
                capture=capture,
            )
        task = tasks[index - 1]
        if arm == "after":
            return post(
                url + "/v1/exact/external-computers", {"tasks": [task]}, capture=capture
            )
        return chat(
            url,
            "Compute the result of the following explicitly specified program on the "
            "given data. This is a canary model-router audit; do not rewrite the "
            "program. SQL uses SQLite with checked int_sub(a,b)=a-b; python_expr uses Python integer semantics. "
            'Return only the result JSON: SQL {"columns":[...],"rows":[...]}, '
            "Python — the expression value itself.\n"
            + json.dumps(task, ensure_ascii=False),
            model,
            capture=capture,
        )

    started = time.monotonic()
    with ThreadPoolExecutor(3) as pool:
        wave = list(pool.map(launch, range(3)))
    first_wave_s = time.monotonic() - started
    facts, failures, checks = {}, [], {}
    try:
        plan = content(wave[0])
        if type(plan) is not dict:
            raise ValueError("plan must be an object")
        checks["plan_shape"] = all(
            isinstance(plan.get(k), list)
            and len(plan[k]) == 3
            and all(isinstance(x, str) and len(x) > 5 for x in plan[k])
            for k in ("checks", "rollback")
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
    merge = chat(
        url,
        "You are the lead engineer. Condense the canary auditors' reports into the "
        "reasoning and actions for a rollout decision. The verified facts and the "
        'decision are assembled by the harness. Return JSON {"reason": '
        '"<explanation>", "actions": [...]}. '
        + ACTION_HINT
        + "Include retain_pending, maintain_reserve and hold_rollout for no-go or proceed_rollout for go. Explain the decision "
        "implied by the reports: go only if n1_ok=true and every root received an "
        "answer (roots=delivered_roots), otherwise no-go. Never prescribe clearing "
        "or dropping pending work without a durable recovery/replay procedure, and "
        "never propose disabling the N+1 reserve without a verified transition "
        "plan.\nReports: "
        + json.dumps(facts, ensure_ascii=False)
        + "\nPlan: "
        + json.dumps(plan, ensure_ascii=False),
        model,
        capture=capture,
    )
    reason, actions, analysis_complete = "", [], False
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
    sampled, prompt, usages_valid = 0, 0, True
    for row in llm_rows:
        usage = row.get("response", {}).get("upstream_usage", {})
        if any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("completion_tokens", "prompt_tokens")
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
        "correct": all(checks.values()) and not failures,
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


def idle(capacity, metrics):
    """Ingress counters alone miss direct backend users; require both."""
    fields = ("active_weight", "waiting", "reserved", "admitted")
    if (
        capacity.get("backend_ready") is not True
        or capacity.get("admission_enabled") is not True
    ):
        return False
    if any(type(capacity.get(k)) is not int or capacity[k] != 0 for k in fields):
        return False
    found = set()
    for line in metrics.splitlines():
        match = re.match(
            r"^vllm:num_requests_(running|waiting)(?:\{[^\n]*\})?\s+(\S+)", line
        )
        if match:
            try:
                value = float(match[2])
            except ValueError:
                return False
            if not math.isfinite(value) or value != 0:
                return False
            found.add(match[1])
    return found == {"running", "waiting"}


class RunAccounting:
    """Scalar ledger of ALL launched work, including incomplete/censored graphs."""

    def __init__(self):
        self.requests, self.started_graphs, self.finished_graphs = {}, set(), set()

    def record(self, row):
        kind, ident = row.get("record_type"), row.get("request_id")
        if kind == "graph_started":
            self.started_graphs.add(row["graph_id"])
        elif kind == "graph":
            self.finished_graphs.add(row["graph_id"])
        elif kind == "request_started":
            self.requests[ident] = {
                "model": row["url"].endswith("/chat/completions"),
                "completed": False,
                "tokens": None,
                "status": None,
            }
        elif kind == "request_completed" and ident in self.requests:
            state = self.requests[ident]
            body = row.get("response", {})
            usage = (
                body.get("upstream_usage", body.get("usage", {}))
                if type(body) is dict
                else {}
            )
            tokens = usage.get("completion_tokens") if type(usage) is dict else None
            state.update(
                completed=True,
                status=row["status"],
                tokens=tokens if type(tokens) is int and tokens >= 0 else None,
            )

    def summary(self):
        model = [r for r in self.requests.values() if r["model"]]
        missing = sum(r["tokens"] is None for r in model)
        known = sum(r["tokens"] for r in model if r["tokens"] is not None)
        return {
            "http_started": len(self.requests),
            "http_completed": sum(r["completed"] for r in self.requests.values()),
            "http_failures": sum(
                r["completed"] and r["status"] != 200 for r in self.requests.values()
            ),
            "llm_started": len(model),
            "completion_tokens": None if missing else known,
            "known_completion_tokens_partial": known,
            "llm_usage_unknown": missing,
            "graphs_started": len(self.started_graphs),
            "graphs_completed": len(self.finished_graphs),
            "censored_graph_ids": sorted(self.started_graphs - self.finished_graphs),
        }


def summarize(rows):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "before",
        "after",
        "out",
        "capacity-url",
        "engine-metrics-url",
        "before-source",
        "after-source",
        "runtime-metadata",
    ):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--replay", help="summarize retained raw, including censored work; no network"
    )
    parser.add_argument("--model", default="qwen3.8-flash-next-nvidia")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7403)
    parser.add_argument(
        "--wait-idle",
        action="store_true",
        help="wait rather than refuse a busy qualification gate",
    )
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.replay), ensure_ascii=False, indent=2))
        return
    if args.rounds < 1 or args.rounds > 1000:
        parser.error("rounds must be in 1..1000")
    if any(
        getattr(args, name) is None
        for name in (
            "before",
            "after",
            "out",
            "capacity_url",
            "engine_metrics_url",
            "before_source",
            "after_source",
            "runtime_metadata",
        )
    ):
        parser.error(
            "live benchmark requires endpoints, out, source paths, runtime metadata and idle endpoints"
        )
    from bench_manifest import source_provenance, load_runtime

    runtime = load_runtime(args.runtime_metadata)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("raw.jsonl", "summary.json")):
        parser.error("fresh output paths required")
    rng, rows, ledger = random.Random(args.seed), [], RunAccounting()
    with (out / "raw.jsonl").open("x") as raw:
        lock = threading.Lock()

        def capture(row):
            with lock:
                ledger.record(row)
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()

        def gate(scoped, stage):
            while True:
                with urllib.request.urlopen(args.capacity_url, timeout=5) as response:
                    capacity = json.load(response)
                with urllib.request.urlopen(
                    args.engine_metrics_url, timeout=5
                ) as response:
                    metrics = response.read(1048577).decode("utf-8")
                clear = len(metrics.encode()) <= 1048576 and idle(capacity, metrics)
                scoped(
                    {
                        "record_type": "preflight" if clear else "wait_busy",
                        "stage": stage,
                        "at": time.time(),
                        "capacity": capacity,
                        "engine_request_gauges": [
                            s
                            for s in metrics.splitlines()
                            if s.startswith("vllm:num_requests_")
                        ],
                    }
                )
                if clear:
                    return
                if not args.wait_idle:
                    raise RuntimeError("qualification refused: backend is busy")
                time.sleep(15)

        hashes = {}
        for arm in ("before", "after"):
            root = Path(getattr(args, arm + "_source")).resolve()
            digest = hashlib.sha256()
            for folder in ("src", "benchmarks"):
                for path in sorted((root / folder).rglob("*.py")):
                    digest.update(
                        str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
                    )
            hashes[arm] = {
                "root": str(root),
                "source_tree_sha256": digest.hexdigest(),
                **source_provenance(root),
            }
        capture(
            {
                "record_type": "manifest",
                "fixture_version": 2,
                "sources": hashes,
                "seed": args.seed,
                "rounds": args.rounds,
                "model": args.model,
                "python": platform.python_version(),
                "max_tokens": "server default",
                "runtime_declared_not_attested": runtime,
                "scope": "provided-program offload workflow, not program selection or GPU C3",
                "harness_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        try:
            for index in range(args.rounds):
                order = ["before", "after"]
                rng.shuffle(order)
                for arm in order:
                    gid = uuid.uuid4().hex

                    def scoped(row):
                        capture(
                            {
                                **row,
                                "graph_id": gid,
                                "seed": args.seed + index,
                                "arm": arm,
                            }
                        )

                    gate(scoped, "before_graph")
                    scoped({"record_type": "graph_started", "at": time.time()})
                    row = graph(
                        getattr(args, arm),
                        arm,
                        args.seed + index,
                        args.model,
                        capture=scoped,
                        before_merge=lambda: gate(scoped, "before_merge"),
                    )
                    scoped(row)
                    rows.append(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("wave", "merge", "result")
                        }
                    )
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        except BaseException as exc:
            capture({"record_type": "run_aborted", "error": type(exc).__name__})
            raise
        finally:
            result = {**summarize(rows), "accounting": ledger.summary()}
            with (out / "summary.json").open("x") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
