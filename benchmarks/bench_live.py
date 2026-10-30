"""Protocol microbenchmark, not a native/MTP or agent-graph speed claim."""

from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parent),
]
from bench_live_support import (
    DEFAULT_MAX_TOKENS,
    SEED,
    chat,
    independent_eval,
    pack,
    score_alone,
    score_marker,
    tasks,
    CAPTURED_SEGMENTS,
    qualification,
)
from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.prompts import MARKER_HINT
from bench_report import aggregate, paired, replay, stats as stats, metric_row
from bench_manifest import (
    load_runtime,
    source_provenance,
    QualificationGate,
    QualificationBusy,
)


def run_alone(port, model, t, max_tokens=DEFAULT_MAX_TOKENS):
    r = chat(port, model, [{"role": "user", "content": t["q"]}], max_tokens=max_tokens)
    return pack(t, score_alone(r["text"]), r, False, True)


def run_marker(port, model, t, external_computer, max_tokens=DEFAULT_MAX_TOKENS):
    r = chat(
        port,
        model,
        [
            {"role": "system", "content": MARKER_HINT},
            {"role": "user", "content": t["q"]},
        ],
        max_tokens=max_tokens,
    )
    emitted, value, parity = score_marker(r["text"], external_computer)
    return pack(t, value, r, emitted, parity)


def run_marker_loop(port, model, t, external_computer, max_tokens=DEFAULT_MAX_TOKENS):
    from exactus.domain.protocol import SpanMachine

    started = time.perf_counter()
    msgs = [
        {"role": "system", "content": MARKER_HINT},
        {"role": "user", "content": t["q"]},
    ]
    segments, usage, ttft, remaining = [], {}, None, max_tokens
    for _ in range(4):
        r = chat(port, model, msgs, max_tokens=remaining)
        segments.append(r)
        if ttft is None and r["ttft"] is not None:
            ttft = time.perf_counter() - started - r["wall"] + r["ttft"]
        if r["finish"] not in ("stop",):
            break
        used = r["usage"].get("completion_tokens")
        if type(used) is not int or used >= remaining:
            break
        remaining -= used
        machine = SpanMachine()
        spans = machine.feed(r["text"])
        machine.finish()
        if not spans or any(s.final for s in spans):
            break
        values = [external_computer.evaluate(s.expression).render() for s in spans]
        msgs = msgs + [
            {"role": "assistant", "content": r["text"]},
            {
                "role": "user",
                "content": "\n".join(f"<calc-result>{v}</calc-result>" for v in values),
            },
        ]
    for key in ("completion_tokens", "prompt_tokens"):
        counts = [s["usage"].get(key) for s in segments]
        if all(type(c) is int for c in counts):
            usage[key] = sum(counts)
    emitted, value, parity = score_marker(r["text"], external_computer)
    merged = {
        **r,
        "usage": usage,
        "wall": time.perf_counter() - started,
        "ttft": ttft,
        "segments": segments,
    }
    return pack(t, value, merged, emitted, parity)


def run_toolcall(port, model, t, max_tokens=DEFAULT_MAX_TOKENS):
    started = time.perf_counter()
    tool = {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "exact calculator",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
    msgs = [
        {
            "role": "system",
            "content": "Solve with the calc tool. Final answer in \\answer{...}.",
        },
        {"role": "user", "content": t["q"]},
    ]
    segments, ttft, remaining = [], None, max_tokens
    for _ in range(8):
        offset = time.perf_counter() - started
        r = chat(port, model, msgs, tools=[tool], max_tokens=remaining)
        segments.append(r)
        if ttft is None and r["ttft"] is not None:
            ttft = offset + r["ttft"]
        used = r["usage"].get("completion_tokens")
        if type(used) is not int or used < 0 or used > remaining:
            r = {**r, "finish": "invalid_usage"}
            break
        remaining -= used
        if r["finish"] != "tool_calls":
            break
        if not r["calls"] or remaining == 0 or time.perf_counter() - started > 600:
            break
        calls, outputs = [], []
        for call in r["calls"]:
            if call.get("name", "calc") != "calc":
                raise ValueError("unexpected tool name")
            expr = json.loads(call["arguments"])["expression"]
            outputs.append(str(independent_eval(expr)))
            calls.append(
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": "calc", "arguments": call["arguments"]},
                }
            )
        msgs = msgs + [{"role": "assistant", "tool_calls": calls}]
        msgs += [
            {"role": "tool", "tool_call_id": c["id"], "content": value}
            for c, value in zip(calls, outputs)
        ]
    usage = {}
    for key in ("completion_tokens", "prompt_tokens"):
        counts = [s["usage"].get(key) for s in segments]
        if all(type(c) is int for c in counts):
            usage[key] = sum(counts)
    return pack(
        t,
        score_alone(r["text"]),
        {
            **r,
            "usage": usage,
            "ttft": ttft,
            "wall": time.perf_counter() - started,
            "segments": segments,
        },
        False,
        True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--model", default="qwen3.8-flash-next-nvidia")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="evidence/bench-live.json")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--arms", default="alone,marker,marker-loop,toolcall")
    ap.add_argument(
        "--warmup", type=int, default=2, help="excluded warmup calls per arm"
    )
    ap.add_argument(
        "--runtime-manifest",
        help="JSON with actual model/tokenizer/container/vLLM SHAs and flags",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", help="rebuild summary from raw JSONL, without network")
    ap.add_argument("--capacity-url", help="read-only ingress qualification endpoint")
    ap.add_argument(
        "--engine-metrics-url", help="read-only engine qualification endpoint"
    )
    args = ap.parse_args()
    if args.replay:
        print(json.dumps(replay(args.replay), ensure_ascii=False, indent=2))
        return
    if min(args.runs, args.n, args.max_tokens) < 1 or args.warmup < 0:
        ap.error("runs/n/max-tokens must be positive and warmup nonnegative")
    external_computer = InProcessExternalComputer(HashlibHasher())
    functions = {
        "alone": lambda t: run_alone(args.port, args.model, t, args.max_tokens),
        "marker": lambda t: run_marker(
            args.port, args.model, t, external_computer, args.max_tokens
        ),
        "marker-loop": lambda t: run_marker_loop(
            args.port, args.model, t, external_computer, args.max_tokens
        ),
        "toolcall": lambda t: run_toolcall(args.port, args.model, t, args.max_tokens),
    }
    names = args.arms.split(",")
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in functions for n in names)
    ):
        ap.error("unknown or duplicate arm")
    rng, schedule = random.Random(SEED), []
    for task in tasks(args.runs * args.n):
        order = names[:]
        rng.shuffle(order)
        schedule.extend((name, task) for name in order)
    root = Path(__file__).resolve().parents[1]
    provenance = source_provenance(root)
    source_hash = hashlib.sha256()
    for folder in (root / "src", root / "benchmarks"):
        for path in sorted(folder.rglob("*.py")):
            source_hash.update(
                str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
            )
    manifest = {
        **provenance,
        "model": args.model,
        "seed": SEED,
        "source_tree_sha256": source_hash.hexdigest(),
        "python": platform.python_version(),
        "max_tokens_per_task": args.max_tokens,
        "unique_tasks": args.n * args.runs,
        "arm_task_observations": len(schedule),
        "http_calls": "counted from raw segments, not arm-task observations",
        "concurrency": 1,
        "warmup_calls_per_arm": args.warmup,
        "scope": "HTTP protocol microbenchmark; NOT native/MTP or agent graph",
        "ttft_scope": "first raw model content, not non-stream proxy visibility",
        "runtime": load_runtime(args.runtime_manifest, names),
        "runtime_provenance": "user_supplied_not_attested",
    }
    if args.dry_run:
        print(json.dumps(manifest))
        return
    if args.runtime_manifest is None:
        ap.error(
            "live evidence requires --runtime-manifest; use --dry-run for offline smoke"
        )
    out = Path(args.out)
    if out.suffix != ".json":
        ap.error("--out must end in .json; raw evidence is a separate .jsonl")
    if out.exists() or out.with_suffix(".jsonl").exists():
        ap.error("fresh output paths required; existing evidence is never overwritten")
    if not args.capacity_url or not args.engine_metrics_url:
        ap.error("live benchmark requires --capacity-url and --engine-metrics-url")
    gate = QualificationGate(args.capacity_url, args.engine_metrics_url)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.with_suffix(".jsonl").open("x") as raw, qualification(gate):
        raw.write(
            json.dumps(
                {"record_type": "manifest", "manifest": manifest}, ensure_ascii=False
            )
            + "\n"
        )
        raw.flush()
        for index, warm in enumerate(tasks(args.warmup * len(names), seed=SEED + 7)):
            name = names[index % len(names)]
            CAPTURED_SEGMENTS.set([])
            try:
                row = functions[name](warm)
            except BaseException as exc:
                # A failed warmup can already have spent one or more calls.
                # Keep its evidence but never turn it into a measured task or
                # silently continue with a different cache/warmup policy.
                raw.write(
                    json.dumps(
                        {
                            "record_type": "qualification_refused"
                            if isinstance(exc, QualificationBusy)
                            else "warmup",
                            "phase": "warmup",
                            "arm": name,
                            "task": warm,
                            "correct": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "segments": CAPTURED_SEGMENTS.get(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw.flush()
                raise
            row.update(arm=name, record_type="warmup")
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()
        for name, task in schedule:
            started = time.perf_counter()
            gate_before = gate.elapsed_s
            CAPTURED_SEGMENTS.set([])
            try:
                row = functions[name](task)
            except QualificationBusy as exc:
                raw.write(
                    json.dumps(
                        {
                            "record_type": "qualification_refused",
                            "arm": name,
                            "task": task,
                            "error": str(exc),
                            "segments": CAPTURED_SEGMENTS.get(),
                        }
                    )
                    + "\n"
                )
                raw.flush()
                raise
            except Exception as exc:
                row = {
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ctok": None,
                    "ttft": None,
                    "truncated": any(
                        s.get("finish") == "length" for s in CAPTURED_SEGMENTS.get()
                    ),
                    "task": task,
                    "failures": ["exception"],
                    "partial": getattr(exc, "partial", None),
                    "segments": CAPTURED_SEGMENTS.get(),
                }
            wall = time.perf_counter() - started
            gate_s = gate.elapsed_s - gate_before
            row.update(
                arm=name,
                wall=max(0.0, wall - gate_s),
                qualification_gate_s=gate_s,
                wall_including_qualification_s=wall,
            )
            rows.append(metric_row(row))
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()
    arms = aggregate(rows)
    with out.open("x") as summary:
        summary.write(
            json.dumps(
                {"manifest": manifest, "arms": arms, "paired": paired(rows)},
                ensure_ascii=False,
                indent=2,
            )
        )
    print(json.dumps(arms, ensure_ascii=False))


if __name__ == "__main__":
    main()
