"""Record measured arithmetic runs and interrupted qualification evidence."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import time

from arithmetic_setup import prepare_run
from bench_live_support import CAPTURED_SEGMENTS, SEED, qualification, tasks
from bench_report import aggregate, metric_row, paired


def run_benchmark(harness):
    """Record all benchmark observations and interrupted work without hiding failures."""
    setup = prepare_run(harness)
    if setup is None:
        return
    args, manifest, schedule, functions, out, names = setup
    gate = harness.QualificationGate(args.capacity_url, args.engine_metrics_url)
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
                raw.write(
                    json.dumps(
                        {
                            "record_type": "qualification_refused"
                            if isinstance(exc, harness.QualificationBusy)
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
            except harness.QualificationBusy as exc:
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
