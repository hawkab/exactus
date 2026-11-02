"""Execute resumable marker-injection benchmark rounds."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import time

from bench_live_support import (
    DEFAULT_MAX_TOKENS,
    pack,
    score_marker,
)

from exactus.prompts import MARKER_HINT


def run_marker_loop(
    port, model, t, external_computer, max_tokens=DEFAULT_MAX_TOKENS, *, chat
):
    """Continue marker-injection rounds while retaining all segment usage."""
    from exactus.domain.protocol import SpanMachine

    started = time.perf_counter()
    msgs = [
        {"role": "system", "content": MARKER_HINT},
        {"role": "user", "content": t["q"]},
    ]
    segments, usage, ttft, remaining = ([], {}, None, max_tokens)
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
        if not spans or any((s.final for s in spans)):
            break
        values = [external_computer.evaluate(s.expression).render() for s in spans]
        msgs = msgs + [
            {"role": "assistant", "content": r["text"]},
            {
                "role": "user",
                "content": "\n".join(
                    (f"<calc-result>{v}</calc-result>" for v in values)
                ),
            },
        ]
    for key in ("completion_tokens", "prompt_tokens"):
        counts = [s["usage"].get(key) for s in segments]
        if all((type(c) is int for c in counts)):
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
