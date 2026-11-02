"""Execute resumable calculator-tool benchmark rounds."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import json
import time

from bench_live_support import (
    DEFAULT_MAX_TOKENS,
    independent_eval,
    pack,
    score_alone,
)


def run_toolcall(port, model, t, max_tokens=DEFAULT_MAX_TOKENS, *, chat):
    """Continue calculator-tool rounds and retain the full conversation and usage."""
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
    segments, ttft, remaining = ([], None, max_tokens)
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
        calls, outputs = ([], [])
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
        if all((type(c) is int for c in counts)):
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
