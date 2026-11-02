"""Bounded independent scoring and transport; external computers are not the judge."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import copy
import json
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exactus.http_io import stream_lines

CAPTURED_SEGMENTS = ContextVar("exactus_benchmark_segments", default=None)
_GATE = ContextVar("exactus_benchmark_gate", default=None)


from arithmetic_cases import tasks as tasks
from arithmetic_scoring import (
    independent_eval as independent_eval,
)
from arithmetic_scoring import (
    pack as pack,
)
from arithmetic_scoring import (
    score_alone as score_alone,
)
from arithmetic_scoring import (
    score_marker as score_marker,
)


@contextmanager
def qualification(gate):
    """Bind the shared-backend qualification gate for the current execution context."""
    token = _GATE.set(gate)
    try:
        yield
    finally:
        _GATE.reset(token)


class BenchFailure(RuntimeError):
    """Carry a benchmark transport failure together with captured partial output."""

    def __init__(self, message, partial):
        """Initialize Bench Failure state and dependencies."""
        super().__init__(message)
        self.partial = partial


SEED = 20261029
DEFAULT_MAX_TOKENS = 4096


def chat(port, model, messages, tools=None, max_tokens=DEFAULT_MAX_TOKENS):
    """Send a model request and retain its actual usage and completion status."""
    gate = _GATE.get()
    if gate is not None:
        gate()
    body = {
        "model": model,
        "messages": copy.deepcopy(messages),
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    started, ttft = time.perf_counter(), None
    chunks, usage, finish, calls, size = [], {}, None, [], 0
    try:
        for raw in stream_lines(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ):
            size += len(raw)
            if size > 2097152:
                raise ValueError("SSE response exceeds 2 MiB")
            line = raw.decode("utf-8", "strict").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    chunks.append(piece)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    if type(idx) is not int or not 0 <= idx < 16:
                        raise ValueError("invalid tool call index")
                    while len(calls) <= idx:
                        calls.append({"id": "", "arguments": "", "name": ""})
                    calls[idx]["id"] = tc.get("id") or calls[idx]["id"]
                    calls[idx]["name"] = (tc.get("function") or {}).get(
                        "name"
                    ) or calls[idx]["name"]
                    calls[idx]["arguments"] += (tc.get("function") or {}).get(
                        "arguments"
                    ) or ""
    except Exception as exc:
        partial = {
            "text": "".join(chunks),
            "usage": usage,
            "finish": finish,
            "calls": calls,
            "request": body,
            "wall": time.perf_counter() - started,
            "ttft": ttft,
            "transport_error": type(exc).__name__,
        }
        captured = CAPTURED_SEGMENTS.get()
        if captured is not None:
            captured.append(partial)
        raise BenchFailure(
            f"{type(exc).__name__}: {exc}",
            partial,
        ) from exc
    result = {
        "text": "".join(chunks),
        "usage": usage,
        "wall": time.perf_counter() - started,
        "ttft": ttft,
        "finish": "tool_calls" if calls and finish == "stop" else finish,
        "upstream_finish": finish,
        "calls": calls,
        "request": body,
    }
    captured = CAPTURED_SEGMENTS.get()
    if captured is not None:
        captured.append(result)
    return result
