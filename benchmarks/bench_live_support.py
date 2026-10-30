"""Bounded independent scoring and transport; external computers are not the judge."""

from __future__ import annotations
import ast
import json
import operator
import random
import re
import sys
import time
import copy
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exactus.domain.protocol import SpanMachine
from exactus.domain.finalization import validate_final
from exactus.http_io import stream_lines

CAPTURED_SEGMENTS = ContextVar("exactus_benchmark_segments", default=None)
_GATE = ContextVar("exactus_benchmark_gate", default=None)


@contextmanager
def qualification(gate):
    token = _GATE.set(gate)
    try:
        yield
    finally:
        _GATE.reset(token)


class BenchFailure(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


SEED = 20261029
DEFAULT_MAX_TOKENS = 4096
ANSWER_RE = re.compile(r"\\answer\{(-?\d+)\}")
_BINOPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


def tasks(n, seed=SEED):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        a, b, c = (
            rnd.randint(10**6, 10**9 - 1),
            rnd.randint(1000, 99999),
            rnd.randint(2, 999),
        )
        op, second = ("*", "+", "-")[i % 3], ("+", "-", "*")[i % 3]
        rows.append(
            {
                "id": f"task-{seed}-{i}",
                "expr": f"({a} {op} {b}) {second} {c}",
                "q": f"Compute exactly: ({a} {op} {b}) {second} {c}. Final answer in \\answer{{...}}.",
                "truth": _BINOPS[second](_BINOPS[op](a, b), c),
            }
        )
    return rows


def chat(port, model, messages, tools=None, max_tokens=DEFAULT_MAX_TOKENS):
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


def independent_eval(expression):
    """Separate bounded AST interpreter. Never eval/exec model-generated code."""
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError("expression too large")
    tree = ast.parse(expression.replace("^", "**").replace("mod", "%"), mode="eval")
    if len(list(ast.walk(tree))) > 512:
        raise ValueError("expression too complex")

    def visit(node, depth=0):
        if depth > 64:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = visit(node.operand, depth + 1)
            result = -value if isinstance(node.op, ast.USub) else value
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if (
                    not 0 <= right <= 4096
                    or (abs(left).bit_length() - 1) * right >= 4096
                ):
                    raise ValueError("power budget")
                result = left**right
            elif isinstance(node.op, (ast.Div, ast.Mod)):
                q = abs(left) // abs(right)
                q = -q if (left < 0) != (right < 0) else q
                result = q if isinstance(node.op, ast.Div) else left - q * right
            else:
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.FloorDiv: operator.floordiv,
                }
                if type(node.op) not in ops:
                    raise ValueError("unsupported operator")
                result = ops[type(node.op)](left, right)
        else:
            raise ValueError("unsupported node")
        if result.bit_length() > 4096:
            raise ValueError("integer budget")
        return result

    return visit(tree.body)


def score_alone(text):
    found = ANSWER_RE.findall(text or "")
    return int(found[0]) if len(found) == 1 else None


def score_marker(text, external_computer):
    try:
        machine = SpanMachine()
        spans = machine.feed(text)
        parsed = machine.finish()
        validate_final(parsed, "strict")
        finals = [s for s in spans if s.final]
        if len(finals) != 1 or spans[-1] != finals[0]:
            return False, None, False
        values = [external_computer.evaluate(s.expression).integer for s in spans]
        parity = all(v == independent_eval(s.expression) for s, v in zip(spans, values))
        return True, values[-1], parity
    except (ValueError, ArithmeticError, RecursionError):
        return False, None, False
    except Exception as exc:
        from exactus.domain.errors import ExactusError

        if isinstance(exc, ExactusError):
            return False, None, False
        raise


def pack(task, value, r, ok_expr, parity):
    usage = r["usage"]
    valid = all(
        type(usage.get(k)) is int and usage[k] >= 0
        for k in ("completion_tokens", "prompt_tokens")
    )
    answer_match = value == task["truth"]
    complete = r["finish"] == "stop"
    failures = []
    if not complete:
        failures.append(
            "truncated" if r["finish"] == "length" else "incomplete_generation"
        )
    if not valid:
        failures.append("invalid_usage")
    if not parity:
        failures.append("invalid_computation")
    if not answer_match:
        failures.append("answer_mismatch")
    return {
        "correct": not failures,
        "failures": failures,
        "answer_match": answer_match,
        "truncated": r["finish"] == "length",
        "usage_valid": valid,
        "finish": r["finish"],
        "value": value,
        "ctok": usage.get("completion_tokens") if valid else None,
        "ptok": usage.get("prompt_tokens") if valid else None,
        "wall": r["wall"],
        "ttft": r["ttft"],
        "marker": ok_expr,
        "parity": parity,
        "task": task,
        "output": r["text"],
        "segments": r.get("segments", [r]),
    }
