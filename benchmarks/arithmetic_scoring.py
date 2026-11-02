"""Independently score arithmetic expressions and completed model output."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import ast
import operator
import re

from exactus.domain.finalization import validate_final
from exactus.domain.protocol import SpanMachine

ANSWER_RE = re.compile(r"\\answer\{(-?\d+)\}")


def independent_eval(expression):
    """Separate bounded AST interpreter. Never eval/exec model-generated code."""
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError("expression too large")
    tree = ast.parse(expression.replace("^", "**").replace("mod", "%"), mode="eval")
    if len(list(ast.walk(tree))) > 512:
        raise ValueError("expression too complex")

    def visit(node, depth=0):
        """Evaluate only integer arithmetic nodes using the independent reference semantics."""
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
    """Extract exactly one final integer from the model answer marker."""
    found = ANSWER_RE.findall(text or "")
    return int(found[0]) if len(found) == 1 else None


def score_marker(text, external_computer):
    """Verify every emitted computation and the final marker against an independent evaluator."""
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
    """Record answer, completion, usage, and timing observations with separate failure reasons."""
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
