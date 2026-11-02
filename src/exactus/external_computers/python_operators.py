"""Evaluate integer and bounded-sequence binary expressions."""

__author__ = "Olshansky Grigory"

import ast

from ..domain.errors import BudgetError, GrammarError
from .python_syntax import _BINARY
from .values import MAX_BITS, MAX_BYTES, MAX_ITEMS, bounded


def binary(interpreter, node, env, depth):
    """Evaluate checked integer or bounded sequence operations."""
    child_depth = depth + 1
    a = interpreter.visit(node.left, env, child_depth)
    b = interpreter.visit(node.right, env, child_depth)
    if type(a) is int and type(b) is int:
        if isinstance(node.op, ast.Pow) and (
            b < 0 or b > MAX_BITS or (abs(a) > 1 and a.bit_length() * b > MAX_BITS)
        ):
            raise BudgetError("Python power exceeds integer budget")
    elif isinstance(node.op, ast.Add) and type(a) is type(b) and type(a) in (list, str):
        if len(a) + len(b) > (MAX_ITEMS if type(a) is list else MAX_BYTES):
            raise BudgetError("Python concatenation exceeds budget")
    elif isinstance(node.op, ast.Mult) and (
        (type(a) in (list, str) and type(b) is int)
        or (type(b) in (list, str) and type(a) is int)
    ):
        seq, n = (a, b) if type(b) is int else (b, a)
        if len(seq) * max(0, n) > (MAX_ITEMS if type(seq) is list else MAX_BYTES):
            raise BudgetError("Python repetition exceeds budget")
    else:
        raise GrammarError("unsupported Python operand types")
    return bounded(_BINARY[type(node.op)](a, b))
