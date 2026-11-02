"""Generate reproducible arithmetic workloads."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import operator
import random

SEED = 20261029
_BINOPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


def tasks(n, seed=SEED):
    """Generate distinct arithmetic tasks with independent integer answers."""
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
