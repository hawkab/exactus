"""Generate operational fixtures and independent expected facts."""

from __future__ import annotations

__author__ = "Olshansky Grigory"

import random


def case(seed):
    """Generate a reproducible workload and independently specified expected facts."""
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
