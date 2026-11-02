"""External computer correctness, capability boundaries, IPC and HTTP contracts."""

__author__ = "Olshansky Grigory"


import random
import unittest

from exactus.domain.errors import ExactusError
from exactus.external_computers.python_expr import evaluate as python

TABLES = {
    "events": {
        "columns": ["root", "attempt", "delivered", "tokens"],
        "rows": [
            ["a", 1, True, 101],
            ["b", 1, False, 20],
            ["b", 2, True, 73],
            ["c", 1, False, 32],
        ],
    }
}


class PythonExternalComputerTests(unittest.TestCase):
    """Verify external computers behavior."""

    def test_technical_comprehensions_and_json(self):
        """Verify technical comprehensions and json."""
        result = python(
            '{"tokens": sum(r["tokens"] for r in runs if r["ok"]), "retry": [r["id"] for r in runs if not r["ok"]]}',
            {
                "runs": [
                    {"id": "a", "tokens": 97, "ok": True},
                    {"id": "b", "tokens": 42, "ok": False},
                ]
            },
        )
        self.assertEqual(result, {"tokens": 97, "retry": ["b"]})

    def test_matches_python_integer_semantics(self):
        """Verify matches python integer semantics."""
        self.assertEqual(
            python("[-7 // 3, -7 % 3, 3**8, 5^3, abs(-17)]", {}), [-3, 2, 6561, 6, 17]
        )

    def test_short_circuit_and_local_scope(self):
        """Verify short circuit and local scope."""
        self.assertEqual(
            python("[x for x in range(3) if x < 2] + [x]", {"x": 99}), [0, 1, 99]
        )
        self.assertEqual(python("False and (1//0)", {}), False)

    def test_unsafe_code_and_unsupported_features_fail_closed(self):
        """Verify unsafe code and unsupported features fail closed."""
        for expr in (
            '__import__("os").system("id")',
            'open("/etc/passwd")',
            "(1).__class__",
            "(lambda: 1)()",
            "[x for x in range(3) for y in range(3)]",
            "2 / 3",
            "2**-1",
            "1.2+3",
            "sum([True])",
            'exec("x=1")',
            "[1][0:1]",
            "x is 12345",
            "sorted([1], reverse=True)",
        ):
            with self.subTest(expr=expr), self.assertRaises(ExactusError):
                python(expr, {})

    def test_size_and_compute_budgets(self):
        """Verify size and compute budgets."""
        for expr in (
            "2**1000000",
            '"a"*1000000000',
            "[1]*100000000",
            "range(9999999)",
            "sum(sum(range(2048)) for x in range(2048))",
        ):
            with self.subTest(expr=expr), self.assertRaises(ExactusError):
                python(expr, {})

    def test_non_json_types_are_rejected(self):
        """Verify non json types are rejected."""
        for inputs in ({"x": 1.2}, {1: "a"}, {"_private": 1}, {"x": object()}):
            with self.subTest(inputs=inputs), self.assertRaises(ExactusError):
                python("x", inputs)

    def test_seeded_semantics_against_independent_loops(self):
        """Verify seeded semantics against independent loops."""
        rng = random.Random(7403)
        for _ in range(120):
            values = [rng.randrange(-50000, 50000) for _ in range(rng.randrange(1, 40))]
            expected_sum, positives = 0, []
            for v in values:
                if v >= 0:
                    positives.append(v)
                    expected_sum += v * v
            result = python(
                '{"sum": sum(v*v for v in xs if v >= 0), "ordered": sorted([v for v in xs if v >= 0])}',
                {"xs": values},
            )
            self.assertEqual(
                result, {"sum": expected_sum, "ordered": sorted(positives)}
            )
