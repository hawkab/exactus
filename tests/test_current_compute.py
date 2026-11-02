"""Executable counterexamples from the e511bf4 review."""

__author__ = "Olshansky Grigory"


import random
from unittest.mock import patch

from exactus.adapters.digest import HashlibHasher
from exactus.adapters.in_process import InProcessExternalComputer
from exactus.domain.errors import ExactusError
from exactus.external_computers.plan import run_plan
from exactus.external_computers.python_expr import evaluate
from exactus.external_computers.sql_read import evaluate as sql
from exactus.external_computers.values import OwnedJSON, packed
from tests._support_current_review import CurrentReviewFixture


class CurrentReview(CurrentReviewFixture):
    """Verify current compute behavior."""

    def test_generators_short_circuit_and_keep_list_comprehensions_eager(self):
        """Verify generators short circuit and keep list comprehensions eager."""
        self.assertTrue(evaluate("any(1//x for x in xs)", {"xs": [1, 0]}))
        self.assertFalse(evaluate("all(1//x for x in xs)", {"xs": [2, 0]}))
        with self.assertRaises(ExactusError):
            evaluate("any([1//x for x in xs])", {"xs": [1, 0]})
        self.assertEqual(evaluate("sum(x*x for x in range(20) if x%2)", {}), 1330)
        self.assertEqual(
            evaluate("sorted(x for x in xs)", {"xs": [3, 1, 2]}), [1, 2, 3]
        )
        for expression in (
            "(x for x in xs)",
            "len(x for x in xs)",
            "sum(x for x in range(1000000))",
        ):
            with self.subTest(expression=expression), self.assertRaises(ExactusError):
                evaluate(expression, {"xs": [1, 2]})

    def test_sql_forbids_hidden_real_and_uses_checked_integer_math(self):
        """Verify sql forbids hidden real and uses checked integer math."""
        tables = {"items": {"columns": ["x"], "rows": [[1], [2]]}}
        for query in (
            "SELECT CAST(9007199254740993*1.0 AS INTEGER)",
            "SELECT CAST((9223372036854775807+1)-9223372036854775807 AS INTEGER)",
            "SELECT 9223372036854775807+1",
            "SELECT int_add(9223372036854775807,1)",
            "SELECT int_div(1,0)",
            "SELECT avg(x)>0 FROM items",
            "SELECT sum('0.5')>0 FROM items",
        ):
            with self.subTest(query=query), self.assertRaises(ExactusError):
                sql(query, tables)
        self.assertEqual(
            sql("SELECT int_sub(sum(x),count(*)) FROM items", tables)["rows"], [[1]]
        )
        self.assertEqual(
            sql("SELECT int_div(-7,2),int_mod(-7,2)", tables)["rows"], [[-4, 1]]
        )

    def test_owned_encoding_matches_boundary_encoding_and_charges_aliases(self):
        """Verify owned encoding matches boundary encoding and charges aliases."""
        values = [
            None,
            True,
            -43,
            "кириллица\n",
            [],
            {},
            {"z": [1, {"x": 'quote"'}], "a": False},
        ]
        for value in values:
            self.assertEqual(OwnedJSON().encode(value), packed(value))
        owner = OwnedJSON()
        repeated = list(range(100))
        owner.encode(repeated)
        self.assertEqual(owner.encode([repeated] * 10), packed([repeated] * 10))
        for bad in ([repeated] * 180, ["x" * 4000] * 20, {"x": 1.5}, "\ud800"):
            with self.subTest(kind=type(bad)), self.assertRaises(ExactusError):
                owner.encode(bad)
        aggregate = owner.sequence()
        with self.assertRaises(ExactusError):
            for _ in range(30):
                aggregate.append("x" * 4000)

    def test_plan_no_repeated_packed_rows_and_same_canonical_digests(self):
        """Verify plan no repeated packed rows and same canonical digests."""
        plan = {
            "tasks": [
                {
                    "id": f"t{i}",
                    "kind": "python_expr",
                    "expression": "xs",
                    "inputs": {"xs": {"$input": "xs"}},
                }
                for i in range(16)
            ],
            "answer": [{"$ref": f"t{i}"} for i in range(16)],
        }
        from exactus.external_computers import plan as module

        with patch.object(module, "packed", wraps=packed) as calls:
            result = run_plan(
                plan,
                InProcessExternalComputer(HashlibHasher()),
                inputs={"xs": list(range(256))},
            )
        self.assertEqual(sum(r["cached"] for r in result["results"]), 15)
        self.assertEqual(calls.call_count, 1)

    def test_sql_catalog_reused_only_inside_one_request(self):
        """Verify sql catalog reused only inside one request."""
        from exactus.external_computers import sql_read
        from exactus.external_computers.batch import run_external_computers

        mutable = {"items": {"columns": ["x"], "rows": [[1]]}}
        with sql_read.SQLSession() as session:
            self.assertEqual(
                sql("SELECT sum(x) FROM items", mutable, session=session)["rows"], [[1]]
            )
            mutable["items"]["rows"].append([2])
            self.assertEqual(
                sql("SELECT sum(x) FROM items", mutable, session=session)["rows"], [[3]]
            )

        tables = {"items": {"columns": ["x"], "rows": [[1], [2]]}}
        tasks = [
            {
                "id": "a",
                "kind": "sql_read",
                "query": "SELECT count(*) FROM items",
                "tables": tables,
            },
            {
                "id": "b",
                "kind": "sql_read",
                "query": "SELECT sum(x) FROM items",
                "tables": tables,
            },
        ]
        with patch.object(sql_read, "_prepare", wraps=sql_read._prepare) as prepare:
            run_external_computers(tasks, InProcessExternalComputer(HashlibHasher()))
            self.assertEqual(prepare.call_count, 1)
            run_external_computers(tasks, InProcessExternalComputer(HashlibHasher()))
            self.assertEqual(prepare.call_count, 2)
        session = sql_read.SQLSession()
        connection = session.connection(tables)
        session.close()
        import sqlite3

        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_owned_json_generated_trees_match_canonical_encoding(self):
        """Verify owned json generated trees match canonical encoding."""
        rng = random.Random(62030)

        def tree(depth):
            """Generate a reproducible nested JSON value for canonical-encoding comparison."""
            if depth == 0 or rng.randrange(3) == 0:
                return rng.choice([None, False, True, -12345, 2**90, '\n\t"\\Ё界'])
            if rng.randrange(2):
                return [tree(depth - 1) for _ in range(rng.randrange(5))]
            return {str(i): tree(depth - 1) for i in range(rng.randrange(5))}

        for _ in range(500):
            value = tree(5)
            self.assertEqual(OwnedJSON().encode(value), packed(value))
