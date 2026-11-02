"""External computer correctness, capability boundaries, IPC and HTTP contracts."""

__author__ = "Olshansky Grigory"


import unittest

from exactus.domain.errors import ExactusError
from exactus.external_computers.sql_read import evaluate as sql

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


class SQLExternalComputerTests(unittest.TestCase):
    """Verify sql computer behavior."""

    def test_named_bindings_and_empty_mapping_are_bounded_values_not_sql(self):
        """Verify named bindings and empty mapping are bounded values not sql."""
        self.assertEqual(sql("SELECT count(*) FROM events", TABLES, {})["rows"], [[4]])
        self.assertEqual(
            sql(
                "SELECT count(DISTINCT root) FROM events WHERE delivered=:ok",
                TABLES,
                {"ok": 1},
            )["rows"],
            [[2]],
        )
        self.assertEqual(
            sql(
                "SELECT count(*) FROM events WHERE root=:root",
                TABLES,
                {"root": "' OR 1=1 --"},
            )["rows"],
            [[0]],
        )
        for parameters in (
            {"bad name": 1},
            {"x": []},
            {f"p{i}": i for i in range(129)},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ExactusError):
                sql("SELECT 1", TABLES, parameters)

    def test_aggregate_parameters_and_join(self):
        """Verify aggregate parameters and join."""
        self.assertEqual(
            sql(
                "SELECT count(DISTINCT root), sum(tokens) FROM events WHERE delivered = ?",
                TABLES,
                [True],
            )["rows"],
            [[2, 174]],
        )
        self.assertEqual(
            sql(
                "SELECT a.root FROM events a JOIN events b ON a.root=b.root WHERE a.attempt < b.attempt",
                TABLES,
            )["rows"],
            [["b"]],
        )

    def test_cte(self):
        """Verify cte."""
        self.assertEqual(
            sql(
                "WITH d AS (SELECT root FROM events WHERE delivered=1) SELECT count(*) FROM d",
                TABLES,
            )["rows"],
            [[2]],
        )

    def test_no_host_capabilities_writes_or_extensions(self):
        """Verify no host capabilities writes or extensions."""
        for query in (
            "ATTACH '/tmp/should-never-exist.db' AS other",
            "PRAGMA database_list",
            "DELETE FROM events",
            "UPDATE events SET tokens=0",
            "CREATE TABLE x(a)",
            "SELECT load_extension('evil')",
            "SELECT readfile('/etc/passwd')",
            "SELECT * FROM sqlite_master",
            "SELECT randomblob(100000000)",
            "SELECT 1; SELECT 2",
            "BEGIN",
        ):
            with self.subTest(query=query), self.assertRaises(ExactusError):
                sql(query, TABLES)

    def test_sql_progress_budget_interrupts_recursive_work(self):
        """Verify sql progress budget interrupts recursive work."""
        with self.assertRaisesRegex(ExactusError, "budget"):
            sql(
                "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT int_add(x,1) FROM n) SELECT sum(x) FROM n",
                TABLES,
            )

    def test_no_silent_truncation_or_float_exactness_claim(self):
        """Verify no silent truncation or float exactness claim."""
        for query in (
            "SELECT a.root FROM events a,events b,events c,events d,events e",
            "SELECT avg(tokens) FROM events",
        ):
            with self.subTest(query=query), self.assertRaises(ExactusError):
                sql(query, TABLES)

    def test_schema_validation(self):
        """Verify schema validation."""
        for tables in (
            {'bad"name': {"columns": ["x"], "rows": [[1]]}},
            {"x": {"columns": ["a", "A"], "rows": []}},
            {"x": {"columns": ["a"], "rows": [[2**64]]}},
            {"x": {"columns": ["a"], "rows": [[1, 2]]}},
        ):
            with self.subTest(tables=tables), self.assertRaises(ExactusError):
                sql("SELECT 1", tables)
