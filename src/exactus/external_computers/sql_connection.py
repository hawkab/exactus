"""Build read-only, request-local SQLite databases from explicit tables."""

__author__ = "Olshansky Grigory"

import sqlite3

from ..domain.errors import BudgetError, DataError
from .sql_values import _FUNCTIONS, _cell, _integer, _IntegerSum, _operation, identifier


def _prepare(tables):
    """Build an ephemeral database and install checked functions and read authorization."""
    import operator

    conn = sqlite3.connect(":memory:", cached_statements=0)
    try:
        for name, op in (
            ("int_add", operator.add),
            ("int_sub", operator.sub),
            ("int_mul", operator.mul),
            ("int_div", operator.floordiv),
            ("int_mod", operator.mod),
        ):
            conn.create_function(name, 2, _operation(op), deterministic=True)
        conn.create_function(
            "abs", 1, lambda value: _integer(abs(_integer(value))), deterministic=True
        )
        conn.create_aggregate("sum", 1, _IntegerSum)
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA trusted_schema=OFF")
        if hasattr(conn, "setlimit"):
            for category, limit in (
                (sqlite3.SQLITE_LIMIT_LENGTH, 65536),
                (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 8192),
                (sqlite3.SQLITE_LIMIT_COLUMN, 64),
                (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 32),
                (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 16),
                (sqlite3.SQLITE_LIMIT_VDBE_OP, 20000),
                (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 128),
                (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
            ):
                conn.setlimit(category, limit)
        total = 0
        for name, table in tables.items():
            qname = identifier(name)
            if type(table) is not dict or set(table) != {"columns", "rows"}:
                raise DataError("SQL table requires columns and rows only")
            columns, rows = table["columns"], table["rows"]
            if (
                type(columns) is not list
                or not 1 <= len(columns) <= 32
                or type(rows) is not list
            ):
                raise DataError("invalid SQL table shape")
            quoted = [identifier(c) for c in columns]
            if len({c.lower() for c in columns}) != len(columns):
                raise DataError("duplicate SQL columns")
            total += len(rows)
            if total > 2048:
                raise BudgetError("SQL input exceeds 2048 rows")
            conn.execute(f"CREATE TABLE {qname} ({','.join(quoted)})")
            for row in rows:
                if type(row) is not list or len(row) != len(columns):
                    raise DataError("SQL row width does not match columns")
                for cell in row:
                    _cell(cell)
            conn.executemany(
                f"INSERT INTO {qname} VALUES ({','.join('?' for _ in columns)})", rows
            )
        conn.commit()
        conn.execute("PRAGMA query_only=ON")
        names = frozenset(tables)

        def authorize(action, a, b, db, source):
            """Permit reads of supplied tables and allowlisted pure SQL functions."""
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                return sqlite3.SQLITE_OK
            if (
                action == sqlite3.SQLITE_READ
                and a in names
                and (db == "main" or (db is None and b == ""))
            ):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (b or "").lower() in _FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        conn.set_authorizer(authorize)
        return conn
    except BaseException:
        conn.close()
        raise
